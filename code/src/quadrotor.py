from functools import partial
from abc import ABC, abstractmethod
from typing import Tuple, NamedTuple, Type
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
from jax import jit, vmap
from jax.lax import cond
from trajax import optimizers

def std_error(arr: Array, axis: int) -> Array:
    return jnp.std(arr, axis=axis) / jnp.sqrt(arr.shape[axis])

class ModelFeedBackParams(NamedTuple):
    proportional_params: Float[Array, "4"] = jnp.zeros(4)
    derivative_params: Float[Array, "4"] = jnp.zeros(4)

def critical_damping(proportional_params: Float[Array, "u_dim"]) -> ModelFeedBackParams:
    derivative_params = 2 * jnp.sqrt(proportional_params)
    return ModelFeedBackParams(
        proportional_params=proportional_params, derivative_params=derivative_params
    )

class OptimalCost(ABC):
    x_dim: int
    u_dim: int
    num_nodes: int
    time_horizon: Tuple[float, float]
    dt: float
    ts: Float[Array, "N"]
    ilqr_params: optimizers.ILQRHyperparams
    ilqr: optimizers.ILQR

    def __init__(
        self,
        time_horizon: Tuple[float, float],
        num_nodes: int = 50,
        sim_params: dict = {},
    ):
        self.num_nodes = num_nodes
        self.time_horizon = time_horizon

        self.dt = (self.time_horizon[1] - self.time_horizon[0]) / num_nodes
        self.ts = jnp.linspace(
            self.time_horizon[0], self.time_horizon[1], num_nodes + 1
        )
        self.ilqr_params = optimizers.ILQRHyperparams(
            maxiter=1000, make_psd=False, psd_delta=1e0
        )
        self.ilqr = optimizers.ILQR(self.cost, self.next_step)

    @abstractmethod
    def cost(
        self,
        x: Float[Array, "x_dim"],
        u: Float[Array, "u_dim"],
        t: float,
        params: dict | None = None,
    ):
        pass

    @abstractmethod
    def next_step(
        self,
        x: Float[Array, "x_dim"],
        u: Float[Array, "u_dim"],
        t: float,
        params: dict | None = None,
    ) -> Float[Array, "x_dim"]:
        pass

    @partial(jit, static_argnums=0)
    def solve(self, x0: Float[Array, "x_dim"]) -> optimizers.ILQRResult:
        initial_U = jnp.zeros(
            shape=(
                self.num_nodes,
                self.u_dim,
            )
        )
        out = self.ilqr.solve(
            cost_params=None,
            dynamics_params=None,
            x0=x0,
            U=initial_U,
            hyperparams=self.ilqr_params,
        )
        return out

    @partial(jit, static_argnums=0)
    def evaluate(
        self,
        X: Float[Array, "N x_dim"],
        U: Float[Array, "N u_dim"],
    ) -> float:
        last_row = U[-1:]
        U_repeated = jnp.concatenate([U, last_row], axis=0)
        cost = optimizers.evaluate(cost=self.cost, X=X, U=U_repeated)
        return jnp.sum(cost)

    @partial(jit, static_argnums=0)
    def rollout(
        self,
        x0: Float[Array, "x_dim"],
        U: Float[Array, "N u_dim"],
        disturbance_params: ModelFeedBackParams,
        feedback_params: ModelFeedBackParams,
    ) -> Tuple[float, Float[Array, "N x_dim"]]:
        dynamics_params = {
            "disturbance_params": disturbance_params,
            "feedback_params": feedback_params,
        }
        X = self.ilqr._rollout(dynamics_params=dynamics_params, U=U, x0=x0)
        cost = self.evaluate(X=X, U=U)
        return cost, X

    @classmethod
    @abstractmethod
    def _constraint(cls, x: Float[Array, "x_dim"]) -> float:
        pass

    @classmethod
    @partial(jit, static_argnums=0)
    def constraint(cls, X: Float[Array, "N x_dim"]) -> float:
        fs = vmap(cls._constraint)(X)
        return jnp.min(fs)


class Simulator:
    x0: Float[Array, "x_dim"]
    disturbance_params: ModelFeedBackParams
    oc: OptimalCost
    U: Float[Array, "N u_dim"]

    def __init__(
        self,
        OC: Type[OptimalCost],
        x0: Float[Array, "x_dim"],
        state_target: Float[Array, "x_dim"],
        action_target: Float[Array, "u_dim"],
        disturbance_params: ModelFeedBackParams,
        T=3,
        N=100,
    ):
        self.x0 = x0
        self.disturbance_params = disturbance_params

        sim_params = {
            "state_target": state_target,
            "action_target": action_target,
        }
        self.oc = OC(time_horizon=(0, T), num_nodes=N, sim_params=sim_params)
        out = self.oc.solve(x0)
        self.U = out.us

    @partial(jit, static_argnums=0)
    def rollout(
        self, feedback_params: ModelFeedBackParams
    ) -> Tuple[float, Float[Array, "N x_dim"]]:
        return self.oc.rollout(
            x0=self.x0,
            U=self.U,
            disturbance_params=self.disturbance_params,
            feedback_params=feedback_params,
        )

    @classmethod
    def plot_feedback_params_safety(
        cls,
        OC: Type[OptimalCost],
        name: str,
        x0: Float[Array, "x_dim"],
        state_target: Float[Array, "x_dim"],
        action_target: Float[Array, "u_dim"],
        fmin: float,
        disturbance_scale: float,
        n_samples=100,
        P=jnp.linspace(0.0, 100.0, 100),
        ylabel="Constraint",
    ):
        def engine(k_p: float, seed: ScalarInt) -> float:
            key = jr.PRNGKey(seed=seed)
            disturbance_params = sample_disturbance_params(
                key=key, disturbance_scale=disturbance_scale
            )
            feedback_params = critical_damping(
                proportional_params=jnp.full((OC.u_dim,), k_p)
            )
            cost, X = cls(
                OC=OC,
                x0=x0,
                disturbance_params=disturbance_params,
                state_target=state_target,
                action_target=action_target,
            ).rollout(feedback_params)
            return OC.constraint(X)

        results = vmap(
            lambda k_p: vmap(lambda seed: engine(k_p, seed))(jnp.arange(n_samples))
        )(P)
        mean = jnp.mean(results, axis=1)
        std = std_error(results, axis=1)  # type: ignore

        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])

        ax.set_xlabel("$k_p$")
        ax.set_ylabel(ylabel)
        ax.axhline(y=fmin, linestyle="--", color="black")  # type: ignore
        ax.plot(P, mean, color="blue")
        ax.fill_between(P, mean - std, mean + std, color="blue", alpha=0.3)  # type: ignore
        store_and_show_fig("", fig, f"robotics/{name}", "Safety per k_p")
        plt.close()


def sample_disturbance_params(
    key: jax.Array, disturbance_scale: float
) -> ModelFeedBackParams:
    proportional_params = disturbance_scale * jnp.square(jr.normal(key, shape=(4,)))
    disturbance_params = critical_damping(proportional_params)
    return disturbance_params

def euler_to_rotation(angles: Float[Array, "3"]) -> Float[Array, "3 3"]:
    phi, theta, psi = angles
    first_row = jnp.array(
        [
            jnp.cos(psi) * jnp.cos(theta)
            - jnp.sin(phi) * jnp.sin(psi) * jnp.sin(theta),
            -jnp.cos(phi) * jnp.sin(psi),
            jnp.cos(psi) * jnp.sin(theta)
            + jnp.cos(theta) * jnp.sin(phi) * jnp.sin(psi),
        ]
    )
    second_row = jnp.array(
        [
            jnp.cos(theta) * jnp.sin(psi)
            + jnp.cos(psi) * jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(phi) * jnp.cos(psi),
            jnp.sin(psi) * jnp.sin(theta)
            - jnp.cos(psi) * jnp.cos(theta) * jnp.sin(phi),
        ]
    )
    third_row = jnp.array(
        [-jnp.cos(phi) * jnp.sin(theta), jnp.sin(phi), jnp.cos(phi) * jnp.cos(theta)]
    )
    return jnp.stack([first_row, second_row, third_row])


def move_frame(angles: Float[Array, "3"]) -> Float[Array, "3 3"]:
    phi, theta, psi = angles
    first_row = jnp.array([jnp.cos(theta), 0, -jnp.cos(phi) * jnp.sin(theta)])
    second_row = jnp.array([0, 1, jnp.sin(phi)])
    third_row = jnp.array([jnp.sin(theta), 0, jnp.cos(phi) * jnp.cos(theta)])
    return jnp.stack([first_row, second_row, third_row])


def quadratic_cost(
    x: Float[Array, "12"],
    u: Float[Array, "4"],
    x_target: Float[Array, "12"],
    u_target: Float[Array, "4"],
    q: Float[Array, "12 12"],
    r: Float[Array, "4 4"],
) -> float:
    assert (
        x.ndim == u.ndim == 1
        and q.ndim == r.ndim == 2
        and x.shape[0] == q.shape[0]
        and u.shape[0] == r.shape[0]
    )
    norm_x = x - x_target
    norm_u = u - u_target
    return norm_x @ q @ norm_x + norm_u @ r @ norm_u


ACTION_TARGET = jnp.array([1.766, 0.0, 0.0, 0.0])
"""Action which results in hovering (approximately) when in state 0 with undisturbed dynamics."""

class QuadrotorEuler:
    """
    Dynamics of quadrotor with 12 dimensional state space and 4 dimensional control
    Why 4 dimensional control: https://www.youtube.com/watch?v=UC8W3SfKGmg (talks about that at around 8min in video)
    Short description for 4 dimensional control:
          [ F  ]         [ F1 ]
          | M1 |  = A *  | F2 |
          | M2 |         | F3 |
          [ M3 ]         [ F4 ]
    """

    x_dim = 12
    u_dim = 4

    time_scaling: Float[Array, "1"]
    state_scaling: Float[Array, "12 12"]
    state_scaling_inv: Float[Array, "12 12"]
    control_scaling: Float[Array, "4 4"]
    control_scaling_inv: Float[Array, "4 4"]
    mass: float
    g: float
    arm_length: float
    height: float
    I: Float[Array, "3 3"]
    invI: Float[Array, "3 3"]
    minF: float
    maxF: float
    km: float
    kf: float
    r: float
    L: float
    H: float
    A: Float[Array, "4 4"]
    invA: Float[Array, "4 4"]
    body_shape: Float[Array, "6 4"]
    B: Float[Array, "2 4"]
    internal_control_scaling_inv: Float[Array, "4 4"]
    state_target: Float[Array, "12"]
    action_target: Float[Array, "4"]
    running_q: Float[Array, "12 12"]
    running_r: Float[Array, "4 4"]
    terminal_q: Float[Array, "12 12"]
    terminal_r: Float[Array, "4 4"]

    def __init__(
        self,
        state_target: Float[Array, "12"],
        action_target: Float[Array, "4"],
        time_scaling: Float[Array, "1"] | None = None,
        state_scaling: Float[Array, "12 12"] | None = None,
        control_scaling: Float[Array, "4 4"] | None = None,
    ):
        self.x_dim = 12
        self.u_dim = 4
        if time_scaling is None:
            time_scaling = jnp.ones(shape=(1,))
        if state_scaling is None:
            state_scaling = jnp.eye(self.x_dim)
        if control_scaling is None:
            control_scaling = jnp.eye(self.u_dim)
        self.time_scaling = time_scaling
        self.state_scaling = state_scaling
        self.state_scaling_inv = jnp.linalg.inv(state_scaling)
        self.control_scaling = control_scaling
        self.control_scaling_inv = jnp.linalg.inv(control_scaling)

        self.mass = 0.18  # kg
        self.g = 9.81  # m/s^2
        self.arm_length = 0.086  # meter
        self.height = 0.05

        self.I = jnp.array(
            [(0.00025, 0, 2.55e-6), (0, 0.000232, 0), (2.55e-6, 0, 0.0003738)]
        )

        self.invI = jnp.linalg.inv(self.I)

        self.minF = 0.0
        self.maxF = 2.0 * self.mass * self.g

        self.km = 1.5e-9
        self.kf = 6.11e-8
        self.r = self.km / self.kf

        self.L = self.arm_length
        self.H = self.height
        #  [ F  ]         [ F1 ]
        #  | M1 |  = A *  | F2 |
        #  | M2 |         | F3 |
        #  [ M3 ]         [ F4 ]
        self.A = jnp.array(
            [
                [1, 1, 1, 1],
                [0, self.L, 0, -self.L],
                [-self.L, 0, self.L, 0],
                [self.r, -self.r, self.r, -self.r],
            ]
        )

        self.invA = jnp.linalg.inv(self.A)

        self.body_frame = jnp.array(
            [
                (self.L, 0, 0, 1),
                (0, self.L, 0, 1),
                (-self.L, 0, 0, 1),
                (0, -self.L, 0, 1),
                (0, 0, 0, 1),
                (0, 0, self.H, 1),
            ]
        )

        self.B = jnp.array([[0, self.L, 0, -self.L], [-self.L, 0, self.L, 0]])

        self.internal_control_scaling_inv = jnp.diag(
            jnp.array([1, 2 * 1e-4, 2 * 1e-4, 1e-3], dtype=jnp.float64)
        )

        # Cost parameters:
        self.state_target = state_target
        self.action_target = action_target
        self.running_q = 1.0 * jnp.diag(
            jnp.array([1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 1, 0.1, 0.1], dtype=jnp.float64)
        )
        self.running_r = (
            1e-2 * 1.0 * jnp.diag(jnp.array([5.0, 0.8, 0.8, 0.3], dtype=jnp.float64))
        )
        self.terminal_q = 5.0 * jnp.eye(self.x_dim)
        self.terminal_r = 0.0 * jnp.eye(self.u_dim)

    @partial(jit, static_argnums=0)
    def _ode(
        self, state: Float[Array, "12"], u: Float[Array, "4"]
    ) -> Float[Array, "12"]:
        # u = self.scaling_u_inv @ u
        # Here we have to decide in which coordinate system we will operate
        # If we operate with F1, F2, F3 and F4 we need to run
        # u = self.A @ u
        u = self.internal_control_scaling_inv @ u
        F, M = u[0], u[1:]
        x, y, z, xdot, ydot, zdot, phi, theta, psi, p, q, r = state
        angles = jnp.array([phi, theta, psi])
        wRb = euler_to_rotation(angles)
        # acceleration - Newton's second law of motion
        accel = (
            1.0
            / self.mass
            * (
                wRb.dot(jnp.array([[0, 0, F]]).T)
                - jnp.array([[0, 0, self.mass * self.g]]).T
            )
        )
        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = jnp.array([p, q, r])
        angles_dot = jnp.linalg.inv(move_frame(angles)) @ omega
        pqrdot = self.invI.dot(M.flatten() - jnp.cross(omega, self.I.dot(omega)))
        state_dot_0 = xdot
        state_dot_1 = ydot
        state_dot_2 = zdot
        state_dot_3 = accel[0].reshape()
        state_dot_4 = accel[1].reshape()
        state_dot_5 = accel[2].reshape()
        state_dot_6 = angles_dot[0]
        state_dot_7 = angles_dot[1]
        state_dot_8 = angles_dot[2]
        state_dot_9 = pqrdot[0]
        state_dot_10 = pqrdot[1]
        state_dot_11 = pqrdot[2]
        return jnp.array(
            [
                state_dot_0,
                state_dot_1,
                state_dot_2,
                state_dot_3,
                state_dot_4,
                state_dot_5,
                state_dot_6,
                state_dot_7,
                state_dot_8,
                state_dot_9,
                state_dot_10,
                state_dot_11,
            ]
        )

    @partial(jit, static_argnums=0)
    def ode(
        self,
        x: Float[Array, "12"],
        u: Float[Array, "4"],
        disturbance_params: ModelFeedBackParams = ModelFeedBackParams(),
        feedback_params: ModelFeedBackParams = ModelFeedBackParams(),
    ) -> Float[Array, "12"]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        position_diff = jnp.concatenate(
            [(x[2] - self.state_target[2]).reshape(1), x[6:9] - self.state_target[6:9]]
        )
        velocity_diff = jnp.concatenate(
            [
                (x[5] - self.state_target[5]).reshape(1),
                x[9:12] - self.state_target[9:12],
            ]
        )
        disturbance = (
            disturbance_params.proportional_params - feedback_params.proportional_params
        ) * position_diff + (
            disturbance_params.derivative_params - feedback_params.derivative_params
        ) * velocity_diff
        return (
            self.state_scaling
            @ self._ode(x, u + disturbance)
            / self.time_scaling.reshape()
        )

    @partial(jit, static_argnums=0)
    def running_cost(self, x: Float[Array, "12"], u: Float[Array, "4"]) -> float:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._running_cost(x, u) / self.time_scaling.reshape()

    @partial(jit, static_argnums=0)
    def terminal_cost(self, x: Float[Array, "12"], u: Float[Array, "4"]) -> float:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._terminal_cost(x, u)

    @partial(jit, static_argnums=0)
    def _running_cost(self, x: Float[Array, "12"], u: Float[Array, "4"]) -> float:
        return quadratic_cost(
            x,
            u,
            x_target=self.state_target,
            u_target=self.action_target,
            q=self.running_q,
            r=self.running_r,
        )

    @partial(jit, static_argnums=0)
    def _terminal_cost(
        self, x: Float[Array, "12"], u: Float[Array, "4"]
    ) -> float:
        return quadratic_cost(
            x,
            u,
            x_target=self.state_target,
            u_target=self.action_target,
            q=self.terminal_q,
            r=self.terminal_r,
        )

class QuadrotorOptimalCost(OptimalCost):
    x_dim = 12
    u_dim = 4
    dynamics: QuadrotorEuler

    def __init__(self, sim_params: dict = {}, **kwargs):
        self.dynamics = QuadrotorEuler(**sim_params)
        super().__init__(**kwargs)

    @partial(jit, static_argnums=0)
    def running_cost(
        self, x: Float[Array, "12"], u: Float[Array, "4"], t: float
    ) -> float:
        return self.dt * self.dynamics.running_cost(x, u)

    @partial(jit, static_argnums=0)
    def terminal_cost(
        self, x: Float[Array, "12"], u: Float[Array, "4"], t: float
    ) -> float:
        return self.dynamics.terminal_cost(x, u)

    @partial(jit, static_argnums=0)
    def cost(
        self,
        x: Float[Array, "12"],
        u: Float[Array, "4"],
        t: float,
        params: dict | None = None,
    ):
        return cond(
            t == self.num_nodes,
            self.terminal_cost,
            self.running_cost,
            x,
            u,
            t,
        )

    @partial(jit, static_argnums=0)
    def next_step(
        self,
        x: Float[Array, "12"],
        u: Float[Array, "4"],
        t: float,
        params: dict | None = None,
    ) -> Float[Array, "12"]:
        dynamics_params = {}
        if params is not None:
            for key in ["disturbance_params", "feedback_params"]:
                if key in params.keys():
                    dynamics_params[key] = params[key]
        return x + self.dt * self.dynamics.ode(x, u, **dynamics_params)

    @classmethod
    @partial(jit, static_argnums=0)
    def _constraint(cls, x: Float[Array, "x_dim"]) -> float:
        return jnp.maximum(0, jnp.nan_to_num(x[2], nan=0.0))

def get_oracle(random_key:jax.Array):
    disturbance_params = sample_disturbance_params(key=random_key, disturbance_scale=1.0) # generate ground truth
    #safe_disturbance_params = jnp.array([0.0, 0.0, 0.0, 10.0]) # safe disturbance params

    X0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    STATE_TARGET = jnp.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    COST_THRESHOLD = 200.0
    sim = Simulator(
        OC=QuadrotorOptimalCost,
        x0=X0,
        disturbance_params=disturbance_params,
        state_target=STATE_TARGET,
        action_target=ACTION_TARGET,
    )

    @jit
    def smoothed_objective(cost: float) -> float:
        """Transforms cost into smoothed objective, ensuring that the objective value is upper bounded by $1$ and lower bounded by $-1$ for costs below `COST_THRESHOLD`."""
        hct = COST_THRESHOLD / 2
        normalized_cost = (cost - hct) / hct
        smoothed_cost = jnp.tanh(normalized_cost)
        return -smoothed_cost

    @jit
    def f_oracle(x: Float[Array, "d"]) -> Float[Array, "2"]:
        feedback_params = critical_damping(proportional_params=x)
        cost, X = sim.rollout(feedback_params)
        objective = smoothed_objective(jnp.nan_to_num(cost, nan=jnp.inf))
        return objective

    return f_oracle

def generate_domain(random_key:jax.Array, domain_cardinality:int):
    return jr.uniform(random_key, (domain_cardinality, 4), minval=0.0, maxval=20.0)

def get_domain_and_f(random_key:jax.Array, domain_cardinality:int):
    key1, key2 = jr.split(random_key)
    domain = generate_domain(key1, domain_cardinality)
    oracle = get_oracle(key2)
    vmapped_oracle = vmap(oracle, in_axes=0, out_axes=0)
    f = vmapped_oracle(domain)
    return domain, f

if __name__ == '__main__':
    key = jr.key(0)
    domain, f = get_domain_and_f(key, 400)

    import matplotlib.pyplot as plt
    plt.hist(f, density=True, bins=50)
    plt.savefig(f'quadrotor histogram.pdf', format='pdf')

# kernel parameters
LENGTHSCALE = 1e-1
NOISE_STD = 1e-1