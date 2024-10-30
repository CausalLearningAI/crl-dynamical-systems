import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Constants
# g = 5. #9.8  # acceleration due to gravity, in m/s^2
# m_c = 1.0  # mass of the cart, in kg
# m_p = 0.1  # mass of the pole, in kg
# l = 0.5  # half-length of the pole, in m
# force = 0.0  # external force applied to the cart


def cart_pole_ode(y, g, m_c, m_p, l, force=0):
    if y.ndim == 1:
        x_dot = y[1]
        theta = y[2]
        theta_dot = y[3]
    else:
        x_dot = y[:, 1]
        theta = y[:, 2]
        theta_dot = y[:, 3]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_dot_sq = theta_dot**2

    common_denominator = m_c + m_p * sin_theta**2

    theta_ddot = (g * sin_theta - cos_theta * ((force + m_p * l * theta_dot_sq * sin_theta) / common_denominator)) / (
        l * (4 / 3 - m_p * cos_theta**2 / common_denominator)
    )
    x_ddot = (force + m_p * l * (theta_dot_sq * sin_theta - theta_ddot * cos_theta)) / common_denominator

    if y.ndim == 1:
        derivatives = np.array([x_dot, x_ddot, theta_dot, theta_ddot])
    else:
        derivatives = np.stack([x_dot, x_ddot, theta_dot, theta_ddot], axis=1)
    return derivatives


# Initial conditions
y0 = [0.0, 0.0, np.pi / 2, 0.0]  # [x, x_dot, theta, theta_dot]
N = 1000
# Time span
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

lower_bounds = [9.7, 1.0, 0.1, 0.1]
upper_bounds = [9.9, 1.2, 0.3, 1.6]

gs = np.random.uniform(lower_bounds[0], upper_bounds[0], N)
m_cs = np.random.uniform(lower_bounds[1], upper_bounds[1], N)
m_ps = np.random.uniform(lower_bounds[2], upper_bounds[2], N)
ls = np.random.uniform(lower_bounds[3], upper_bounds[3], N)

est_params = []

for g, m_c, m_p, l in zip(gs, m_cs, m_ps, ls):
    # Solve ODE
    sol = solve_ivp(lambda t, y: cart_pole_ode(y, g, m_c, m_p, l), t_span, y0, t_eval=t_eval, rtol=1e-8)
    ys = sol.y.T
    ts = sol.t

    target = np.gradient(ys, ts, axis=0)

    def residuals(params, input, target):
        return (cart_pole_ode(input, *params) - target).ravel()

    bounds = (lower_bounds, upper_bounds)
    res = least_squares(residuals, lower_bounds, args=(ys, target), bounds=bounds)
    est_params += [res.x.tolist()]

est_params = np.array(est_params).reshape(-1, 4)
print(
    "error",
    np.square(est_params - np.array([gs, m_cs, m_ps, ls]).T).mean(axis=0),
    np.square(est_params - np.array([gs, m_cs, m_ps, ls]).T).std(axis=0),
)
# opt_params = res.x
# print("residual error", residuals)
