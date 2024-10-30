import itertools
import math

import torch


def ode_forward(
    coefficients: torch.Tensor,
    rhs_equation: torch.Tensor,
    rhs_init: torch.Tensor,
    steps: torch.Tensor,
    n_steps: int = None,
    n_init_var_steps: int = None,
) -> torch.Tensor:
    """
    coefficients: (..., n_steps[b], n_equations, n_dims, n_orders)
    rhs_equation: (..., n_steps[b], n_equations[e])
    rhs_init: (..., n_init_var_steps[b], n_dims[e], n_init_var_orders[e])
    steps: (..., n_steps-1[b])
    return: (..., n_steps, n_dims, n_orders)
    """

    dtype: torch.dtype = coefficients.dtype
    device: torch.device = coefficients.device

    n_steps: int = steps.size(-1) + 1 if n_steps is None else n_steps
    assert n_steps >= 2
    n_init_var_steps: int = rhs_init.size(-3) if n_init_var_steps is None else n_init_var_steps

    *batch_coefficients, n_steps_coefficients, n_equations, n_dims, n_orders = coefficients.shape
    assert n_steps_coefficients in [n_steps, 1]
    *batch_rhs_equation, n_steps_rhs_equation, n_equations_rhs_equation = rhs_equation.shape
    assert n_steps_rhs_equation in [n_steps, 1] and n_equations_rhs_equation == n_equations
    *batch_rhs_init, n_init_var_steps_rhs_init, n_dims_rhs_init, n_init_var_orders = rhs_init.shape
    assert n_init_var_steps_rhs_init in [n_init_var_steps, 1] and n_dims_rhs_init == n_dims
    *batch_steps, n_steps_steps = steps.shape
    assert n_steps_steps in [n_steps - 1, 1]
    batch_lhs: torch.Size = torch.broadcast_shapes(batch_coefficients, batch_steps)
    batch: torch.Size = torch.broadcast_shapes(batch_lhs, batch_rhs_equation, batch_rhs_init)

    # ode equation constraints
    c: torch.Tensor = coefficients.flatten(start_dim=-2)  # (..., n_steps[b], n_equations, n_dims * n_orders)
    ct: torch.Tensor = c.transpose(-2, -1)  # (..., n_steps[b], n_dims * n_orders, n_equations)
    block_diag_0: torch.Tensor = ct @ c  # (..., n_steps[b], n_dims * n_orders, n_dims * n_orders)
    beta: torch.Tensor = ct @ rhs_equation[..., None]  # (..., n_steps[b], n_dims * n_orders, 1)

    block_diag_0: torch.Tensor = block_diag_0.repeat(
        *[ss // s for ss, s in zip(batch_lhs, block_diag_0.shape[:-3])],
        n_steps // block_diag_0.size(-3),
        1,
        1,
    )  # (..., n_steps, n_dims * n_orders, n_dims * n_orders)
    beta: torch.Tensor = beta.repeat(
        *[ss // s for ss, s in zip(batch, beta.shape[:-3])],
        n_steps // beta.size(-3),
        1,
        1,
    )  # (..., n_steps, n_dims * n_orders, 1)

    # initial-value constraints
    init_idx: torch.Tensor = torch.arange(n_init_var_orders, device=device).repeat(n_dims) + n_orders * torch.arange(
        n_dims, device=device
    ).repeat_interleave(n_init_var_orders)
    # (n_dims * n_init_var_orders)
    block_diag_0[..., :n_init_var_steps, init_idx, init_idx] += 1.0
    beta[..., :n_init_var_steps, :, 0] += torch.cat(
        [
            rhs_init,
            torch.zeros(*rhs_init.shape[:-1], n_orders - n_init_var_orders, dtype=dtype, device=device),
        ],
        dim=-1,
    ).flatten(start_dim=-2)

    # smoothness constraints (forward & backward)
    order_idx: torch.Tensor = torch.arange(n_orders, device=device)  # (n_orders)
    sign_vec: torch.Tensor = order_idx % 2 * (-2) + 1  # (n_orders)
    sign_map: torch.Tensor = sign_vec * sign_vec[:, None]  # (n_orders, n_orders)

    expansions: torch.Tensor = steps[..., None] ** order_idx  # (..., n_steps-1[b], n_orders)
    et_e_diag: torch.Tensor = expansions**2  # (..., n_steps-1[b], n_orders)
    et_e_diag[..., -1] = 0.0
    factorials: torch.Tensor = (-(order_idx - order_idx[:, None] + 1).triu().to(dtype=dtype).lgamma()).exp()
    # (n_orders, n_orders)
    factorials[-1, -1] = 0.0
    e_outer: torch.Tensor = expansions[..., None] * expansions[..., None, :]  # (..., n_steps-1[b], n_orders, n_orders)
    et_ft_f_e: torch.Tensor = e_outer * (factorials.t() @ factorials)  # (..., n_steps-1[b], n_orders, n_orders)

    smooth_block_diag_1: torch.Tensor = e_outer * -(factorials + factorials.transpose(-2, -1) * sign_map)
    # (..., n_steps-1[b], n_orders, n_orders)
    smooth_block_diag_0: torch.Tensor = torch.zeros(*batch_lhs, n_steps, n_orders, n_orders, dtype=dtype, device=device)
    # (..., n_steps, n_orders, n_orders)
    smooth_block_diag_0[..., :-1, :, :] += et_ft_f_e
    smooth_block_diag_0[..., 1:, :, :] += et_ft_f_e * sign_map
    smooth_block_diag_0[..., :-1, order_idx, order_idx] += et_e_diag
    smooth_block_diag_0[..., 1:, order_idx, order_idx] += et_e_diag

    smooth_block_diag_1: torch.Tensor = smooth_block_diag_1.repeat(
        *([1] * len(batch_lhs)),
        (n_steps - 1) // smooth_block_diag_1.size(-3),
        1,
        1,
    )  # (..., n_steps-1, n_orders, n_orders)
    steps: torch.Tensor = steps.repeat(*([1] * len(batch_lhs)), (n_steps - 1) // steps.size(-1))  # (..., n_steps-1)

    # smoothness constraints (central)
    steps2: torch.Tensor = steps[..., :-1] + steps[..., 1:]  # (..., n_steps-2)
    steps26: torch.Tensor = steps2 ** (n_orders * 2 - 6)  # (..., n_steps-2)
    steps25: torch.Tensor = steps2 ** (n_orders * 2 - 5)  # (..., n_steps-2)
    steps24: torch.Tensor = steps2 ** (n_orders * 2 - 4)  # (..., n_steps-2)

    smooth_block_diag_0[..., :-2, n_orders - 2, n_orders - 2] += steps26
    smooth_block_diag_0[..., 2:, n_orders - 2, n_orders - 2] += steps26
    smooth_block_diag_0[..., 1:-1, n_orders - 1, n_orders - 1] += steps24
    smooth_block_diag_1[..., :-1, n_orders - 1, n_orders - 2] += steps25
    smooth_block_diag_1[..., 1:, n_orders - 2, n_orders - 1] -= steps25
    smooth_block_diag_2: torch.Tensor = torch.zeros(
        *batch_lhs, n_steps - 2, n_orders, n_orders, dtype=dtype, device=device
    )
    # (..., n_steps-2, n_orders, n_orders)
    smooth_block_diag_2[..., n_orders - 2, n_orders - 2] = -steps26

    # copy to n_dims
    block_diag_1: torch.Tensor = torch.zeros(
        *batch_lhs, n_steps - 1, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device
    )
    # (..., n_steps-1, n_dims * n_orders, n_dims * n_orders)
    block_diag_2: torch.Tensor = torch.zeros(
        *batch_lhs, n_steps - 2, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device
    )
    # (..., n_steps-2, n_dims * n_orders, n_dims * n_orders)
    for dim in range(n_dims):
        i1: int = dim * n_orders
        i2: int = (dim + 1) * n_orders
        block_diag_0[..., i1:i2, i1:i2] += smooth_block_diag_0
        block_diag_1[..., i1:i2, i1:i2] = smooth_block_diag_1
        block_diag_2[..., i1:i2, i1:i2] = smooth_block_diag_2

    # blocked cholesky decomposition
    block_diag_0_list: list[torch.Tensor] = list(block_diag_0.unbind(dim=-3))
    block_diag_1_list: list[torch.Tensor] = list(block_diag_1.unbind(dim=-3))
    block_diag_2_list: list[torch.Tensor] = list(block_diag_2.unbind(dim=-3))
    for step in range(n_steps):
        if step >= 2:
            block_diag_2_list[step - 2] = torch.linalg.solve_triangular(
                block_diag_0_list[step - 2].transpose(-2, -1),
                block_diag_2_list[step - 2],
                upper=True,
                left=False,
            )
            block_diag_1_list[step - 1] = block_diag_1_list[step - 1] - block_diag_2_list[step - 2] @ block_diag_1_list[
                step - 2
            ].transpose(-2, -1)
        if step >= 1:
            block_diag_1_list[step - 1] = torch.linalg.solve_triangular(
                block_diag_0_list[step - 1].transpose(-2, -1),
                block_diag_1_list[step - 1],
                upper=True,
                left=False,
            )
            if step >= 2:
                block_diag_0_list[step] = block_diag_0_list[step] - block_diag_2_list[step - 2] @ block_diag_2_list[
                    step - 2
                ].transpose(-2, -1)
            block_diag_0_list[step] = block_diag_0_list[step] - block_diag_1_list[step - 1] @ block_diag_1_list[
                step - 1
            ].transpose(-2, -1)
        block_diag_0_list[step], _ = torch.linalg.cholesky_ex(
            block_diag_0_list[step],
            upper=False,
            check_errors=False,
        )

    # A X = B => L (Lt X) = B
    # solve L Y = B, block forward substitution
    b_list: list[torch.Tensor] = list(beta.unbind(dim=-3))
    y_list: list[torch.Tensor | None] = [None] * n_steps
    for step in range(n_steps):
        b_step: torch.Tensor = b_list[step]
        if step >= 2:
            b_step = b_step - block_diag_2_list[step - 2] @ y_list[step - 2]
        if step >= 1:
            b_step = b_step - block_diag_1_list[step - 1] @ y_list[step - 1]
        y_list[step] = torch.linalg.solve_triangular(
            block_diag_0_list[step],
            b_step,
            upper=False,
            left=True,
        )

    # solve Lt X = Y, block backward substitution
    x_list: list[torch.Tensor | None] = [None] * n_steps
    for step in range(n_steps - 1, -1, -1):
        y_step: torch.Tensor = y_list[step]
        if step < n_steps - 2:
            y_step = y_step - block_diag_2_list[step].transpose(-2, -1) @ x_list[step + 2]
        if step < n_steps - 1:
            y_step = y_step - block_diag_1_list[step].transpose(-2, -1) @ x_list[step + 1]
        x_list[step] = torch.linalg.solve_triangular(
            block_diag_0_list[step].transpose(-2, -1),
            y_step,
            upper=True,
            left=True,
        )

    u: torch.Tensor = torch.stack(x_list, dim=-3).reshape(*batch, n_steps, n_dims, n_orders)
    # (..., n_steps, n_dims, n_orders)
    return u


def ode_forward_baseline(
    coefficients: torch.Tensor,
    rhs_equation: torch.Tensor,
    rhs_init: torch.Tensor,
    steps: torch.Tensor,
) -> torch.Tensor:
    """
    coefficients: (..., n_steps, n_equations, n_dims, n_orders)
    rhs_equation: (..., n_steps, n_equations)
    rhs_init: (..., n_init_var_steps, n_dims, n_init_var_orders)
    steps: (..., n_steps-1)
    return: (..., n_steps, n_dims, n_orders)
    """
    dtype: torch.dtype = coefficients.dtype
    device: torch.device = coefficients.device

    *batches, n_steps, n_equations, n_dims, n_orders = coefficients.shape
    *_, n_init_var_steps, _, n_init_var_orders = rhs_init.shape

    A_eq = torch.zeros(*batches, n_steps * n_equations, n_steps * n_dims * n_orders, dtype=dtype, device=device)
    for i, (step, equation) in enumerate(itertools.product(range(n_steps), range(n_equations))):
        A_eq[..., i, step * n_dims * n_orders : (step + 1) * n_dims * n_orders] = coefficients[
            ..., step, equation, :, :
        ].flatten(start_dim=-2)
    beta_eq = rhs_equation.flatten(start_dim=-2)

    A_in = torch.zeros(
        *batches, n_init_var_steps * n_dims * n_init_var_orders, n_steps * n_dims * n_orders, dtype=dtype, device=device
    )
    for i, (step, dim, order) in enumerate(
        itertools.product(range(n_init_var_steps), range(n_dims), range(n_init_var_orders))
    ):
        A_in[..., i, (step * n_dims + dim) * n_orders + order] = 1.0
    beta_in = rhs_init.flatten(start_dim=-3)

    A_sf = torch.zeros(
        *batches, (n_steps - 1) * n_dims * (n_orders - 1), n_steps * n_dims * n_orders, dtype=dtype, device=device
    )
    for i, (step, dim, order) in enumerate(itertools.product(range(n_steps - 1), range(n_dims), range(n_orders - 1))):
        for o in range(order, n_orders):
            A_sf[..., i, (step * n_dims + dim) * n_orders + o] = steps[..., step] ** o / math.factorial(o - order)
        A_sf[..., i, ((step + 1) * n_dims + dim) * n_orders + order] = -steps[..., step] ** order

    A_sb = torch.zeros(
        *batches, (n_steps - 1) * n_dims * (n_orders - 1), n_steps * n_dims * n_orders, dtype=dtype, device=device
    )
    for i, (step, dim, order) in enumerate(itertools.product(range(n_steps - 1), range(n_dims), range(n_orders - 1))):
        for o in range(order, n_orders):
            A_sb[..., i, ((step + 1) * n_dims + dim) * n_orders + o] = (-steps[..., step]) ** o / math.factorial(
                o - order
            )
        A_sb[..., i, (step * n_dims + dim) * n_orders + order] = -((-steps[..., step]) ** order)

    A_sc = torch.zeros(*batches, (n_steps - 2) * n_dims, n_steps * n_dims * n_orders, dtype=dtype, device=device)
    for i, (step, dim) in enumerate(itertools.product(range(n_steps - 2), range(n_dims))):
        A_sc[..., i, (step * n_dims + dim) * n_orders + (n_orders - 2)] = (steps[..., step] + steps[..., step + 1]) ** (
            n_orders - 3
        )
        A_sc[..., i, ((step + 1) * n_dims + dim) * n_orders + (n_orders - 1)] = (
            steps[..., step] + steps[..., step + 1]
        ) ** (n_orders - 2)
        A_sc[..., i, ((step + 2) * n_dims + dim) * n_orders + (n_orders - 2)] = -(
            (steps[..., step] + steps[..., step + 1]) ** (n_orders - 3)
        )

    A = torch.cat([A_eq, A_in, A_sb, A_sc, A_sf], dim=-2)
    beta = torch.cat(
        [
            beta_eq,
            beta_in,
            torch.zeros_like(A_sf[..., 0]),
            torch.zeros_like(A_sc[..., 0]),
            torch.zeros_like(A_sb[..., 0]),
        ],
        dim=-1,
    )

    AtA = A.transpose(-2, -1) @ A
    Atb = A.transpose(-2, -1) @ beta[..., None]

    L, info = torch.linalg.cholesky_ex(AtA, upper=False, check_errors=False)
    u = Atb.cholesky_solve(L, upper=False)
    u = u.reshape(*batches, n_steps, n_dims, n_orders)
    return u


def test():
    torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    dtype = torch.float64
    device = torch.device("cuda:0")
    batches = (11,)
    n_steps, n_equations, n_dims, n_orders = 7, 2, 3, 5
    n_init_var_steps, n_init_var_orders = 3, 4
    coefficients = torch.nn.Parameter(
        torch.randn(*batches, n_steps, n_equations, n_dims, n_orders, dtype=dtype, device=device)
    )
    rhs_equation = torch.nn.Parameter(torch.randn(*batches, n_steps, n_equations, dtype=dtype, device=device))
    rhs_init = torch.nn.Parameter(
        torch.randn(*batches, n_init_var_steps, n_dims, n_init_var_orders, dtype=dtype, device=device)
    )
    steps = torch.nn.Parameter(torch.rand(*batches, n_steps - 1, dtype=dtype, device=device))
    u = ode_forward(coefficients, rhs_equation, rhs_init, steps)
    u0 = ode_forward_baseline(coefficients, rhs_equation, rhs_init, steps)
    diff = u - u0
    print(diff.abs().max().item())
    u.sum().backward()
    u0.sum().backward()
    u = None


if __name__ == "__main__":
    test()
