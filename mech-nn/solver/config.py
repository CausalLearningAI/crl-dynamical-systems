from enum import Enum


class SolverType(Enum):
    DENSE_CHOLESKY = 1
    SPARSE_QR = 10
    SPARSE_INDIRECT_CG = 100
    SPARSE_INDIRECT_BLOCK_CG = 1000
    LEAST_SQUARES = 10000


class ODEConfig:
    # linear_solver = SolverType.DENSE_CHOLESKY

    # uncomment to choose linear solver sparse conjuate gradient
    # linear_solver = SolverType.SPARSE_INDIRECT_BLOCK_CG
    # cg_max_iter = 200

    linear_solver = SolverType.LEAST_SQUARES
