import numpy as np

"""
1) LINEARIZED SYSTEM

Assumptions:
    - y_e = 0 (xz plane)
    - constant attitude
"""


def compute_M(F, r_eq, h=1e-6):
    """
    Inputs:
        - F = r_dotdot + 2.Omega x r_dot = grad(U) + a_srp
        - r_eq : equilibrium point
        - h : step
    Returns:
        - M : Jacobian matrix
    """

    M = np.zeros((3, 3))

    # build a column of M for each iteration
    for j in range(3):
        dr = np.zeros(3)
        dr[j] = h
        F_plus, F_minus = F(r_eq + dr), F(r_eq - dr)
        M[:, j] = (F_plus - F_minus) / (2 * h)

    return M

def build_A(M):
    Z = np.zeros([3,3])
    I = np.eye(3)
    Theta = np.array([[0,2,0],[-2,0,0],[0,0,0]])
    A = np.block([[Z,  I  ],
                  [M,Theta]])
    return A

def find_eigmodes(A, tol=1e-10):
    """
    return:
        imag_modes = [(l1,v1),(l2,v2),...], where li = 0+aj
        real_modes = [(l1,v1),(l2,v2),...], where li = a+0j
    observations: 
        1. for the imag_modes, return only the positive ones 
        2. ascending order in lambda
    Example: 
        if the eigenvalues of A are [+1,-1,+2j,-2j,+3,-3],
        it will return [(2,v)], [(-3,v),(-1,v),(1,v),(3,v)], 
        where v is the respective eigvec
    """
    eigvals, eigvecs = np.linalg.eig(A)

    imag_modes = []
    real_modes = []

    for lam, v in zip(eigvals, eigvecs.T):
        # imag mode
        if   abs(lam.real) < tol and lam.imag > 0: imag_modes.append((lam.imag, v))
        # real mode
        elif abs(lam.imag) < tol: real_modes.append((lam.real, v))

    imag_modes.sort(key=lambda x: x[0])
    real_modes.sort(key=lambda x: x[0])

    return imag_modes, real_modes