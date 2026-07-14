"""
2) NONLINEAR APPROXIMATION

-> Lindstedt-Poincaré
-> Periodic approximations
-> The frequency is a perturbation of the linear solution

Assumptions:
    - y_e = 0 (xz plane)
    - constant attitude
    - work well for z_eq and A_z (amplitude) small
"""

import numpy as np
import sympy as sp
from itertools import product
from sympy.simplify.fu import TRpower,TR8

eps = sp.Symbol("epsilon")
tau = sp.Symbol("tau")
Az,k,m = sp.symbols("A_z k m")
lamb,phi = sp.symbols("lambda phi")
T = sp.Function("T")(tau) # T = lambda*tau + phi

def expand_xyz(order):
    """
    order n: 
        x = eps**1 * x1 + ... + eps**n * xn
    """

    x,y,z = 0,0,0
    for i in range(1,order+1):
        xn = sp.Function(f"x_{i}")(tau)
        yn = sp.Function(f"y_{i}")(tau)
        zn = sp.Function(f"z_{i}")(tau)
        x += eps**i * xn
        y += eps**i * yn
        z += eps**i * zn
    return x,y,z

def expand_omega(order):
    """
    order n: 
        w = 1 + eps**1 * w1 + ... + eps**(n-1) * w(n_1)

    Actually we expand omega until n-1 because it multiplies x,y,z.
    Since every term of x,y,z has an eps factor, eps**n * wn vanishes.
    """

    w = 1
    for i in range(1,order):
        wn = sp.symbols(f"omega_{i}")
        w += eps**i * wn
    return w


def truncate_eps(expr, order):
    """
    terms > order vanish 
    """
    return sp.series(sp.expand(expr), eps, 0, order + 1).removeO()

def truncate_matrix(M, order):
    return M.applyfunc(lambda e: truncate_eps(e, order))


def build_left_side(order, xyz, w):
    """
    Left: 
    (xddot-2*ydot)
    (yddot+2*xdot)
    (   zddot    )
    """
    
    x,y,z = xyz
    w2 = sp.expand(w**2)
    w2 = truncate_eps(w2,order-1) # (>= order) vanish

    Left = sp.Matrix([
        w2*sp.diff(x, tau, 2) - 2*w*sp.diff(y, tau),
        w2*sp.diff(y, tau, 2) + 2*w*sp.diff(x, tau),
        w2*sp.diff(z, tau, 2)
    ])
    Left = truncate_matrix(Left, order)

    return Left

def build_right_side(order,xyz):
    """
    Right:
    Taylor expansion of F about the equilibrium point

    r*delF + 1/2*r*r*del2F + 1/6*r*r*r*del3F + ...
    """

    x,y,z = xyz
    Right = sp.Matrix([0,0,0])

    # nth term of the Taylor expansion
    for n in range(1,order+1):

        for idx,dim in enumerate(["x","y","z"]):
            taylor_n = 0
            # example coordinates (n = 3): [z,x,x]
            for coordinates in product(["x","y","z"],repeat=n):
                # del_x del_y del_x ... del_z Fy -> Fyxyx...z 
                # (in this example, dim is 'y')
                delF = sp.Symbol("F_" + dim + "".join(coordinates))
                # x*y*x*...*z
                delr = 1
                for c in coordinates:
                    if   c == "x": delr *= x
                    elif c == "y": delr *= y
                    elif c == "z": delr *= z
                delr = truncate_eps(delr,order)
                taylor_n += 1/sp.factorial(n)*delr*delF
            Right[idx] += taylor_n
    
    return Right


def harmonic_coeffs(eq, order):
    """
    return coeffs = [dx,dy,dz],
    where di is a dict corresponding to components i = x, y or z, 
    containing the coefficients of the independent term ("const"),
    as well as the coefficients of cos(jT) and sin(jT)
    """

    coeffs = []

    # x, y and z
    for i in range(3):

        # i'th component's expression 
        expr = sp.expand(eq[i])

        # i'th component's dictionary of coefficients
        d = {}

        # d_aux: parameters to be used in sp.simplify to obtain the constant term
        # That is, d_aux = {x_n = 0, cos(jT) = 0, sin(jT) = 0}
        d_aux = {}
        for j in range(1, order + 1):
            d_aux[sp.cos(j*T)] = 0
            d_aux[sp.sin(j*T)] = 0
        d_aux[sp.Function(f"x_{order}")(tau)] = 0  
        d_aux[sp.Function(f"y_{order}")(tau)] = 0
        d_aux[sp.Function(f"z_{order}")(tau)] = 0

        # constant term
        d["const"] = sp.simplify(expr.subs(d_aux))

        # cos(jT) and sin(jT) terms
        for j in range(1, order + 1):
            d[sp.cos(j*T)] = sp.simplify(expr.coeff(sp.cos(j*T)))
            d[sp.sin(j*T)] = sp.simplify(expr.coeff(sp.sin(j*T)))

        coeffs.append(d)

    return coeffs

def substitute_prev_solutions(eq, prev_solutions):
    """
    Substitute x_1,y_1,z_1,...,x_(n-1),y_(n-1),z_(n-1)
    into nth order equation.

    These terms are in the form of a sum of K*cos(jT) and k*sin(jT).

    prev_solutions = [c_1,...,c_(n-1)], where ci are the ith order coeffs.
    See definition of "coeffs" in obtain_equation_coeffs(order).
    """

    # will be substituted into eq
    subst = {}

    # ci = ith order coeffs
    for i,ci in enumerate(prev_solutions,1):

        # ith domention's dict with coefficients
        for di,var in zip(ci,("x","y","z")):

            expr = 0

            # term = "const" or sp.cos(j*T) or sp.sin(j*T)
            for term,coeff in di.items():
                if term == "const": expr += coeff
                else: expr += coeff*term

            # ex: y_3(tau) = ...
            subst[sp.Function(f"{var}_{i}")(tau)] = expr

    eq = eq.subs(subst).doit()

    return eq


def obtain_equation_coeffs(order, prev_solutions):
    """
    Build the equation of order n (Eq = Right - Left)

    Return "coeffs", a list with 3 dictionaries of coefficients, 
    one for each dimension: [alpha_ni, beta_ni, gamma_ni].

    Example: 
    - coeffs = [dx,dy,dz]
    - dx["const"] gives the constant term in x
    - dy[sp.sin(3*T)] giver the term multiplying sin(3T) in y
    """

    # order n means eps**n
    xyz = expand_xyz(order)
    w = expand_omega(order)

    # nth order equation (Left = Right)
    Left = build_left_side(order,xyz,w)
    Right = build_right_side(order,xyz)
    Eq = Right - Left

    # take only eps^n terms
    Eq_order_n = Eq.applyfunc(lambda e: sp.expand(e).coeff(eps, order))

    # substitute x_1,y_1,z_1,...,x_(n-1),y_(n-1),z_(n-1)
    Eq_order_n = substitute_prev_solutions(Eq_order_n, prev_solutions)

    # dT/dtau = lamb, d2T/dtau2 = 0
    Eq_order_n = Eq_order_n.subs({
        sp.diff(T, tau): lamb,
        sp.diff(T, tau, 2): 0
    })

    # cos^n(T) -> cos(nT) + ... + cos(T)
    # sin^n(T) -> sin(nT) + ... + sin(T)
    Eq_order_n = Eq_order_n.applyfunc(lambda e: sp.expand(TR8(sp.expand(TRpower(sp.expand(e))))))

    coeffs = harmonic_coeffs(Eq_order_n, order) 

    return coeffs

def calculate_prev_omega(order,coeffs):
    """
    calculate and return omega_(n-1)
    order = n
    coeffs = order n coeffs
    """

    if order%2 == 0: return 0

    alpha_n1 = coeffs[0][sp.cos(T)]
    beta_n1  = coeffs[1][sp.sin(T)]
    gamma_n1 = coeffs[2][sp.cos(T)]
    b = sp.Symbol("F_xz")
    c = sp.Symbol("F_yy")
    e = sp.Symbol("F_zz")
    eq = 2*lamb*beta_n1/(c+lamb**2) + b*gamma_n1/(e+lamb**2) - alpha_n1
    
    # common denominator
    eq = sp.together(eq)

    solutions = sp.solve(sp.Eq(eq, 0),sp.Symbol(f"omega_{order-1}"))

    if len(solutions) != 1:
        raise RuntimeError(f"0 or >1 solutions for omega obtained")

    prev_omega = sp.simplify(solutions[0])

    return prev_omega

def substitute_prev_omegas(coeffs,subst):
    """
    Substitute omegas into coeffs.
    """

    coeffs_new = []

    for di in coeffs:
        coeffs_new.append({
            term: expr.subs(subst).doit()
            for term, expr in di.items()
        })

    return coeffs_new

def obtain_solution(order,coeffs):
    """
    order = n
    Return [dxn,dyn,dzn], where "din" is a dictionary 
    containing the coefficients of x_n, y_n, z_n.
    That is, dxn,dyn,dzn correspond to p,q,s respectively.
    """

    dxn = {}
    dyn = {}
    dzn = {}

    alpha_n,beta_n,gamma_n = coeffs

    a = sp.Symbol("F_xx")
    b = sp.Symbol("F_xz")
    c = sp.Symbol("F_yy")
    d = sp.Symbol("F_zx")
    e = sp.Symbol("F_zz")

    # order even: 0,2,4,...,n
    # order odd:  1,3,5,...,n
    for j in range(order%2,order+1,2):

        Aj = a + j**2*lamb**2
        Cj = c + j**2*lamb**2
        Ej = e + j**2*lamb**2
        Dj = Aj*Cj*Ej - 4*j**2*lamb**2*Ej - b*d*Cj

        if j==0:
            alpha_nj = alpha_n["const"]
            beta_nj  = beta_n ["const"]
            gamma_nj = gamma_n["const"]
        else:
            alpha_nj = alpha_n[sp.cos(j*T)]
            beta_nj  = beta_n [sp.sin(j*T)]
            gamma_nj = gamma_n[sp.cos(j*T)]

        if j==1:
            pnj = 0
            qnj = -beta_nj /(c+lamb**2)
            snj = -gamma_nj/(e+lamb**2)
        else:
            pnj = (2*j*lamb*beta_nj*Ej + b*gamma_nj*Cj - alpha_nj*Cj*Ej) / Dj
            qnj = (2*j*lamb*(alpha_nj*Ej - b*gamma_nj) + beta_nj*(b*d - Aj*Ej)) / Dj
            snj = (d*alpha_nj*Cj - 2*d*j*lamb*beta_nj + gamma_nj*(4*j**2*lamb**2 - Aj*Cj)) / Dj

        pnj, qnj, snj = tuple(map(sp.simplify, (pnj, qnj, snj)))

        if j==0:
            dxn["const"] = pnj
            dyn["const"] = qnj
            dzn["const"] = snj
        else:
            dxn[sp.cos(j*T)] = pnj
            dyn[sp.sin(j*T)] = qnj
            dzn[sp.cos(j*T)] = snj

    return [dxn,dyn,dzn]


def find_next_solution(order,prev_solutions,prev_omegas):
    """ 
    Given all previous solutions / omegas (order 1 to n-1), 
    calculate nth order solution and omega_(n-1)
    """

    # [alpha_ni, beta_ni, gamma_ni]
    coeffs = obtain_equation_coeffs(order,prev_solutions)

    # substitute omegas_1,...,omega_(n-2) into coeffs
    coeffs = substitute_prev_omegas(coeffs,prev_omegas)

    # omega_(n-1) 
    prev_omega = calculate_prev_omega(order, coeffs)

    # substitute omega_(n-1) into coeffs
    subst = {sp.Symbol(f"omega_{order-1}") : prev_omega}
    coeffs = substitute_prev_omegas(coeffs,subst)

    # build x_n, y_n, z_n
    sn = obtain_solution(order,coeffs)
    
    return sn, prev_omega


def all_nonlinear_approximations(max_order):

    """ 
    return a list with all nonlinear approximations 
    from order 1 to "max_order" 
    """

    # 1st order solution
    dx1 = {sp.cos(T) : k*Az}
    dy1 = {sp.sin(T) : m*Az}
    dz1 = {sp.cos(T) : Az}
    s1 = [dx1,dy1,dz1] 

    # only 1st order
    prev_solutions = [s1]

    # {sp.Symbol("omega_{i}") : omega_i}
    prev_omegas = {}

    for order in range(2,max_order+1):
        sn, prev_omega = find_next_solution(order,prev_solutions,prev_omegas)
        prev_solutions.append(sn)
        prev_omegas[sp.Symbol(f"omega_{order-1}")] = prev_omega

    return prev_solutions