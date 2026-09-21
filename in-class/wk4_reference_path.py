"""
=============================================================================
 Advanced Mechanics (Fall 2026) - M. Breinig
 Week 4 Numeric Live Session: STATIONARY ACTION
 -----------------------------------------------------------------------
 REFERENCE PATH GENERATOR  (instructor tool -- not distributed to students)
=============================================================================

 PURPOSE
 -------
 Produce the high-accuracy reference path q*(t) and reference action S*
 against which the students' discretized action-minimization is measured.
 The session's headline result -- the discrete path converging at ratio
 4.000 -- is only meaningful if the reference is accurate to ~1e-13, i.e.
 several orders below the finest student grid error (~3e-7 at N=2560).

 SYSTEM
 ------
     L(q, qdot) = (1/2) m qdot^2 - V(q),     V(q) = (1/2) w^2 q^2 + (1/4) lam q^4

 Fixed-endpoint (Dirichlet) boundary value problem:

     q(0) = QA,   q(T) = QB

 Euler-Lagrange / Newton equation:

     m qddot = -V'(q) = -(w^2 q + lam q^3)

 ---------------------------------------------------------------------------
 ARCHITECTURE -- READ THIS BEFORE EDITING
 ---------------------------------------------------------------------------
 SHOOTING IS THE PRIMARY SOLVER. `solve_bvp` is NOT used as the primary.

 Reason (verified): seeding scipy.integrate.solve_bvp with the obvious
 straight-line initial guess converges to a SPURIOUS high-energy solution
 (v0 ~ -19.83, ~97,000 mesh nodes, status=1 max_nodes exceeded). This BVP
 has FIVE distinct solutions for the parameters below -- all with identical
 endpoints. A boundary value problem does not have "the" solution, so the
 reference path must be *selected*, not merely *solved for*.

 The pipeline is therefore:
   Stage 1  Enumerate ALL stationary paths by shooting + bracketed brentq.
   Stage 2  Select the fundamental branch (smallest |v0| -> no interior
            oscillation).
   Stage 3  Cross-validate that branch three independent ways:
              (a) collocation (solve_bvp) SEEDED from the shooting solution
              (b) energy conservation along the trajectory
              (c) action computed from both representations
   Stage 4  Compute S* by composite Gauss-Legendre quadrature.
   Stage 5  Verify the discrete stationary path converges at ratio 4.000
            (this is the number quoted in the session).

 ---------------------------------------------------------------------------
 VERIFIED OUTPUT (this machine, scipy 1.x)
 ---------------------------------------------------------------------------
 Five stationary paths, same endpoints:

     v0 = -22.015365232126    S = +156.585374062283    (3 sign changes)
     v0 = -10.807647636882    S =  +40.714253535636    (3 sign changes)
     v0 =  -2.306319364952    S =   +1.778166052567    (1)  <-- FUNDAMENTAL
     v0 =  +7.626968737797    S =  +12.707790792758    (1 sign change)
     v0 = +20.996824188020    S = +135.060350347343    (3 sign changes)

 Reference values (fundamental branch):
     v0*    = -2.306319364952331
     S*     =  1.778166052567412
     E*     =  3.409554506577      (conserved; drift 6.8e-13)

 Cross-check agreement (shooting vs seeded collocation):
     |v0 difference|        1.2e-13
     max |q_col - q_shoot|  2.8e-13
     |S_col - S_shoot|      3.3e-13

 Discrete action-minimization convergence (exact Newton, banded Jacobian):

        N   max|q - q*|      ratio     |residual|
       20   5.267258e-03        --       3.3e-15
       40   1.309207e-03     4.023       7.3e-15
       80   3.268843e-04     4.005       9.0e-15
      160   8.173528e-05     3.999       3.0e-14
      320   2.043201e-05     4.000       5.9e-14
      640   5.107890e-06     4.000       1.3e-13
     1280   1.276967e-06     4.000       2.6e-13
     2560   3.192419e-07     4.000       5.2e-13

 Second-order (ratio 4) is the DELIBERATE CONTRAST with the Week 2
 quadrature session, where the singular integrand degraded the midpoint
 rule to ratio sqrt(2) = 1.41.
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq
from scipy.linalg import solve_banded
from numpy.polynomial.legendre import leggauss
from pathlib import Path
OUT_PATH = Path(__file__).with_name("wk4_reference.npz")   # always beside the .py
N_EXPORT = 20001    # dense: spline error ~1e-15, well below the N=2560 row
# =============================================================================
#  1. CONFIGURATION
# =============================================================================

M_MASS = 1.0        # mass m
OMEGA  = 1.0        # harmonic frequency w
LAMBDA = 1.0        # quartic stiffness lam  (set 0.0 for the pure oscillator)

QA = 1.0            # q(0)
QB = -0.5           # q(T)
T_END = 2.0         # endpoint time T

# Shooting: window and resolution for enumerating solution branches.
V0_WINDOW = (-30.0, 30.0)
V0_SCAN_POINTS = 3001

# Tolerances. IVP is deliberately tighter than the target reference accuracy.
IVP_RTOL, IVP_ATOL = 1e-13, 1e-15
BVP_TOL = 1e-10     # verified: status=0, 3502 nodes, path error 2.8e-13.
                    # 1e-11 or tighter trips max_nodes without improving accuracy.

GL_SUBINTERVALS = 400   # composite Gauss-Legendre panels
GL_NODES = 16           # nodes per panel

NEWTON_MAX_ITER = 60
NEWTON_TOL = 1e-14


# =============================================================================
#  2. THE PHYSICS
# =============================================================================

def V(q):
    """Potential energy."""
    return 0.5 * OMEGA**2 * q**2 + 0.25 * LAMBDA * q**4


def dV(q):
    """dV/dq -- the force is -dV."""
    return OMEGA**2 * q + LAMBDA * q**3


def d2V(q):
    """d^2V/dq^2 -- needed for the exact Newton Jacobian and for the
    second-variation operator."""
    return OMEGA**2 + 3.0 * LAMBDA * q**2


def lagrangian(q, qdot):
    return 0.5 * M_MASS * qdot**2 - V(q)


def energy(q, qdot):
    return 0.5 * M_MASS * qdot**2 + V(q)


def _rhs_scalar(t, y):
    """RHS for solve_ivp: y = [q, qdot]."""
    return [y[1], -dV(y[0]) / M_MASS]


def _rhs_vectorized(t, y):
    """RHS for solve_bvp, which passes a whole mesh at once."""
    return np.vstack([y[1], -dV(y[0]) / M_MASS])


def _bc(ya, yb):
    """Dirichlet endpoint conditions."""
    return np.array([ya[0] - QA, yb[0] - QB])


# =============================================================================
#  3. STAGE 1-2  --  ENUMERATE AND SELECT STATIONARY PATHS (SHOOTING)
# =============================================================================

def _shoot(v0, dense=False):
    """Integrate the IVP from (QA, v0) and return the solution object."""
    return solve_ivp(_rhs_scalar, (0.0, T_END), [QA, v0],
                     rtol=IVP_RTOL, atol=IVP_ATOL, dense_output=dense)


def _endpoint_miss(v0):
    """Shooting residual: q(T; v0) - QB. Its roots are the stationary paths."""
    return _shoot(v0).y[0, -1] - QB


def enumerate_stationary_paths(verbose=True):
    """
    Find every stationary path with the prescribed endpoints, by scanning the
    initial velocity and bracketing sign changes of the endpoint miss.

    Returns a list of dicts sorted by |v0|, each holding v0, the action, the
    dense solution, and the number of interior sign changes of q (a crude but
    effective index of how oscillatory the branch is).
    """
    grid = np.linspace(V0_WINDOW[0], V0_WINDOW[1], V0_SCAN_POINTS)
    miss = np.array([_endpoint_miss(v) for v in grid])

    brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
                if miss[i] * miss[i + 1] < 0.0]

    branches = []
    for a, b in brackets:
        v0 = brentq(_endpoint_miss, a, b, xtol=1e-15, rtol=8.9e-16)
        sol = _shoot(v0, dense=True)
        S = action_from_dense(sol.sol)
        tt = np.linspace(0.0, T_END, 4001)
        n_sign = int(np.sum(np.diff(np.sign(sol.sol(tt)[0])) != 0))
        branches.append(dict(v0=v0, action=S, sol=sol, sign_changes=n_sign))

    branches.sort(key=lambda d: abs(d["v0"]))

    if verbose:
        print(f"Stationary paths found in v0 in {V0_WINDOW}: {len(branches)}")
        print(f"  {'v0':>22}  {'S[q]':>20}  {'sign changes':>12}")
        for br in branches:
            print(f"  {br['v0']:>22.12f}  {br['action']:>20.12f}"
                  f"  {br['sign_changes']:>12d}")
        print("\n  NOTE: all of these share the SAME endpoints. Stationarity")
        print("        does not single out a unique path. This is the buffer")
        print("        extension of the live session.\n")

    return branches


def select_fundamental(branches):
    """
    Select the fundamental branch: smallest |v0|, i.e. the least oscillatory
    path. This is the one used as the reference, and the one a student's
    Newton iteration will find when seeded from a straight line.
    """
    return branches[0]


# =============================================================================
#  4. HIGH-ACCURACY ACTION  --  COMPOSITE GAUSS-LEGENDRE
# =============================================================================
def _as_callable(dense_sol):
    """Accept either an OdeResult (from solve_ivp(dense_output=True)) or its
    .sol interpolant, and always return the callable."""
    return dense_sol.sol if hasattr(dense_sol, "sol") else dense_sol

def action_from_dense(dense_sol, n_sub=GL_SUBINTERVALS, n_nodes=GL_NODES):
    """Composite Gauss-Legendre quadrature of S = ∫ L(q, qdot) dt."""
    dense_sol = _as_callable(dense_sol)          # from the previous fix

    x, w = leggauss(n_nodes)                     # was: leggauss(n_nodes)

    edges = np.linspace(0.0, T_END, n_sub + 1)   # was: n_panels / n_sub mismatch?
    half  = 0.5 * (edges[1] - edges[0])

    S = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        ts  = mid + half * x
        y   = dense_sol(ts)
        S  += half * np.dot(w, lagrangian(y[0], y[1]))
    return S


def action_convergence_report(dense_sol):
    """Show that the quadrature itself is converged (panel refinement)."""
    print("Gauss-Legendre panel refinement (should saturate):")
    for n_sub in (50, 100, 200, 400, 800):
        print(f"    n_sub = {n_sub:>4d}   S = "
              f"{action_from_dense(dense_sol, n_sub):.15f}")
    print()


# =============================================================================
#  5. STAGE 3  --  INDEPENDENT CROSS-VALIDATION
# =============================================================================

def collocation_crosscheck(reference):
    """
    Re-solve the BVP by collocation, SEEDED FROM THE SHOOTING SOLUTION, as a
    methodologically independent confirmation.

    Do not seed this with a straight line. Verified failure mode: a linear
    guess drives solve_bvp to a spurious branch at v0 ~ -19.83 with ~97,000
    nodes and status=1.
    """
    t_seed = np.linspace(0.0, T_END, 801)
    y_seed = reference["sol"].sol(t_seed)

    sol_c = solve_bvp(_rhs_vectorized, _bc, t_seed, y_seed,
                      tol=BVP_TOL, max_nodes=200000)

    tt = np.linspace(0.0, T_END, 4001)
    dv0 = abs(sol_c.sol(0.0)[1] - reference["v0"])
    dq = np.max(np.abs(sol_c.sol(tt)[0] - reference["sol"].sol(tt)[0]))
    S_col = action_from_dense(sol_c.sol)
    dS = abs(S_col - reference["action"])

    print("Collocation cross-check (seeded from shooting):")
    print(f"    status              {sol_c.status}  (0 = converged)")
    print(f"    mesh nodes          {sol_c.x.size}")
    print(f"    |v0 difference|     {dv0:.2e}")
    print(f"    max |q_col - q*|    {dq:.2e}")
    print(f"    |S_col - S*|        {dS:.2e}")

    ok = (sol_c.status == 0) and (dq < 1e-10) and (dS < 1e-10)
    print(f"    -> {'PASS' if ok else 'FAIL'}\n")
    return dict(sol=sol_c, dv0=dv0, dq=dq, dS=dS, passed=ok)


def energy_crosscheck(reference):
    """
    Third, physics-based check: the Lagrangian is time-independent, so energy
    must be conserved along the reference path. This catches integrator
    failure that endpoint agreement alone would not.
    """
    tt = np.linspace(0.0, T_END, 4001)
    y = reference["sol"].sol(tt)
    E = energy(y[0], y[1])
    E0 = energy(QA, reference["v0"])
    drift = np.max(np.abs(E - E0))

    print("Energy conservation along reference path:")
    print(f"    E*                  {E0:.12f}")
    print(f"    max drift           {drift:.2e}")
    ok = drift < 1e-9
    print(f"    -> {'PASS' if ok else 'FAIL'}\n")
    return dict(E0=E0, drift=drift, passed=ok)


# =============================================================================
#  6. STAGE 5  --  DISCRETE STATIONARY PATH (what the students build)
# =============================================================================

def discrete_action(q_interior, N):
    """
    Discrete action on a uniform grid with fixed endpoints, using the
    trapezoid/midpoint-consistent form the students write:

        S_h = sum_k [ (m/2) ((q_{k+1}-q_k)/h)^2 * h  -  V(q_k) * h ]

    Included so the instructor version and the student version can be
    compared directly. The reference path itself does not depend on it.
    """
    h = T_END / N
    q = np.concatenate(([QA], q_interior, [QB]))
    kinetic = 0.5 * M_MASS * np.sum((np.diff(q) / h) ** 2) * h
    potential = np.sum(V(q[:-1])) * h
    return kinetic - potential


def discrete_stationary_path(N, seed_dense):
    """
    Solve dS_h/dq_k = 0 for the interior nodes by exact Newton with the
    banded (tridiagonal) Jacobian.

    The stationarity condition is the discrete Euler-Lagrange equation

        -m (q_{k+1} - 2 q_k + q_{k-1}) / h  -  h V'(q_k)  =  0

    which is precisely the Verlet stencil. That identification is the
    session's punchline: minimizing the action *is* integrating Newton's law.

    Newton is used rather than fsolve because fsolve emits xtol warnings at
    the accuracy needed here and cannot reach machine-precision residuals.
    """
    t = np.linspace(0.0, T_END, N + 1)
    h = T_END / N
    q = seed_dense(t[1:-1])[0].copy()

    residual = np.inf
    for _ in range(NEWTON_MAX_ITER):
        q_full = np.concatenate(([QA], q, [QB]))
        F = (-M_MASS * (q_full[2:] - 2.0 * q_full[1:-1] + q_full[:-2]) / h
             - h * dV(q))
        residual = np.max(np.abs(F))
        if residual < NEWTON_TOL:
            break
        # Tridiagonal Jacobian dF_i/dq_j in scipy banded storage.
        ab = np.zeros((3, N - 1))
        ab[0, 1:] = -M_MASS / h                       # super-diagonal
        ab[1, :] = 2.0 * M_MASS / h - h * d2V(q)      # diagonal
        ab[2, :-1] = -M_MASS / h                      # sub-diagonal
        q = q + solve_banded((1, 1), ab, -F)

    return t, np.concatenate(([QA], q, [QB])), residual


def convergence_report(reference, grids=(20, 40, 80, 160, 320, 640, 1280, 2560)):
    """
    The table quoted in the live session. Expect ratio -> 4.000 exactly
    (second-order). Contrast with Week 2, where a singular integrand gave
    ratio sqrt(2).
    """
    print("Discrete stationary path vs reference (expect ratio -> 4.000):")
    print(f"    {'N':>6}  {'max|q - q*|':>15}  {'ratio':>7}  {'|residual|':>11}")
    prev = None
    rows = []
    for N in grids:
        t, q, res = discrete_stationary_path(N, reference["sol"].sol)
        err = np.max(np.abs(q - reference["sol"].sol(t)[0]))
        ratio = np.nan if prev is None else prev / err
        rows.append((N, err, ratio, res))
        rtxt = "  --  " if prev is None else f"{ratio:.3f}"
        print(f"    {N:>6d}  {err:>15.6e}  {rtxt:>7}  {res:>11.1e}")
        prev = err

    tail = [r[2] for r in rows[-4:]]
    ok = all(abs(x - 4.0) < 0.05 for x in tail)
    print(f"    -> {'PASS' if ok else 'FAIL'} (second order confirmed)\n")
    return rows, ok


# =============================================================================
#  7. SECOND VARIATION  --  the "stationary is not minimum" keystone
# =============================================================================

def second_variation_coefficient(N=4000):
    """
    For the PURE harmonic oscillator (LAMBDA = 0) with q(0) = q(T) = 0, the
    true path is q* == 0 and the second variation along the lowest-mode
    perturbation eta(t) = sin(pi t / T) evaluates in closed form to

        d2S = (T/4) [ (pi/T)^2 - w^2 ]

    which is POSITIVE for T < pi/w, ZERO at the conjugate point T = pi/w, and
    NEGATIVE beyond. The true path is then a saddle, not a minimum.

    Returned alongside a direct quadrature of the same quantity so the
    session's analytic curve is backed by a numerical evaluation.
    """
    t = np.linspace(0.0, T_END, N + 1)
    eta = np.sin(np.pi * t / T_END)
    eta_dot = (np.pi / T_END) * np.cos(np.pi * t / T_END)
    # d2S = (1/2) \int [ m etadot^2 - V''(q*) eta^2 ] dt, with q* = 0
    integrand = M_MASS * eta_dot**2 - d2V(0.0) * eta**2
    numeric = 0.5 * np.trapezoid(integrand, t)
    analytic = 0.25 * T_END * ((np.pi / T_END) ** 2 - OMEGA**2)
    return numeric, analytic


# =============================================================================
#  8. EXPORT
# =============================================================================

def export_reference(ref, path=OUT_PATH, n_export=N_EXPORT):
    """Write ONLY float arrays / scalars. No solver objects, no dicts."""
    dense = _as_callable(ref["sol"])

    t = np.linspace(0.0, T_END, n_export)
    y = dense(t)                          # shape (2, n_export)

    np.savez_compressed(
        path,
        t=np.asarray(t, dtype=float),
        q=np.asarray(y[0], dtype=float),
        qdot=np.asarray(y[1], dtype=float),
        S_star=float(ref["action"]),
        v0_star=float(ref["v0"]),
        E_star=float(energy(y[0, 0], y[1, 0])),
        m=float(M_MASS), omega=float(OMEGA), lam=float(LAMBDA),
        qa=float(QA), qb=float(QB), T_end=float(T_END),
    )

    # Fail loudly if anything non-numeric slipped through.
    with np.load(path) as chk:
        for k in chk.files:
            if chk[k].dtype == object:
                raise TypeError(f"'{k}' exported as object dtype — not numeric")
    return path


def reference_interpolant(reference):
    """
    Return a callable q*(t) for direct use in a notebook. This is the ODE
    solver's own dense output (a C1 interpolant of the true solution), not a
    spline through samples, so it retains full solver accuracy.
    """
    return lambda t: reference["sol"].sol(np.atleast_1d(t))[0]


# =============================================================================
#  9. MAIN
# =============================================================================

def build_reference(verbose=True):
    """
    Run the full pipeline and return the validated reference. Raises if any
    of the three independent cross-checks fails -- an answer key that has not
    been validated is worse than no answer key.
    """
    if verbose:
        print("=" * 74)
        print(" WEEK 4 REFERENCE PATH  --  stationary action")
        print("=" * 74)
        print(f" V(q) = {0.5*OMEGA**2:.3g} q^2 + {0.25*LAMBDA:.3g} q^4     "
              f"m = {M_MASS:g}")
        print(f" q(0) = {QA:g},  q({T_END:g}) = {QB:g}\n")

    branches = enumerate_stationary_paths(verbose=verbose)
    ref = select_fundamental(branches)

    if verbose:
        print("FUNDAMENTAL BRANCH (the reference):")
        print(f"    v0*  =  {ref['v0']:.15f}")
        print(f"    S*   =  {ref['action']:.15f}\n")
        action_convergence_report(ref["sol"])

    chk_col = collocation_crosscheck(ref)
    chk_E = energy_crosscheck(ref)

    if verbose:
        rows, ok_conv = convergence_report(ref)
        num, ana = second_variation_coefficient()
        print("Second-variation coefficient (harmonic limit, lowest mode):")
        print(f"    numeric   {num:+.9f}")
        print(f"    analytic  {ana:+.9f}   [(T/4)((pi/T)^2 - w^2)]")
        print(f"    conjugate point at T = pi/w = {np.pi/OMEGA:.6f}")
        print(f"    -> true path is a {'MINIMUM' if ana > 0 else 'SADDLE'} "
              f"at T = {T_END:g}\n")
    else:
        rows, ok_conv = convergence_report(ref)

    if not (chk_col["passed"] and chk_E["passed"] and ok_conv):
        raise RuntimeError("Reference validation FAILED -- do not use these "
                           "numbers in the answer key.")

    if verbose:
        print("=" * 74)
        print(" ALL CHECKS PASSED -- reference is safe to quote.")
        print("=" * 74)

    ref["branches"] = branches
    ref["convergence"] = rows
    return ref


if __name__ == "__main__":
    reference = build_reference(verbose=True)
    export_reference(reference)

    # Convenience: callable reference path.
    q_star = reference_interpolant(reference)
    print(f"\nSpot check:  q*(1.0) = {q_star(1.0)[0]:.15f}")
