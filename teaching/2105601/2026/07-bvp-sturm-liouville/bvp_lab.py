"""Session 7 (2026, semester 2): boundary value problems, Sturm-Liouville eigenproblems, Green's functions,
pellet shape and the axial dispersion reactor.

Course notation: z or x in (0, 1) a scaled position; phi the Thiele modulus based on the half thickness or radius;
s the shape index (0 slab, 1 cylinder, 2 sphere); Bi the mass Biot number; Pe the axial Peclet number, Da the
Damkohler number; Danckwerts conditions c(0) - c'(0)/Pe = 1, c'(1) = 0.

Run `python bvp_lab.py` for a summary, or `python bvp_lab.py --json` to write bvp-results.json.
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from scipy.special import iv


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, n=4):
    return f"{float(v):.{n}g}"


# ---------------------------------------------------------------------------
# Part 1. A slab with external mass transfer (Robin condition)
# ---------------------------------------------------------------------------
def robin_slab(phi, Bi):
    """u'' = phi^2 u, u'(0) = 0, u'(1) = Bi (1 - u(1)). Overall eta = (tanh phi/phi) / (1 + phi tanh phi / Bi)."""
    t = np.tanh(phi)
    exact = t / phi if np.isinf(Bi) else (t / phi) / (1 + phi * t / Bi)
    z = np.linspace(0, 1, 101)
    if np.isinf(Bi):
        bc = lambda ya, yb: np.array([ya[1], yb[0] - 1])
    else:
        bc = lambda ya, yb: np.array([ya[1], yb[1] - Bi * (1 - yb[0])])
    sol = solve_bvp(lambda z, y: np.vstack([y[1], phi ** 2 * y[0]]), bc, z, np.vstack([np.ones_like(z), z]), tol=1e-9)
    return {"phi": phi, "Bi": "inf" if np.isinf(Bi) else Bi, "exact": float(exact), "num": float(sol.sol(1.0)[1] / phi ** 2),
            "us": float(sol.sol(1.0)[0]), "nodes": int(sol.x.size), "ok": bool(sol.success)}


def shooting_gain(phis=(2, 5, 10, 20, 40)):
    """Shooting from the center: u(1) = s cosh(phi), so an error in s is amplified cosh(phi) times."""
    return [{"phi": p, "gain": float(np.cosh(p))} for p in phis]


# ---------------------------------------------------------------------------
# Part 2. Sturm-Liouville: -u'' = lam u, u(0) = 0, u'(1) + h u(1) = 0
# ---------------------------------------------------------------------------
def sl_robin(h=1.0, n=5, N=400):
    mu = [brentq(lambda m: np.tan(m) + m / h, (k - 0.5) * np.pi + 1e-9, k * np.pi - 1e-9) for k in range(1, n + 1)]
    lam = np.array(mu) ** 2
    # finite differences with a ghost node at x = 1: u_{N+1} = u_{N-1} - 2 dx h u_N
    dx = 1.0 / N
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 2
        if i > 0: A[i, i - 1] = -1
        if i < N - 1: A[i, i + 1] = -1
    A[N - 1, N - 2] = -2; A[N - 1, N - 1] = 2 + 2 * dx * h
    A /= dx * dx
    # symmetrize with the weight of the last node (1/2) to get real eigenvalues reliably
    fd = np.sort(np.linalg.eigvals(A).real)[:n]
    x = np.linspace(0, 1, 2001)
    U = np.array([np.sin(m * x) for m in mu])
    U = U / np.sqrt(np.trapezoid(U * U, x, axis=1))[:, None]
    G = np.array([[np.trapezoid(U[i] * U[j], x) for j in range(n)] for i in range(n)])
    off = float(np.abs(G - np.eye(n)).max())
    return {"h": h, "mu": [float(m) for m in mu], "lam": lam.tolist(), "fd": fd.tolist(), "fd_err": (np.abs(fd - lam) / lam).tolist(),
            "x": x[::20].tolist(), "U": U[:, ::20].tolist(), "offdiag": off}


# ---------------------------------------------------------------------------
# Part 3. Green's function and eigenfunction expansion: -u'' = f, u(0) = u(1) = 0, f = 1 on (0, 1/2)
# ---------------------------------------------------------------------------
def step_source(Ns=(1, 3, 10, 30, 100)):
    x = np.linspace(0, 1, 401)
    exact = np.where(x <= 0.5, -x ** 2 / 2 + 3 * x / 8, (1 - x) / 8)
    # Green's function quadrature check: u(x) = int_0^1/2 G(x, s) ds, G = s (1 - x) for s < x, x (1 - s) for s > x
    sgrid = np.linspace(0, 0.5, 2001)
    g = np.array([np.trapezoid(np.where(sgrid < xi, sgrid * (1 - xi), xi * (1 - sgrid)), sgrid) for xi in x])
    out = {"x": x.tolist(), "exact": exact.tolist(), "green_err": float(np.abs(g - exact).max()), "series": []}
    for N in Ns:
        n = np.arange(1, N + 1)
        fn = (1 - np.cos(n * np.pi / 2)) / (n * np.pi)                     # int_0^1 f sin(n pi x) dx
        u = (2 * fn / (n * np.pi) ** 2) @ np.sin(np.outer(n, np.pi * x))
        f = (2 * fn) @ np.sin(np.outer(n, np.pi * x))
        out["series"].append({"N": int(N), "err": float(np.abs(u - exact).max()), "u": u.tolist(), "f": f.tolist(),
                              "fmax": float(f.max())})
    return out


# ---------------------------------------------------------------------------
# Part 4. Pellet shape: slab, cylinder, sphere
# ---------------------------------------------------------------------------
def eta_shape(phi, s):
    if s == 0:
        return float(np.tanh(phi) / phi)
    if s == 1:
        return float(2 * iv(1, phi) / (phi * iv(0, phi)))
    return float(3 / phi ** 2 * (phi / np.tanh(phi) - 1))


def eta_shape_numeric(phi, s):
    """u'' + (s/r) u' = phi^2 u, u'(0) = 0, u(1) = 1; eta = (s + 1) u'(1)/phi^2 (solve_bvp with the singular term S/r)."""
    r = np.linspace(0, 1, 201)
    S = np.array([[0, 0], [0, -s]])
    sol = solve_bvp(lambda r, y: np.vstack([y[1], phi ** 2 * y[0]]), lambda ya, yb: np.array([ya[1], yb[0] - 1]),
                    r, np.vstack([np.ones_like(r), r]), S=S, tol=1e-9)
    return float((s + 1) * sol.sol(1.0)[1] / phi ** 2)


def shapes():
    phis = np.round(np.concatenate([np.arange(0.1, 2.0, 0.1), np.arange(2.0, 30.01, 0.5)]), 2)
    rows = [{"phi": float(p), "eta": [eta_shape(float(p), s) for s in (0, 1, 2)]} for p in phis]
    check = [{"s": s, "exact": eta_shape(3.0, s), "num": eta_shape_numeric(3.0, s)} for s in (0, 1, 2)]
    return {"rows": rows, "check3": check}


# ---------------------------------------------------------------------------
# Part 5. Axial dispersion reactor with Danckwerts conditions
# ---------------------------------------------------------------------------
def disp_exit(Pe, Da):
    """Wehner-Wilhelm exit concentration, written to avoid overflow."""
    q = np.sqrt(1 + 4 * Da / Pe)
    return float(4 * q * np.exp(Pe * (1 - q) / 2) / ((1 + q) ** 2 - (1 - q) ** 2 * np.exp(-q * Pe)))


def disp_profile(Pe, Da, n=401):
    q = np.sqrt(1 + 4 * Da / Pe)
    r1, r2 = Pe * (1 + q) / 2, Pe * (1 - q) / 2
    # c = A e^{r1 (z-1)} + B e^{r2 z}
    M = np.array([[np.exp(-r1) * (1 - r1 / Pe), 1 - r2 / Pe], [r1, r2 * np.exp(r2)]])
    A, B = np.linalg.solve(M, [1.0, 0.0])
    z = np.linspace(0, 1, n)
    c = A * np.exp(r1 * (z - 1)) + B * np.exp(r2 * z)
    return {"Pe": Pe, "Da": Da, "z": z.tolist(), "c": c.tolist(), "c0": float(c[0]), "c1": float(c[-1]), "X": float(1 - c[-1])}


def disp_numeric(Pe, Da):
    z = np.linspace(0, 1, 201)
    sol = solve_bvp(lambda z, y: np.vstack([y[1], Pe * (y[1] + Da * y[0])]),
                    lambda ya, yb: np.array([ya[0] - ya[1] / Pe - 1, yb[1]]), z, np.vstack([np.exp(-Da * z), -Da * np.exp(-Da * z)]), tol=1e-9)
    return {"c1": float(sol.sol(1.0)[0]), "c0": float(sol.sol(0.0)[0]), "nodes": int(sol.x.size), "ok": bool(sol.success)}


def disp_second(Pe, Da):
    """Second-order rate: (1/Pe) u'' - u' - Da u^2 = 0 with Danckwerts conditions (solve_bvp); no closed form."""
    z = np.linspace(0, 1, 201)
    sol = solve_bvp(lambda z, y: np.vstack([y[1], Pe * (y[1] + Da * y[0] ** 2)]),
                    lambda ya, yb: np.array([ya[0] - ya[1] / Pe - 1, yb[1]]), z, np.vstack([np.exp(-z), -np.exp(-z)]), tol=1e-8)
    return {"Pe": Pe, "Da": Da, "X": float(1 - sol.sol(1.0)[0]), "ok": bool(sol.success), "nodes": int(sol.x.size),
            "pfr": Da / (1 + Da), "cstr": float(1 - (np.sqrt(1 + 4 * Da) - 1) / (2 * Da))}


PE_LIST = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000]


def disp_sweep(Da):
    return [{"Pe": p, "X": 1 - disp_exit(p, Da)} for p in PE_LIST]


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return 0.0 if abs(v) < 1e-300 else v
    if isinstance(o, np.integer):
        return int(o)
    return o


LECT_DA, LAB_DA = 2.0, 3.0


def classroom_results():
    R = {"lecture": {"robin": [robin_slab(3.0, b) for b in (1.0, 3.0, 10.0, 100.0, np.inf)], "gain": shooting_gain(),
                     "sl": sl_robin(), "green": step_source(), "shapes": shapes(),
                     "disp": {"Da": LECT_DA, "prof": [disp_profile(p, LECT_DA) for p in (1, 10, 100)], "sweep": disp_sweep(LECT_DA),
                              "num10": disp_numeric(10, LECT_DA), "pfr": float(1 - np.exp(-LECT_DA)), "cstr": LECT_DA / (1 + LECT_DA)}},
         "lab": {"Da": LAB_DA, "prof": [disp_profile(p, LAB_DA) for p in (1, 5, 50)], "num5": disp_numeric(5, LAB_DA),
                 "sweep": disp_sweep(LAB_DA), "second": disp_second(5.0, LAB_DA), "pfr": float(1 - np.exp(-LAB_DA)), "cstr": LAB_DA / (1 + LAB_DA)},
         "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("bvp-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote bvp-results.json"); sys.exit()
    L = R["lecture"]
    print("robin", [(q["Bi"], r6(q["exact"]), r6(q["num"]), r6(q["us"]), q["nodes"]) for q in L["robin"]])
    print("gain", [(q["phi"], sig(q["gain"])) for q in L["gain"]])
    S = L["sl"]; print("SL mu", r6(S["mu"]), "lam", r6(S["lam"]), "fd rel err", [sig(e, 2) for e in S["fd_err"]], "offdiag", sig(S["offdiag"], 2))
    G = L["green"]; print("green err", sig(G["green_err"], 2), "series", [(q["N"], sig(q["err"], 3), r6(q["fmax"])) for q in G["series"]])
    print("shapes phi=3", [(q["s"], r6(q["exact"]), r6(q["num"])) for q in L["shapes"]["check3"]])
    D = L["disp"]; print("disp", [(q["Pe"], r6(q["c0"]), r6(q["X"])) for q in D["prof"]], "num10", D["num10"], "pfr", r6(D["pfr"]), "cstr", r6(D["cstr"]))
    print("sweep", [(q["Pe"], r6(q["X"])) for q in D["sweep"]])
    B = R["lab"]; print("lab", [(q["Pe"], r6(q["c0"]), r6(q["X"])) for q in B["prof"]], "num5", B["num5"], r6(B["pfr"]), r6(B["cstr"]))
    print("lab second", B["second"])
    print("lab sweep", [(q["Pe"], r6(q["X"])) for q in B["sweep"]])
