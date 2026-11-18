"""Session 6 (2026, semester 2): scaling, regular and singular perturbation, boundary layers.

Course notation: epsilon a small dimensionless parameter; outer solution y_o, inner solution y_i in the stretched
variable X = x/epsilon (or x/delta); composite y_c = y_o + y_i - (common limit). Thiele modulus phi for a catalyst slab:
d2c/dz2 = phi^2 r(c), c'(0) = 0, c(1) = 1; effectiveness factor eta = (1/phi^2) c'(1).

Run `python perturb_lab.py` for a summary, or `python perturb_lab.py --json` to write perturb-results.json.
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, n=4):
    return f"{float(v):.{n}g}"


# ---------------------------------------------------------------------------
# Part 1. Regular perturbation: algebra, then a weakly nonlinear decay
# ---------------------------------------------------------------------------
def regular_roots(eps_list=(0.3, 0.1, 0.03, 0.01)):
    """x^2 + eps x - 1 = 0: positive root ~ 1 - eps/2 + eps^2/8."""
    out = []
    for e in eps_list:
        exact = (-e + np.sqrt(e * e + 4)) / 2
        a1, a2 = 1 - e / 2, 1 - e / 2 + e * e / 8
        out.append({"eps": e, "exact": float(exact), "err1": float(abs(exact - a1)), "err2": float(abs(exact - a2))})
    return out


def singular_roots(eps_list=(0.3, 0.1, 0.03, 0.01)):
    """eps x^2 + x - 1 = 0: regular root ~ 1 - eps; singular root ~ -1/eps - 1 + eps."""
    out = []
    for e in eps_list:
        r = np.sort(np.roots([e, 1, -1]).real)
        out.append({"eps": e, "big": float(r[0]), "big_approx": float(-1 / e - 1 + e), "small": float(r[1]), "small_approx": float(1 - e)})
    return out


def weak_nonlinear(eps=0.2):
    """dc/dt = -c - eps c^2, c(0) = 1: exact c = e^-t / (1 + eps (1 - e^-t)); two-term c0 + eps c1."""
    t = np.linspace(0, 5, 101)
    exact = np.exp(-t) / (1 + eps * (1 - np.exp(-t)))
    c0 = np.exp(-t); c1 = -(np.exp(-t) - np.exp(-2 * t))
    return {"eps": eps, "t": t.tolist(), "exact": exact.tolist(), "one": c0.tolist(), "two": (c0 + eps * c1).tolist(),
            "err1": float(np.abs(exact - c0).max()), "err2": float(np.abs(exact - c0 - eps * c1).max())}


# ---------------------------------------------------------------------------
# Part 2. Singular perturbation: a boundary layer
# ---------------------------------------------------------------------------
def layer(eps, n=401):
    """eps y'' + y' + y = 0, y(0) = 0, y(1) = 1. Outer e^(1-x); inner e(1 - e^(-x/eps)); composite e^(1-x) - e^(1-x/eps)."""
    x = np.linspace(0, 1, n)
    d = np.sqrt(1 - 4 * eps)
    r1, r2 = (-1 + d) / (2 * eps), (-1 - d) / (2 * eps)
    # y = A(e^{r1 x} - e^{r2 x}) with y(1) = 1; written to avoid overflow
    y = (np.exp(r1 * (x - 1)) - np.exp(r2 * x - r1)) / (1 - np.exp(r2 - r1))
    comp = np.exp(1 - x) - np.exp(1 - x / eps)
    return {"eps": eps, "x": x.tolist(), "exact": y.tolist(), "outer": np.exp(1 - x).tolist(), "comp": comp.tolist(),
            "err": float(np.abs(y - comp).max())}


def layer_errors():
    return [{"eps": float(e), "err": layer(float(e))["err"]} for e in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005)]


# ---------------------------------------------------------------------------
# Part 3. Michaelis-Menten kinetics and the quasi-steady-state approximation
# ---------------------------------------------------------------------------
MM = {"k1": 1.0e6, "km1": 1.0, "k2": 1.0, "S0": 1.0e-5}     # 1/(M s), 1/s, 1/s, M  (SYNTHETIC values)


def michaelis(E0):
    k1, km1, k2, S0 = MM["k1"], MM["km1"], MM["k2"], MM["S0"]
    Km = (km1 + k2) / k1
    eps = E0 / (S0 + Km)                                   # Segel and Slemrod (1989)
    tc = 1 / (k1 * (S0 + Km))                               # time to reach quasi-steady state
    ts = (S0 + Km) / (k2 * E0)                              # substrate depletion time
    def full(t, y):
        s, c = y
        return [-k1 * (E0 - c) * s + km1 * c, k1 * (E0 - c) * s - (km1 + k2) * c]
    t = np.linspace(0, 4 * ts, 201)
    sol = solve_ivp(full, (0, t[-1]), [S0, 0.0], t_eval=t, method="Radau", rtol=1e-10, atol=1e-16)
    q = solve_ivp(lambda tt, s: [-k2 * E0 * s[0] / (Km + s[0])], (0, t[-1]), [S0], t_eval=t, method="Radau", rtol=1e-10, atol=1e-16)
    return {"E0": E0, "eps": float(eps), "Km": float(Km), "tc": float(tc), "ts": float(ts), "t": (t / ts).tolist(),
            "s_full": (sol.y[0] / S0).tolist(), "s_qssa": (q.y[0] / S0).tolist(), "c_full": (sol.y[1] / E0).tolist(),
            "err": float(np.abs(sol.y[0] - q.y[0]).max() / S0)}


# ---------------------------------------------------------------------------
# Part 4. Lab: catalyst slab, first and second order, small and large Thiele modulus
# ---------------------------------------------------------------------------
def slab_numeric(phi, order=1):
    """d2c/dz2 = phi^2 c^n on (0, 1), c'(0) = 0, c(1) = 1; eta = c'(1)/phi^2 (solve_bvp, continuation in phi)."""
    z = np.linspace(0, 1, 401)
    y0 = np.vstack([np.cosh(phi * z) / np.cosh(phi), phi * np.sinh(phi * z) / np.cosh(phi)])
    fun = lambda z, y: np.vstack([y[1], phi ** 2 * np.maximum(y[0], 0) ** order])
    bc = lambda ya, yb: np.array([ya[1], yb[0] - 1])
    sol = solve_bvp(fun, bc, z, y0, tol=1e-8, max_nodes=200000)
    return {"phi": phi, "order": order, "eta": float(sol.sol(1.0)[1] / phi ** 2), "z": z.tolist(), "c": sol.sol(z)[0].tolist(), "ok": bool(sol.success)}


def slab_first(phi):
    return {"phi": float(phi), "exact": float(np.tanh(phi) / phi), "small": float(1 - phi ** 2 / 3), "large": float(1 / phi)}


def slab_sweep():
    out = []
    for phi in np.round(np.concatenate([np.arange(0.2, 2.0, 0.2), np.arange(2.0, 20.01, 1.0)]), 2):
        q = slab_first(float(phi))
        q["eta2_num"] = slab_numeric(float(phi), 2)["eta"]
        q["eta2_asym"] = float(np.sqrt(2 / 3) / phi)
        out.append(q)
    return out


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


def classroom_results():
    R = {"lecture": {"reg": regular_roots(), "sing": singular_roots(), "weak": weak_nonlinear(), "layer": layer(0.05),
                     "layer_err": layer_errors(), "mm": [michaelis(E) for E in (1e-8, 1e-7, 1e-6, 1e-5)]},
         "lab": {"phi5": slab_first(5.0), "prof1": slab_numeric(5.0, 1), "prof2": slab_numeric(5.0, 2), "sweep": slab_sweep()},
         "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("perturb-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote perturb-results.json"); sys.exit()
    L = R["lecture"]
    print("reg", [(q["eps"], sig(q["err1"]), sig(q["err2"])) for q in L["reg"]])
    print("sing", [(q["eps"], r6(q["big"]), r6(q["big_approx"])) for q in L["sing"]])
    print("weak err", sig(L["weak"]["err1"]), sig(L["weak"]["err2"]))
    print("layer err", [(q["eps"], sig(q["err"])) for q in L["layer_err"]])
    print("mm", [(q["E0"], sig(q["eps"]), sig(q["err"]), sig(q["tc"]), sig(q["ts"])) for q in L["mm"]])
    B = R["lab"]; print("phi5", B["phi5"], "num1", r6(B["prof1"]["eta"]), "num2", r6(B["prof2"]["eta"]), B["prof2"]["ok"])
    print("sweep", [(q["phi"], r6(q["exact"]), r6(q["eta2_num"]), r6(q["eta2_asym"])) for q in B["sweep"]][::4])
