"""Session 4 (2026, semester 2): nonlinear ODEs, linearization, stability and the phase plane.

Model: the nonadiabatic CSTR with a first-order exothermic reaction in dimensionless form
(Uppal, Ray and Poore, 1974, in the limit of large activation energy):
    dx1/dt = -x1 + Da (1 - x1) exp(x2)                 conversion x1
    dx2/dt = -x2 + B Da (1 - x1) exp(x2) - beta x2     dimensionless temperature rise x2
Da: Damkohler number; B: adiabatic temperature rise; beta: heat-transfer coefficient.
Course notation: x* a steady state, J = df/dx at x*, eigenvalues lambda of J (Sessions 2 and 3).

Run `python stability_lab.py` for a summary, or `python stability_lab.py --json` to write stability-results.json.
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

P = {"B": 8.0, "beta": 0.3}
DA_LECTURE, DA_LAB = 0.06, 0.045


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, n=4):
    return f"{float(v):.{n}g}"


def f(x, Da, B=P["B"], beta=P["beta"]):
    r = Da * (1 - x[0]) * np.exp(x[1])
    return np.array([-x[0] + r, -x[1] + B * r - beta * x[1]])


def jac(x, Da, B=P["B"], beta=P["beta"]):
    e = np.exp(x[1])
    return np.array([[-1 - Da * e, Da * (1 - x[0]) * e], [-B * Da * e, -1 - beta + B * Da * (1 - x[0]) * e]])


def steady_states(Da, B=P["B"], beta=P["beta"]):
    """At steady state x2 = B x1/(1 + beta); solve g(x1) = 0 on [0, 1) by bracketing."""
    g = lambda x1: x1 - Da * (1 - x1) * np.exp(B * x1 / (1 + beta))
    xs = np.linspace(0, 1 - 1e-9, 20001); v = g(xs)
    roots = [brentq(g, xs[i], xs[i + 1], xtol=1e-14) for i in range(len(xs) - 1) if v[i] * v[i + 1] < 0]
    return [np.array([x1, B * x1 / (1 + beta)]) for x1 in roots]


def classify(w):
    re, im = np.real(w), np.imag(w)
    if np.all(re < 0):
        return "stable focus" if np.any(np.abs(im) > 1e-12) else "stable node"
    if np.all(re > 0):
        return "unstable focus" if np.any(np.abs(im) > 1e-12) else "unstable node"
    return "saddle"


def analyse(Da):
    out = []
    for x in steady_states(Da):
        J = jac(x, Da); w = np.linalg.eigvals(J)
        w = w[np.argsort(-w.real)]
        out.append({"x": x.tolist(), "J": J.tolist(), "tr": float(np.trace(J)), "det": float(np.linalg.det(J)),
                    "lam_re": np.real(w).tolist(), "lam_im": np.imag(w).tolist(), "type": classify(w),
                    "residual": float(np.abs(f(x, Da)).max())})
    return out


def heat_curves(Da, B=P["B"], beta=P["beta"]):
    """Van Heerden view: generation G(x2) = B Da e/(1 + Da e) against removal R(x2) = (1 + beta) x2."""
    x2 = np.linspace(0, 8, 161); e = np.exp(x2)
    return {"x2": x2.tolist(), "G": (B * Da * e / (1 + Da * e)).tolist(), "R": ((1 + beta) * x2).tolist()}


def trajectories(Da, starts, tmax=30.0):
    out = []
    for s in starts:
        sol = solve_ivp(lambda t, x: f(x, Da), (0, tmax), s, rtol=1e-9, atol=1e-11, method="LSODA", dense_output=True)
        t = np.linspace(0, tmax, 301)
        X = sol.sol(t).T
        out.append({"x0": list(map(float, s)), "X": X.tolist(), "end": X[-1].tolist()})
    return out


def separatrix(Da):
    """Stable manifold of the saddle: integrate backward in time from the saddle along its stable eigenvector."""
    sad = [s for s in analyse(Da) if s["type"] == "saddle"][0]
    x = np.array(sad["x"]); w, V = np.linalg.eig(np.array(sad["J"]))
    v = V[:, np.argmin(w.real)].real
    branches = []
    def leave(t, z):                       # stop when the backward path leaves the window 0 <= x1 <= 1, -0.5 <= x2 <= 8.5
        return min(z[0], 1.0 - z[0], z[1] + 0.5, 8.5 - z[1])
    leave.terminal = True
    for sgn in (1, -1):
        sol = solve_ivp(lambda t, z: -f(z, Da), (0, 12), x + sgn * 1e-4 * v, rtol=1e-9, atol=1e-11, method="LSODA",
                        events=leave, max_step=0.05)
        branches.append(sol.y.T[::max(1, sol.y.shape[1] // 300)].tolist())
    unstable = []
    u = V[:, np.argmax(w.real)].real
    for sgn in (1, -1):
        sol = solve_ivp(lambda t, z: f(z, Da), (0, 30), x + sgn * 1e-3 * u, rtol=1e-9, atol=1e-11, method="LSODA")
        unstable.append({"end": sol.y[:, -1].tolist(), "X": sol.y.T[::max(1, sol.y.shape[1] // 300)].tolist()})
    return {"saddle": sad["x"], "stable_branches": branches, "unstable": unstable}


def transient_growth():
    """A stable but non-normal linear system: ||exp(Jt)|| rises before it decays (Session 2)."""
    from scipy.linalg import expm
    J = np.array([[-1.0, 20.0], [0.0, -2.0]])
    t = np.linspace(0, 6, 121)
    g = [float(np.linalg.norm(expm(J * s), 2)) for s in t]
    k = int(np.argmax(g))
    return {"J": J.tolist(), "t": t.tolist(), "g": g, "gmax": g[k], "tmax": float(t[k])}


def da_sweep():
    out = []
    for Da in np.round(np.arange(0.010, 0.0901, 0.0025), 4):
        S = analyse(float(Da))
        out.append({"Da": float(Da), "n": len(S), "x2": [s["x"][1] for s in S], "types": [s["type"] for s in S],
                    "maxre": [s["lam_re"][0] for s in S]})
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return 0.0 if abs(v) < 1e-14 else v
    if isinstance(o, np.integer):
        return int(o)
    return o


def classroom_results():
    grid = [[a, b] for a in (0.0, 0.5, 1.0) for b in (0.0, 4.0, 8.0)]
    lab_starts = []
    S = analyse(DA_LAB); sad = [s for s in S if s["type"] == "saddle"][0]
    w, V = np.linalg.eig(np.array(sad["J"])); u = V[:, np.argmax(w.real)].real
    for sgn in (1, -1):
        lab_starts.append((np.array(sad["x"]) + sgn * 0.02 * u).tolist())
    R = {"lecture": {"P": P, "Da": DA_LECTURE, "ss": analyse(DA_LECTURE), "heat": heat_curves(DA_LECTURE),
                     "traj": trajectories(DA_LECTURE, grid), "sep": separatrix(DA_LECTURE), "tg": transient_growth()},
         "lab": {"Da": DA_LAB, "ss": S, "kick": trajectories(DA_LAB, lab_starts), "sweep": da_sweep(),
                 "heat": heat_curves(DA_LAB), "sep": separatrix(DA_LAB)},
         "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("stability-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote stability-results.json"); sys.exit()
    for key in ("lecture", "lab"):
        L = R[key]; print(key, "Da", L["Da"])
        for s in L["ss"]:
            print("  x*", r6(s["x"]), "lam", r6(s["lam_re"]), r6(s["lam_im"]), s["type"], "res", sig(s["residual"]))
    print("traj ends", [r6(q["end"]) for q in R["lecture"]["traj"]])
    print("lab kick ends", [r6(q["end"]) for q in R["lab"]["kick"]])
    print("tg", r6(R["lecture"]["tg"]["gmax"]), r6(R["lecture"]["tg"]["tmax"]))
    print("sweep", [(q["Da"], q["n"]) for q in R["lab"]["sweep"]])
