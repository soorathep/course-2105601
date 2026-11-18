"""Session 5 (2026, semester 2): bifurcation and multiplicity, by continuation.

Model: the dimensionless cooled CSTR of Session 4 (Uppal, Ray and Poore, 1974, large activation energy):
    f1 = -x1 + Da (1 - x1) exp(x2),  f2 = -x2 + B Da (1 - x1) exp(x2) - beta x2.
Steady states satisfy x1 = (1 + beta) x2 / B, so Da(x2) = x1 exp(-x2) / (1 - x1) gives the whole S-curve in closed form;
the pseudo-arclength continuation below finds the same curve without that shortcut, as it would for a model without one.
Folds (turning points): det J = 0. Hopf points: tr J = 0 with det J > 0. Multiplicity requires B > 4 (1 + beta).

Run `python bifurcation_lab.py` for a summary, or `python bifurcation_lab.py --json` to write bifurcation-results.json.
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, n=4):
    return f"{float(v):.{n}g}"


def f(x, Da, B, beta):
    r = Da * (1 - x[0]) * np.exp(x[1])
    return np.array([-x[0] + r, -x[1] + B * r - beta * x[1]])


def jac(x, Da, B, beta):
    e = np.exp(x[1])
    return np.array([[-1 - Da * e, Da * (1 - x[0]) * e], [-B * Da * e, -1 - beta + B * Da * (1 - x[0]) * e]])


def dfdDa(x, Da, B, beta):
    r = (1 - x[0]) * np.exp(x[1])
    return np.array([r, B * r])


# ---------------------------------------------------------------------------
# Closed-form S-curve and its special points
# ---------------------------------------------------------------------------
def scurve(B, beta, n=400):
    c = (1 + beta) / B
    x2 = np.linspace(1e-3, 1 / c - 1e-3, n); x1 = c * x2
    Da = x1 * np.exp(-x2) / (1 - x1)
    tr, det = [], []
    for a, b, d in zip(x1, x2, Da):
        J = jac([a, b], d, B, beta); tr.append(np.trace(J)); det.append(np.linalg.det(J))
    return {"x2": x2.tolist(), "x1": x1.tolist(), "Da": Da.tolist(), "tr": tr, "det": det}


def special_points(B, beta):
    """Folds from c x2^2 - x2 + 1 = 0 (c = (1 + beta)/B); Hopf from tr J = 0 with det J > 0 on the curve."""
    c = (1 + beta) / B
    out = {"cusp_B": 4 * (1 + beta), "folds": [], "hopf": []}
    disc = 1 - 4 * c
    if disc > 0:
        for x2 in sorted(((1 - np.sqrt(disc)) / (2 * c), (1 + np.sqrt(disc)) / (2 * c))):
            x1 = c * x2; Da = x1 * np.exp(-x2) / (1 - x1)
            out["folds"].append({"x2": float(x2), "x1": float(x1), "Da": float(Da), "det": float(np.linalg.det(jac([x1, x2], Da, B, beta)))})
    def trace_at(x2):
        x1 = c * x2; Da = x1 * np.exp(-x2) / (1 - x1); return np.trace(jac([x1, x2], Da, B, beta))
    xs = np.linspace(1e-3, 1 / c - 1e-3, 4001); tv = [trace_at(x) for x in xs]
    for i in range(len(xs) - 1):
        if tv[i] * tv[i + 1] < 0:
            x2 = brentq(trace_at, xs[i], xs[i + 1], xtol=1e-13); x1 = c * x2; Da = x1 * np.exp(-x2) / (1 - x1)
            d = np.linalg.det(jac([x1, x2], Da, B, beta))
            if d > 0:
                out["hopf"].append({"x2": float(x2), "x1": float(x1), "Da": float(Da), "omega": float(np.sqrt(d)), "period": float(2 * np.pi / np.sqrt(d))})
    return out


# ---------------------------------------------------------------------------
# Pseudo-arclength continuation in (x1, x2, Da)
# ---------------------------------------------------------------------------
def continuation(B, beta, Da0=0.01, ds=0.02, nmax=4000, Da_max=0.3):
    """Follow the steady-state branch from the cold state at Da0 through both folds (Keller's method)."""
    c = (1 + beta) / B
    g = lambda x1: x1 - Da0 * (1 - x1) * np.exp(x1 / c)
    x1 = brentq(g, 0, 0.5); y = np.array([x1, x1 / c, Da0])
    F = lambda y: f(y[:2], y[2], B, beta)
    def Fy(y):
        return np.column_stack([jac(y[:2], y[2], B, beta), dfdDa(y[:2], y[2], B, beta)])
    t = np.array([0.0, 0.0, 1.0]); A = Fy(y)
    t = np.linalg.svd(A)[2][-1]; t = t if t[2] > 0 else -t
    pts, steps, newton_its = [y.copy()], 0, []
    while steps < nmax and y[2] < Da_max and y[0] < 0.995:
        yp = y + ds * t; z = yp.copy()
        for it in range(20):
            G = np.concatenate([F(z), [t @ (z - y) - ds]])
            M = np.vstack([Fy(z), t])
            dz = np.linalg.solve(M, -G); z = z + dz
            if np.abs(dz).max() < 1e-12:
                break
        newton_its.append(it + 1)
        tn = np.linalg.svd(Fy(z))[2][-1]; tn = tn if tn @ t > 0 else -tn
        y, t = z, tn; pts.append(y.copy()); steps += 1
    P = np.array(pts)
    dets = [float(np.linalg.det(jac(p[:2], p[2], B, beta))) for p in P]
    trs = [float(np.trace(jac(p[:2], p[2], B, beta))) for p in P]
    # check against the closed form Da(x2)
    err = float(np.max(np.abs(P[:, 2] - (c * P[:, 1]) * np.exp(-P[:, 1]) / (1 - c * P[:, 1]))))
    fold_idx = [i for i in range(len(dets) - 1) if dets[i] * dets[i + 1] < 0]
    hopf_idx = [i for i in range(len(trs) - 1) if trs[i] * trs[i + 1] < 0 and dets[i] > 0]
    keep = slice(None, None, max(1, len(P) // 400))
    return {"Da": P[keep, 2].tolist(), "x1": P[keep, 0].tolist(), "x2": P[keep, 1].tolist(), "steps": steps,
            "max_newton": int(max(newton_its)), "err_closed_form": err,
            "folds_Da": [float(P[i, 2]) for i in fold_idx], "hopf_Da": [float(P[i, 2]) for i in hopf_idx]}


# ---------------------------------------------------------------------------
# Hysteresis: slow ramp of Da up and down, by simulation
# ---------------------------------------------------------------------------
def hysteresis(B, beta, Da_lo=0.02, Da_hi=0.09, n=57, hold=40.0):
    up, down = [], []
    x = np.array([0.0, 0.0])
    grid = np.linspace(Da_lo, Da_hi, n)
    for Da in grid:
        x = solve_ivp(lambda t, z: f(z, Da, B, beta), (0, hold), x, rtol=1e-9, atol=1e-11, method="LSODA").y[:, -1]
        up.append(float(x[1]))
    for Da in grid[::-1]:
        x = solve_ivp(lambda t, z: f(z, Da, B, beta), (0, hold), x, rtol=1e-9, atol=1e-11, method="LSODA").y[:, -1]
        down.append(float(x[1]))
    return {"Da": grid.tolist(), "up": up, "down": down[::-1]}


# ---------------------------------------------------------------------------
# Limit cycles beyond the Hopf point
# ---------------------------------------------------------------------------
def oscillation(B, beta, Da, tmax=200.0):
    sol = solve_ivp(lambda t, z: f(z, Da, B, beta), (0, tmax), [0.5, 3.0], rtol=1e-9, atol=1e-11, method="LSODA", dense_output=True)
    t = np.linspace(tmax - 40, tmax, 801); X = sol.sol(t)
    x2 = X[1]
    pk = [i for i in range(1, len(x2) - 1) if x2[i] > x2[i - 1] and x2[i] >= x2[i + 1] and x2[i] > x2.mean()]
    period = float(np.mean(np.diff(t[pk]))) if len(pk) > 1 else None
    amp = float(x2.max() - x2.min())
    return {"Da": Da, "t": (t - t[0]).tolist(), "x1": X[0].tolist(), "x2": x2.tolist(), "period": period, "amp": amp,
            "x2max": float(x2.max()), "x2min": float(x2.min())}


def amplitude_sweep(B, beta, Das):
    return [{"Da": float(d), "amp": oscillation(B, beta, float(d))["amp"]} for d in Das]


def beta_sweep(B, betas):
    out = []
    for b in betas:
        sp = special_points(B, b)
        out.append({"beta": float(b), "folds": [q["Da"] for q in sp["folds"]], "hopf": [q["Da"] for q in sp["hopf"]], "cusp_B": sp["cusp_B"]})
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


LECT = {"B": 8.0, "beta": 0.3}
LAB = {"B": 14.0, "beta": 2.0}


def classroom_results():
    Bl, bl = LECT["B"], LECT["beta"]
    Bk, bk = LAB["B"], LAB["beta"]
    R = {"lecture": {"P": LECT, "s": scurve(Bl, bl), "sp": special_points(Bl, bl), "cont": continuation(Bl, bl, Da_max=0.12),
                     "hyst": hysteresis(Bl, bl),
                     "cusp": [{"beta": float(b), "B": 4 * (1 + b)} for b in np.linspace(0, 3, 31)]},
         "lab": {"P": LAB, "s": scurve(Bk, bk), "sp": special_points(Bk, bk), "cont": continuation(Bk, bk, Da_max=0.2),
                 "osc": oscillation(Bk, bk, 0.12), "amp": amplitude_sweep(Bk, bk, [0.11, 0.115, 0.12, 0.125, 0.128, 0.13]),
                 "beta": beta_sweep(Bk, np.round(np.arange(1.0, 3.01, 0.1), 2))},
         "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("bifurcation-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote bifurcation-results.json"); sys.exit()
    for k in ("lecture", "lab"):
        L = R[k]; sp = L["sp"]
        print(k, L["P"], "cusp B", sp["cusp_B"], "folds", [(r6(q["Da"]), r6(q["x2"])) for q in sp["folds"]], "hopf", [(r6(q["Da"]), r6(q["x2"]), r6(q["period"])) for q in sp["hopf"]])
        c = L["cont"]; print("  continuation steps", c["steps"], "max newton", c["max_newton"], "err", sig(c["err_closed_form"]), "folds", r6(c["folds_Da"]), "hopf", r6(c["hopf_Da"]))
    H = R["lecture"]["hyst"]; print("hyst jump up at", [H["Da"][i] for i in range(1, len(H["up"])) if H["up"][i] - H["up"][i-1] > 1][:1], "down at", [H["Da"][i] for i in range(len(H["down"]) - 1) if H["down"][i + 1] - H["down"][i] > 1][:1])
    O = R["lab"]["osc"]; print("osc Da", O["Da"], "period", r6(O["period"]), "amp", r6(O["amp"]), O["x2min"], O["x2max"])
    print("amp", [(q["Da"], r6(q["amp"])) for q in R["lab"]["amp"]])
    print("beta", [(q["beta"], r6(q["folds"]), r6(q["hopf"])) for q in R["lab"]["beta"]][::5])
