"""Session 3 (2026, semester 2): linear ODE systems, the matrix exponential, and transfer functions.

Course notation: dx/dt = K x, x(0) = x0; K = V Lambda V^-1 with eigenvalues lambda_i (Session 2);
x(t) = expm(K t) x0 = sum_i c_i exp(lambda_i t) v_i with c = V^-1 x0. Time constants tau_i = -1/Re(lambda_i).

Run `python linode_lab.py` for a printed summary, or `python linode_lab.py --json`
to write linode-results.json, which the slides and the lab page read.
"""
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig, expm, null_space


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, n=4):
    return f"{float(v):.{n}g}"


def eig_sorted(K):
    """Eigenvalues sorted by real part, descending (0 first for a closed network), real when the imaginary part is negligible."""
    w, V = eig(K)
    order = np.argsort(-w.real); w, V = w[order], V[:, order]
    if np.all(np.abs(w.imag) < 1e-12):
        w, V = w.real, V.real
    return w, V


# ---------------------------------------------------------------------------
# Part 1. Consecutive reactions A -> B -> C
# ---------------------------------------------------------------------------
CONSEC = {"k1": 1.0, "k2": 0.5}          # 1/min


def consecutive():
    k1, k2 = CONSEC["k1"], CONSEC["k2"]
    K = np.array([[-k1, 0, 0], [k1, -k2, 0], [0, k2, 0]])
    w, V = eig_sorted(K)
    x0 = np.array([1.0, 0, 0])
    t = np.linspace(0, 10, 101)
    X = np.array([expm(K * s) @ x0 for s in t])
    sol = solve_ivp(lambda s, x: K @ x, (0, 10), x0, t_eval=t, rtol=1e-10, atol=1e-12)
    tmax = np.log(k1 / k2) / (k1 - k2)
    left = null_space(K.T).ravel(); left = left / left[0]
    return {"K": K.tolist(), "lam": w.tolist(), "V": V.T.tolist(), "t": t.tolist(), "X": X.tolist(),
            "max_diff_ivp": float(np.abs(sol.y.T - X).max()), "tmaxB": float(tmax), "Bmax": float(expm(K * tmax)[1, 0]),
            "left_null": left.tolist()}


# ---------------------------------------------------------------------------
# Part 2. Complex eigenvalues: an underdamped second-order process
# ---------------------------------------------------------------------------
def oscillator(zeta=0.2, wn=1.0):
    K = np.array([[0, 1], [-wn ** 2, -2 * zeta * wn]])
    w, _ = eig(K)
    t = np.linspace(0, 20, 201)
    X = np.array([expm(K * s) @ np.array([1.0, 0.0]) for s in t])
    return {"zeta": zeta, "lam_re": float(w[0].real), "lam_im": float(abs(w[0].imag)), "t": t.tolist(), "y": X[:, 0].tolist(),
            "period": float(2 * np.pi / abs(w[0].imag)), "decay": float(np.exp(2 * np.pi * w[0].real / abs(w[0].imag)))}


# ---------------------------------------------------------------------------
# Part 3. Two CSTRs in series: transfer function, poles, step response
# ---------------------------------------------------------------------------
TANKS = {"tau1": 2.0, "tau2": 1.0}       # min


def tanks():
    t1, t2 = TANKS["tau1"], TANKS["tau2"]
    K = np.array([[-1 / t1, 0], [1 / t2, -1 / t2]])
    b = np.array([1 / t1, 0])
    w, _ = eig_sorted(K)
    t = np.linspace(0, 12, 121)
    # step response y(t) = c2(t) for a unit step in the feed: x(t) = K^-1 (expm(Kt) - I) b
    y = [float((np.linalg.solve(K, (expm(K * s) - np.eye(2)) @ b))[1]) for s in t]
    exact = [1 - (t1 * np.exp(-s / t1) - t2 * np.exp(-s / t2)) / (t1 - t2) for s in t]
    return {"poles": w.tolist(), "t": t.tolist(), "y": y, "max_diff": float(np.max(np.abs(np.array(y) - exact))),
            "t63": float(t[np.argmax(np.array(y) >= 1 - np.exp(-1))])}


# ---------------------------------------------------------------------------
# Part 4. Stiffness: explicit Euler on eigenvalues -1 and -1000
# ---------------------------------------------------------------------------
def stiffness():
    K = np.array([[-1.0, 0.0], [999.0, -1000.0]])
    w, _ = eig_sorted(K)
    x0 = np.array([1.0, 0.0])
    out = []
    for h in (0.0005, 0.0019, 0.0021, 0.01):
        x = x0.copy(); n = int(round(0.1 / h))
        for _ in range(n):
            x = x + h * (K @ x)
        xi = x0.copy()
        Mi = np.linalg.inv(np.eye(2) - h * K)
        for _ in range(n):
            xi = Mi @ xi
        ex = expm(K * n * h) @ x0
        out.append({"h": h, "t": n * h, "euler": x.tolist(), "implicit": xi.tolist(), "exact": ex.tolist()})
    return {"lam": w.tolist(), "ratio": float(w.min() / w.max()), "hmax": float(2 / abs(w.min())), "runs": out}


# ---------------------------------------------------------------------------
# Part 5. Lab: butene isomerization network (reversible triangle, detailed balance)
# ---------------------------------------------------------------------------
# Species 1-butene (1), cis-2-butene (2), trans-2-butene (3). Equilibrium mole fractions and the
# symmetric "conductances" c_ij = k_ij x_i* = k_ji x_j* are SYNTHETIC values chosen to resemble the
# network Wei and Prater (1962) analyzed; units of c: 1/h.
BUT = {"names": ["1-butene", "cis-2-butene", "trans-2-butene"], "xeq": [0.15, 0.30, 0.55],
       "c": {"12": 0.60, "13": 0.15, "23": 0.30}}


def rate_matrix(c13=None):
    x = np.array(BUT["xeq"]); c = dict(BUT["c"])
    if c13 is not None:
        c["13"] = c13
    k = np.zeros((3, 3))                     # k[i, j]: rate constant of i -> j
    for key, v in c.items():
        i, j = int(key[0]) - 1, int(key[1]) - 1
        k[i, j] = v / x[i]; k[j, i] = v / x[j]
    K = k.T - np.diag(k.sum(axis=1))
    return K, k


def butene(c13=None):
    K, k = rate_matrix(c13)
    w, V = eig_sorted(K)
    xeq = np.array(BUT["xeq"])
    for i in range(3):
        V[:, i] /= np.abs(V[:, i]).max() * np.sign(V[np.argmax(np.abs(V[:, i])), i])
    taus = [float(-1 / w[i]) for i in (1, 2)]
    runs = {}
    t = np.linspace(0, 3, 151)
    for s, x0 in (("pure1", [1, 0, 0]), ("pure2", [0, 1, 0]), ("pure3", [0, 0, 1]),
                  ("slow", (xeq + 0.12 * V[:, 1]).tolist()), ("fast", (xeq + 0.12 * V[:, 2]).tolist())):
        X = np.array([expm(K * u) @ np.array(x0, float) for u in t])
        runs[s] = {"x0": list(map(float, x0)), "X": X.tolist()}
    sol = solve_ivp(lambda s, x: K @ x, (0, 0.25), [1, 0, 0], t_eval=[0.25], rtol=1e-11, atol=1e-13, method="Radau")
    x2 = expm(K * 0.25) @ np.array([1.0, 0, 0])
    return {"K": K.tolist(), "k": k.tolist(), "lam": w.tolist(), "V": V.T.tolist(), "taus": taus, "ratio": float(w[2] / w[1]),
            "t": t.tolist(), "runs": runs, "x_at2": x2.tolist(), "ivp_at2": sol.y[:, 0].tolist(),
            "colsum": float(np.abs(K.sum(axis=0)).max())}


def butene_sweep():
    out = []
    for c13 in np.round(np.arange(0.0, 1.201, 0.05), 2):
        K, _ = rate_matrix(float(c13)); w, _ = eig_sorted(K)
        out.append({"c13": float(c13), "lam": w.tolist(), "ratio": float(w[2] / w[1])})
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
    R = {"lecture": {"consec": consecutive(), "osc": oscillator(), "tanks": tanks(), "stiff": stiffness()},
         "lab": {"but": butene(), "sweep": butene_sweep(), "names": BUT["names"], "xeq": BUT["xeq"], "c": BUT["c"]},
         "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("linode-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote linode-results.json"); sys.exit()
    C = R["lecture"]["consec"]
    print("1 consec lam", r6(C["lam"]), "left null", r6(C["left_null"]), "tmaxB", r6(C["tmaxB"]), "Bmax", r6(C["Bmax"]), "ivp diff", sig(C["max_diff_ivp"]))
    O = R["lecture"]["osc"]; print("2 osc lam", r6(O["lam_re"]), r6(O["lam_im"]), "period", r6(O["period"]), "decay/cycle", r6(O["decay"]))
    T = R["lecture"]["tanks"]; print("3 tanks poles", r6(T["poles"]), "t63", r6(T["t63"]), "diff", sig(T["max_diff"]))
    S = R["lecture"]["stiff"]; print("4 stiff lam", r6(S["lam"]), "hmax", r6(S["hmax"]))
    for q in S["runs"]:
        print("  h", q["h"], "euler", [sig(v) for v in q["euler"]], "implicit", [sig(v) for v in q["implicit"]], "exact", [sig(v) for v in q["exact"]])
    B = R["lab"]["but"]; print("5 butene lam", r6(B["lam"]), "taus", r6(B["taus"]), "ratio", r6(B["ratio"]), "x(2)", r6(B["x_at2"]), "ivp", r6(B["ivp_at2"]))
    print("  V", r6(B["V"]))
    print("  sweep", [(q["c13"], round(q["ratio"], 3)) for q in R["lab"]["sweep"]][::4])
