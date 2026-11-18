"""Session 2 (2026, semester 2): eigenvalue problems, diagonalization, the SVD and conditioning.

Course notation: M a general matrix; lambda an eigenvalue, v an eigenvector (M v = lambda v);
M = U Sigma V^T the singular value decomposition with sigma_1 >= sigma_2 >= ... >= 0;
kappa(M) = sigma_max / sigma_min the 2-norm condition number.

Run `python eigen_lab.py` for a printed summary, or `python eigen_lab.py --json`
to write eigen-results.json, which the slides and the lab page read.
"""
import json
import sys

import numpy as np
from scipy.linalg import eig, eigh, svd

R_GAS = 8.314462618          # J/(mol K)


def r6(v):
    """Round for printing, so recorded and live runs print the same text."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, n=4):
    """Significant-figure string, stable across platforms for printing."""
    return f"{float(v):.{n}g}"


# ---------------------------------------------------------------------------
# Part 1. Eigenvalues and eigenvectors of 2 x 2 matrices
# ---------------------------------------------------------------------------
SYM = [[3.0, 1.0], [1.0, 2.0]]          # symmetric: orthogonal eigenvectors
DEFECTIVE = [[2.0, 1.0], [0.0, 2.0]]    # one eigenvalue, one eigenvector
SHEAR = [[1.0, 2.0], [0.0, 1.0]]        # eigenvalues 1, 1; stretches by 2.414


def two_by_two():
    S = np.array(SYM); w, V = eigh(S)
    order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    for k in range(2):
        if V[0, k] < 0:
            V[:, k] = -V[:, k]
    D = np.array(DEFECTIVE); wd, Vd = eig(D)
    Sh = np.array(SHEAR); U, s, Vt = svd(Sh)
    we, _ = eig(Sh)
    return {"sym": {"M": SYM, "lam": w.tolist(), "V": V.T.tolist(), "trace": float(np.trace(S)), "det": float(np.linalg.det(S)),
                    "VtV": (V.T @ V).tolist()},
            "defective": {"M": DEFECTIVE, "lam": np.real(wd).tolist(), "rank_of_M_minus_2I": int(np.linalg.matrix_rank(D - 2 * np.eye(2))),
                          "cond_V": float(np.linalg.cond(Vd))},
            "shear": {"M": SHEAR, "lam": np.real(we).tolist(), "sv": s.tolist(), "cond": float(s[0] / s[1])}}


# ---------------------------------------------------------------------------
# Part 2. Identifiability of a first-order decay: the sensitivity matrix
# ---------------------------------------------------------------------------
DECAY = {"c0": 1.0, "k": 0.5}           # mol/L and 1/min


def decay_design(tmax, n=8):
    """Sensitivity matrix S = [dc/dc0, dc/dk] at n equally spaced times on (0, tmax]."""
    c0, k = DECAY["c0"], DECAY["k"]
    t = np.linspace(tmax / n, tmax, n)
    S = np.column_stack([np.exp(-k * t), -c0 * t * np.exp(-k * t)])
    s = svd(S, compute_uv=False)
    F = S.T @ S
    cov = np.linalg.inv(F)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    return {"tmax": float(tmax), "sv": s.tolist(), "cond": float(s[0] / s[1]), "corr": float(corr),
            "se_rel_k": float(np.sqrt(cov[1, 1]) / k)}       # relative standard error of k per unit noise


def decay_sweep():
    return [decay_design(float(t)) for t in np.round(np.arange(0.5, 12.01, 0.5), 2)]


# ---------------------------------------------------------------------------
# Part 3. Low-rank structure: synthetic axial temperature profiles (Eckart-Young)
# ---------------------------------------------------------------------------
def profiles():
    """20 runs x 25 axial positions built from two modes plus fixed noise (SYNTHETIC)."""
    rng = np.random.default_rng(7)
    z = np.linspace(0, 1, 25)
    m1 = np.exp(-((z - 0.3) / 0.15) ** 2)          # hot spot
    m2 = z                                          # linear rise
    a = rng.uniform(20, 60, 20); b = rng.uniform(5, 30, 20)
    X = 500 + np.outer(a, m1) + np.outer(b, m2)
    Xc = X - X.mean(axis=0)
    noise = rng.normal(0, 0.5, X.shape)
    Xn = Xc + noise
    U, s, Vt = svd(Xn, full_matrices=False)
    err = [float(np.linalg.norm(Xn - (U[:, :k] * s[:k]) @ Vt[:k], 2)) for k in range(0, 6)]
    return {"sv": s.tolist(), "err2": err, "noise_sd": 0.5, "shape": list(Xn.shape),
            "noise_level": float(0.5 * (np.sqrt(20) + np.sqrt(25)))}


# ---------------------------------------------------------------------------
# Part 4. Lab: Arrhenius conditioning, and the numerical rank of mixture spectra
# ---------------------------------------------------------------------------
ARR = {"T": [300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0],            # K
       "Ea": 75000.0, "lnA": np.log(1.0e9),                                   # J/mol, ln(1/s)
       # SYNTHETIC multiplicative errors on k (fixed, about +-3 %)
       "err": [0.021, -0.028, 0.012, 0.031, -0.017, -0.006, 0.024]}


def arrhenius():
    T = np.array(ARR["T"]); x = 1 / T
    lnk = ARR["lnA"] - ARR["Ea"] / (R_GAS * T) + np.log1p(np.array(ARR["err"]))
    out = {"T": T.tolist(), "k": np.exp(lnk).tolist(), "lnk": lnk.tolist()}
    Tm = 1 / x.mean()
    designs = {"raw": np.column_stack([np.ones_like(x), x]),
               "centered": np.column_stack([np.ones_like(x), x - x.mean()]),
               "scaled": np.column_stack([np.ones_like(x), (x - x.mean()) / (x - x.mean()).std()])}
    for key, X in designs.items():
        s = svd(X, compute_uv=False)
        beta, *_ = np.linalg.lstsq(X, lnk, rcond=None)
        resid = lnk - X @ beta
        s2 = resid @ resid / (len(T) - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        if key == "raw":
            Ea = -beta[1] * R_GAS; lnA = beta[0]
        elif key == "centered":
            Ea = -beta[1] * R_GAS; lnA = beta[0] - beta[1] * x.mean()
        else:
            slope = beta[1] / (x - x.mean()).std(); Ea = -slope * R_GAS; lnA = beta[0] - slope * x.mean()
        out[key] = {"cond": float(s[0] / s[1]), "sv": s.tolist(), "beta": beta.tolist(), "corr": float(corr),
                    "Ea_kJ": float(Ea / 1000), "lnA": float(lnA), "s_resid": float(np.sqrt(s2))}
    out["Tm"] = float(Tm)
    out["k_Tm"] = float(np.exp(out["centered"]["beta"][0]))
    return out


SPEC = {"species": 3, "mixtures": 15, "wavelengths": 60, "noise": 0.002,
        "centers": [270.0, 300.0, 335.0], "widths": [18.0, 22.0, 20.0], "heights": [0.9, 0.7, 0.3]}


def spectra(noise=None, seed=11):
    """Absorbance matrix X (mixtures x wavelengths) = C E + noise (Beer-Lambert, SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(240, 370, SPEC["wavelengths"])
    E = np.array([h * np.exp(-((lam - c) / w) ** 2) for c, w, h in zip(SPEC["centers"], SPEC["widths"], SPEC["heights"])])
    Cc = rng.uniform(0.05, 1.0, (SPEC["mixtures"], SPEC["species"]))
    Z = rng.normal(0, 1, (SPEC["mixtures"], SPEC["wavelengths"]))
    sd = SPEC["noise"] if noise is None else noise
    X = Cc @ E + sd * Z
    s = svd(X, compute_uv=False)
    delta = sd * (np.sqrt(X.shape[0]) + np.sqrt(X.shape[1]))     # expected largest singular value of the noise
    return {"noise": float(sd), "sv": s.tolist(), "delta": float(delta), "rank": int(np.sum(s > delta)),
            "lam": lam.tolist(), "E": E.tolist(), "X0": X[0].tolist(), "X1": X[1].tolist()}


def spectra_X(noise=None, seed=11):
    """The absorbance matrix itself (mixtures x wavelengths), for the Lab 2 code step."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(240, 370, SPEC["wavelengths"])
    E = np.array([h * np.exp(-((lam - c) / w) ** 2) for c, w, h in zip(SPEC["centers"], SPEC["widths"], SPEC["heights"])])
    Cc = rng.uniform(0.05, 1.0, (SPEC["mixtures"], SPEC["species"]))
    Z = rng.normal(0, 1, (SPEC["mixtures"], SPEC["wavelengths"]))
    return Cc @ E + (SPEC["noise"] if noise is None else noise) * Z


def low_rank():
    """Numerical rank from the noise threshold, and the Eckart-Young error of the rank-r reconstruction."""
    X = spectra_X(); m, n = X.shape
    U, s, Vt = svd(X, full_matrices=False)
    delta = SPEC["noise"] * (np.sqrt(m) + np.sqrt(n)); r = int(np.sum(s > delta))
    Xr = (U[:, :r] * s[:r]) @ Vt[:r]
    rms = float(np.linalg.norm(X - Xr) / np.sqrt(m * n))
    return {"rank": r, "rms": rms, "tail": float(np.sqrt(np.sum(s[r:] ** 2)) / np.sqrt(m * n)),
            "expected": float(SPEC["noise"] * np.sqrt((m - r) * (n - r) / (m * n)))}


def spectra_sweep():
    return [{"noise": float(v), **{k: spectra(float(v))[k] for k in ("sv", "delta", "rank")}}
            for v in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]]


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
    R = {"lecture": {"two": two_by_two(), "decay": decay_sweep(), "profiles": profiles()},
         "lab": {"arr": arrhenius(), "spec": spectra(), "sweep": spectra_sweep(), "lowrank": low_rank()},
         "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("eigen-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote eigen-results.json"); sys.exit()
    T = R["lecture"]["two"]
    print("1 sym lam", r6(T["sym"]["lam"]), "V", r6(T["sym"]["V"]))
    print("1 defective lam", r6(T["defective"]["lam"]), "rank(M-2I)", T["defective"]["rank_of_M_minus_2I"], "cond V", sig(T["defective"]["cond_V"]))
    print("1 shear lam", r6(T["shear"]["lam"]), "sv", r6(T["shear"]["sv"]), "cond", r6(T["shear"]["cond"]))
    for d in R["lecture"]["decay"][::4]:
        print("2 decay tmax", d["tmax"], "cond", sig(d["cond"]), "corr", r6(d["corr"]), "se_rel_k", r6(d["se_rel_k"]))
    P = R["lecture"]["profiles"]; print("3 profiles sv", [sig(v) for v in P["sv"][:6]], "noise level", sig(P["noise_level"]), "err", [sig(v) for v in P["err2"]])
    A = R["lab"]["arr"]
    for k in ("raw", "centered", "scaled"):
        print("4", k, "cond", sig(A[k]["cond"]), "corr", r6(A[k]["corr"]), "Ea", r6(A[k]["Ea_kJ"]), "lnA", r6(A[k]["lnA"]))
    Sp = R["lab"]["spec"]; print("4 spectra sv", [sig(v) for v in Sp["sv"][:6]], "delta", sig(Sp["delta"]), "rank", Sp["rank"])
    print("4 sweep", [(q["noise"], q["rank"]) for q in R["lab"]["sweep"]])
