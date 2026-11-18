"""Session 1 (2026, semester 2): vector spaces, linear operators and the four fundamental subspaces.

Course notation, used in every function below:
    A  atom (element) matrix, elements x species: A[e, s] = atoms of element e in species s
    N  stoichiometric matrix, species x reactions: N[s, j] = coefficient of species s in reaction j
       (products positive, reactants negative)
    n  molar amounts or flows of the species; xi  extents of reaction;  n = n0 + N xi
Element conservation is A n = A n0, so every column of N lies in the null space of A.

Run `python subspace_lab.py` for a printed summary, or `python subspace_lab.py --json`
to write subspace-results.json, which the slides and the lab page read.
"""
import json
import sys
from fractions import Fraction

import numpy as np
from scipy.linalg import null_space, orth, svdvals
from scipy.stats import chi2 as chi2_dist

TOL = 1e-10          # numerical rank threshold relative to the largest singular value


def r6(v):
    """Round for printing, so recorded and live runs print the same text."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def num_rank(M, tol=TOL):
    """Numerical rank: number of singular values above tol * largest."""
    s = svdvals(np.asarray(M, float))
    return int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0


def rref(M):
    """Exact reduced row echelon form with fractions; returns (R, pivot columns)."""
    R = [[Fraction(x) for x in row] for row in M]
    m, n = len(R), len(R[0])
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if R[i][c] != 0), None)
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        R[r] = [x / R[r][c] for x in R[r]]
        for i in range(m):
            if i != r and R[i][c] != 0:
                R[i] = [a - R[i][c] * b for a, b in zip(R[i], R[r])]
        piv.append(c); r += 1
        if r == m:
            break
    return R, piv


def exact_null_basis(M):
    """Integer null-space basis from the RREF: one vector per free column."""
    R, piv = rref(M)
    n = len(M[0]); free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        v = [Fraction(0)] * n; v[f] = Fraction(1)
        for i, p in enumerate(piv):
            v[p] = -R[i][f]
        den = 1
        for x in v:
            den = den * x.denominator // np.gcd(den, x.denominator)
        w = [int(x * den) for x in v]
        if next(x for x in w if x != 0) < 0:
            w = [-x for x in w]
        out.append(w)
    return {"rref": [[str(x) for x in row] for row in R], "pivots": piv, "free": free, "basis": out}


def subspaces(M):
    """Dimensions of the four fundamental subspaces of M (m x n) and orthonormal bases."""
    M = np.asarray(M, float); m, n = M.shape; r = num_rank(M)
    return {"shape": [m, n], "rank": r, "dim_col": r, "dim_row": r, "dim_null": n - r, "dim_left_null": m - r,
            "sv": svdvals(M).tolist(),
            "null": null_space(M, rcond=TOL).T.tolist(), "left_null": null_space(M.T, rcond=TOL).T.tolist(),
            "col": orth(M, rcond=TOL).T.tolist()}


# ---------------------------------------------------------------------------
# Part 1. Lecture example: methanol synthesis
# ---------------------------------------------------------------------------
MEOH = {"species": ["CO", "CO2", "H2", "CH3OH", "H2O"], "elements": ["C", "H", "O"],
        "A": [[1, 1, 0, 1, 0],
              [0, 0, 2, 4, 2],
              [1, 2, 0, 1, 1]],
        "reactions": {"CO + 2 H2 -> CH3OH": [-1, 0, -2, 1, 0],
                      "CO2 + H2 -> CO + H2O": [1, -1, -1, 0, 1],
                      "CO2 + 3 H2 -> CH3OH + H2O": [0, -1, -3, 1, 1]}}


MEOH_REC = {"n0": [1.0, 0.3, 3.0, 0.0, 0.0],          # feed CO, CO2, H2, CH3OH, H2O, mol/s
            "xi_true": [0.4, 0.1],                     # extents of CO + 2 H2 -> CH3OH and of reverse WGS (synthetic)
            # SYNTHETIC measured outlet, mol/s, and the stated meter standard deviations
            "measured": [0.71, 0.19, 2.16, 0.38, 0.11],
            "sigma": [0.01, 0.01, 0.05, 0.01, 0.01]}


def methanol_reconciliation(sigma_h2=None):
    """Reconcile the synthetic methanol-reactor measurements; sigma_h2 overrides the H2 meter's sigma."""
    A = np.array(MEOH["A"], float); n0 = np.array(MEOH_REC["n0"]); b = A @ n0
    sig = list(MEOH_REC["sigma"])
    if sigma_h2 is not None:
        sig[2] = sigma_h2
    N2 = np.array(list(MEOH["reactions"].values())[:2], float).T
    out = {"n_true": (n0 + N2 @ np.array(MEOH_REC["xi_true"])).tolist(), "b": b.tolist(),
           "unweighted": reconcile(A, MEOH_REC["measured"], b), "weighted": reconcile(A, MEOH_REC["measured"], b, sig)}
    for k in ("unweighted", "weighted"):
        xi, *_ = np.linalg.lstsq(N2, np.array(out[k]["n_hat"]) - n0, rcond=None)
        out[k]["xi"] = xi.tolist()
    return out


def methanol():
    A = np.array(MEOH["A"], float)
    N = np.array(list(MEOH["reactions"].values()), float).T          # species x reactions
    SA, SN = subspaces(A), subspaces(N)
    # conserved quantities: left null space of N equals the row space of A when the reactions span null(A)
    LN = np.array(SN["left_null"]); proj = LN.T @ np.linalg.lstsq(LN.T, A.T, rcond=None)[0]
    return {"A": SA, "N": SN, "AN_max": float(np.abs(A @ N).max()), "exact": exact_null_basis(MEOH["A"]),
            "rowA_in_leftnullN": float(np.abs(proj - A.T).max()),
            "null_N_integer": exact_null_basis(N.astype(int).tolist())["basis"]}


# ---------------------------------------------------------------------------
# Part 2. Operators on function spaces: d/dx on polynomials, Gram matrices
# ---------------------------------------------------------------------------
def derivative_operator(deg=3):
    """Matrix of d/dx on polynomials of degree <= deg in the monomial basis 1, x, ..., x^deg."""
    D = np.zeros((deg + 1, deg + 1))
    for k in range(1, deg + 1):
        D[k - 1, k] = k
    S = subspaces(D)
    return {"deg": deg, "D": D.tolist(), "rank": S["rank"], "dim_null": S["dim_null"], "dim_left_null": S["dim_left_null"]}


def gram_monomials(n, a=0.0, b=1.0):
    """Gram matrix <x^i, x^j> = integral_a^b x^(i+j) dx for i, j < n (the Hilbert matrix on [0, 1])."""
    G = np.array([[(b ** (i + j + 1) - a ** (i + j + 1)) / (i + j + 1) for j in range(n)] for i in range(n)])
    return G


def legendre_by_gram_schmidt(n=4):
    """Orthogonalize 1, x, x^2, ... on [-1, 1] through the Cholesky factor of the Gram matrix.
    Returns monomial coefficients scaled so that p_k(1) = 1 (the Legendre polynomials)."""
    G = gram_monomials(n, -1.0, 1.0)
    L = np.linalg.cholesky(G)
    C = np.linalg.inv(L).T                    # columns: orthonormal polynomials in the monomial basis
    out = []
    for k in range(n):
        c = C[:, k]; c = c / c.sum()          # p_k(1) = sum of coefficients
        out.append([0.0 if abs(x) < 1e-12 else float(x) for x in c])
    return out


def hilbert_conditioning(nmax=10):
    return [{"n": n, "cond": float(np.linalg.cond(gram_monomials(n)))} for n in range(2, nmax + 1)]


# ---------------------------------------------------------------------------
# Part 3. Lab: steam methane reforming network and data reconciliation
# ---------------------------------------------------------------------------
SMR = {"species": ["CH4", "H2O", "CO", "H2", "CO2"], "elements": ["C", "H", "O"],
       "A": [[1, 0, 1, 0, 1],
             [4, 2, 0, 2, 0],
             [0, 1, 1, 0, 2]],
       "reactions": {"R1  CH4 + H2O -> CO + 3 H2": [-1, -1, 1, 3, 0],
                     "R2  CO + H2O -> CO2 + H2": [0, -1, -1, 1, 1],
                     "R3  CH4 + 2 H2O -> CO2 + 4 H2": [-1, -2, 0, 4, 1]},
       "n0": [1.0, 3.0, 0.0, 0.0, 0.0],             # feed, mol/s (steam-to-carbon ratio 3)
       "xi_true": [0.70, 0.30],                     # extents of R1 and R2, mol/s (synthetic case)
       # SYNTHETIC measurements of the outlet, mol/s: true outlet plus fixed errors of about 2 to 5 %
       "measured": [0.31, 1.95, 0.42, 2.33, 0.29],
       "sigma": [0.01, 0.05, 0.01, 0.05, 0.01]}     # stated standard deviations of the meters, mol/s

EXT = {"species": ["CH4", "H2O", "CO", "H2", "CO2", "C(s)", "CH3OH"],
       "A": [[1, 0, 1, 0, 1, 1, 1],
             [4, 2, 0, 2, 0, 0, 4],
             [0, 1, 1, 0, 2, 0, 1]]}


def reconcile(A, m, b, sigma=None):
    """Minimize sum ((n - m)_s / sigma_s)^2 subject to A n = b. Closed form:
    n_hat = m - S A^T (A S A^T)^-1 (A m - b), with S = diag(sigma^2) (S = I when sigma is None)."""
    A = np.asarray(A, float); m = np.asarray(m, float); b = np.asarray(b, float)
    S = np.eye(len(m)) if sigma is None else np.diag(np.asarray(sigma, float) ** 2)
    r = A @ m - b
    lam = np.linalg.solve(A @ S @ A.T, r)
    nh = m - S @ A.T @ lam
    return {"n_hat": nh.tolist(), "imbalance_before": r.tolist(), "imbalance_after": (A @ nh - b).tolist(),
            "correction": (nh - m).tolist(), "chi2": float(np.sum(((nh - m) ** 2) / np.diag(S))),
            "dof": int(num_rank(A)), "chi2_crit95": float(chi2_dist.ppf(0.95, num_rank(A)))}


GROSS = {"meter": 2, "bias": 0.06}           # SYNTHETIC: the CO meter reads 0.06 mol/s high


def gross_error(meter=GROSS["meter"], bias=GROSS["bias"]):
    """Measurement test (Narasimhan and Jordache): z_j = |a_j| / sqrt(V_jj) with adjustments a = n_hat - m
    and their covariance V = S A^T (A S A^T)^-1 A S, for the SMR data with one biased meter."""
    A = np.array(SMR["A"], float); n0 = np.array(SMR["n0"]); b = A @ n0
    m = np.array(SMR["measured"], float); m[meter] += bias
    S = np.diag(np.array(SMR["sigma"], float) ** 2)
    W = S @ A.T @ np.linalg.inv(A @ S @ A.T)
    a = -W @ (A @ m - b); V = W @ A @ S
    z = np.abs(a) / np.sqrt(np.diag(V))
    return {"m": m.tolist(), "z": z.tolist(), "suspect": int(np.argmax(z)), "z_max": float(z.max()),
            "chi2": float(np.sum(a ** 2 / np.diag(S))), "chi2_crit95": float(chi2_dist.ppf(0.95, 3))}


def lab_results():
    A = np.array(SMR["A"], float); N = np.array(list(SMR["reactions"].values()), float).T
    n0 = np.array(SMR["n0"]); b = A @ n0
    Nind = N[:, :2]
    n_true = n0 + Nind @ np.array(SMR["xi_true"])
    SA, SN = subspaces(A), subspaces(N)
    rec_u = reconcile(A, SMR["measured"], b)
    rec_w = reconcile(A, SMR["measured"], b, SMR["sigma"])
    out = {}
    for key, rec in (("unweighted", rec_u), ("weighted", rec_w)):
        dn = np.array(rec["n_hat"]) - n0
        xi, *_ = np.linalg.lstsq(Nind, dn, rcond=None)
        P = np.eye(5) - A.T @ np.linalg.solve(A @ A.T, A)                 # projector onto null(A)
        rec["xi"] = xi.tolist()
        rec["xi_residual"] = float(np.linalg.norm(Nind @ xi - dn))
        rec["CH4_conversion"] = float(1 - rec["n_hat"][0] / n0[0])
        rec["H2_yield"] = float(rec["n_hat"][3] / n0[0])
        rec["orth_check"] = float(np.abs(P @ np.array(rec["correction"])).max()) if key == "unweighted" else None
        out[key] = rec
    m = np.array(SMR["measured"])
    xi_raw, *_ = np.linalg.lstsq(Nind, m - n0, rcond=None)
    SE = subspaces(EXT["A"])
    return {"A": SA, "N": SN, "AN_max": float(np.abs(A @ N).max()), "exact": exact_null_basis(SMR["A"]),
            "R3_minus_R1_R2": (N[:, 2] - N[:, 0] - N[:, 1]).tolist(),
            "n_true": n_true.tolist(), "b": b.tolist(), "measured": SMR["measured"], "sigma": SMR["sigma"],
            "rec": out, "gross": gross_error(), "xi_raw": xi_raw.tolist(), "raw_residual": float(np.linalg.norm(Nind @ xi_raw - (m - n0))),
            "CH4_conversion_raw": float(1 - m[0] / n0[0]),
            "extended": {"species": EXT["species"], "rank": SE["rank"], "dim_null": SE["dim_null"],
                         "basis": exact_null_basis(EXT["A"])["basis"]}}


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
    R = {"lecture": {"methanol": methanol(), "derivative": derivative_operator(3),
                     "legendre": legendre_by_gram_schmidt(4), "hilbert": hilbert_conditioning(10),
                     "meoh_rec": methanol_reconciliation(),
                     "meoh_sweep": [{"s": float(v), "n_hat": methanol_reconciliation(float(v))["weighted"]["n_hat"]}
                                    for v in np.round(np.arange(0.01, 0.2001, 0.01), 2)]},
         "lab": lab_results(), "versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("subspace-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote subspace-results.json"); sys.exit()
    M = R["lecture"]["methanol"]
    print("1 methanol A rank", M["A"]["rank"], "null", M["A"]["dim_null"], "sv", r6(M["A"]["sv"]))
    print("1 methanol exact null basis", M["exact"]["basis"], "max|AN|", M["AN_max"])
    print("1 methanol N rank", M["N"]["rank"], "null", M["N"]["dim_null"], "left null", M["N"]["dim_left_null"],
          "integer null(N)", M["null_N_integer"], "row(A) in leftnull(N)", r6(M["rowA_in_leftnullN"]))
    Q = R["lecture"]["meoh_rec"]
    for k in ("unweighted", "weighted"):
        print("1 methanol", k, "n_hat", r6(Q[k]["n_hat"]), "xi", r6(Q[k]["xi"]), "imb", r6(Q[k]["imbalance_before"]), "chi2", r6(Q[k]["chi2"]))
    D = R["lecture"]["derivative"]; print("2 d/dx on P3 rank", D["rank"], "null", D["dim_null"], "left null", D["dim_left_null"])
    print("2 Legendre", [r6(c) for c in R["lecture"]["legendre"]])
    print("2 Hilbert cond", [(h["n"], f"{h['cond']:.3e}") for h in R["lecture"]["hilbert"]])
    L = R["lab"]
    print("3 SMR A rank", L["A"]["rank"], "null", L["A"]["dim_null"], "exact basis", L["exact"]["basis"])
    print("3 SMR N rank", L["N"]["rank"], "R3-R1-R2", L["R3_minus_R1_R2"], "max|AN|", L["AN_max"])
    print("3 true outlet", r6(L["n_true"]), "b", r6(L["b"]))
    print("3 raw extents", r6(L["xi_raw"]), "residual", r6(L["raw_residual"]), "X_CH4 raw", r6(L["CH4_conversion_raw"]))
    for k, q in L["rec"].items():
        print("3", k, "n_hat", r6(q["n_hat"]), "imb before", r6(q["imbalance_before"]), "after", r6(q["imbalance_after"]))
        print("3", k, "xi", r6(q["xi"]), "resid", r6(q["xi_residual"]), "X_CH4", r6(q["CH4_conversion"]),
              "H2 yield", r6(q["H2_yield"]), "chi2", r6(q["chi2"]), "crit", r6(q["chi2_crit95"]), "orth", q["orth_check"])
    E = L["extended"]; print("3 extended rank", E["rank"], "null", E["dim_null"], "basis", E["basis"])
