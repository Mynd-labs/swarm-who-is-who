"""Who is who in a swarm? Simulation of four identity schemes for attributing agent actions.
Pure numpy. Run: python3 whoiswho_sim.py  (writes results.json). Written Nov 9 2026 for the
Apart Research x AIAF Swarm Alignment sprint. No pre-existing code was reused."""
import json, sys
import numpy as np

DEF = dict(N=100, K=10, D=8, T=40, E=3, rho=0.10, imp=0.10, imp_p=0.3, clone=0.10,
           sigma=1.0, kappa=0.02, enrol=5)
SCHEMES = ["Name", "Operator", "Key", "Fingerprint", "Key+Fingerprint"]

def run(seed, **over):
    p = dict(DEF); p.update(over)
    N, K, D, T, E = p["N"], p["K"], p["D"], p["T"], p["E"]
    r = np.random.default_rng(seed)
    op = np.repeat(np.arange(K), N // K)
    r.shuffle(op)
    base = r.normal(0, 1.0, (K, D))
    style = base[op] + r.normal(0, 0.7, (N, D))
    key0 = np.arange(N)
    n_clone = int(round(p["clone"] * N))
    ids = r.permutation(N)
    clones, parents = ids[:n_clone], ids[n_clone:]
    for c in clones:                      # a clone copies a parent's weights and signing key
        par = r.choice(parents)
        style[c] = style[par] + r.normal(0, 0.05, D)
        key0[c] = key0[par]
    imp_set = set(r.choice(N, int(round(p["imp"] * N)), replace=False).tolist())
    # enrolment: centroids from clean samples (registry holds key0 and these centroids)
    cent = np.array([(style[a] + r.normal(0, p["sigma"], (p["enrol"], D))).mean(0) for a in range(N)])
    name = np.arange(N); nxt_name = N
    key = key0.copy(); nxt_key = N
    A = []; NM = []; KY = []; OPS = []; X = []; VIC = []
    for t in range(T):
        for a in range(N):
            if r.random() < p["rho"]:
                name[a] = nxt_name; nxt_name += 1
            if r.random() < p["kappa"]:
                key[a] = nxt_key; nxt_key += 1
        for a in range(N):
            for _ in range(E):
                claim, vic = name[a], -1
                if a in imp_set and r.random() < p["imp_p"]:
                    vic = int(r.integers(N))
                    if vic == a: vic = (vic + 1) % N
                    claim = name[vic]
                A.append(a); NM.append(claim); KY.append(key[a]); OPS.append(op[a]); VIC.append(vic)
                X.append(style[a] + r.normal(0, p["sigma"], D))
    A = np.array(A); NM = np.array(NM); KY = np.array(KY); OPS = np.array(OPS)
    VIC = np.array(VIC); X = np.array(X)
    M = len(A)
    # fingerprint: nearest enrolled centroid over all agents
    d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    fp = d.argmin(1)
    # key + fingerprint: known enrolled key -> candidates are agents enrolled with it; else all agents
    owners = {}
    for a in range(N): owners.setdefault(int(key0[a]), []).append(a)
    kf = np.empty(M, int)
    for i in range(M):
        c = owners.get(int(KY[i]))
        if c is None: kf[i] = fp[i]
        else:
            c = np.array(c); kf[i] = c[d[i, c].argmin()]
    out = {}
    def registry_eval(lab, registered):
        """Fair registry lookup: an identity label is attributed to the agent(s) that enrolled it.
        Labels never enrolled (renamed names, rotated keys) are unattributable -> -1 (counted wrong).
        If several agents enrolled the same label (shared key, shared operator) pick one uniformly."""
        pred = np.empty(M, int)
        for i in range(M):
            c = registered.get(int(lab[i]))
            pred[i] = -1 if c is None else c[r.integers(len(c))]
        return pred
    reg_name = {a: [a] for a in range(N)}
    reg_op = {}
    for a in range(N): reg_op.setdefault(int(op[a]), []).append(a)
    reg_key = owners
    preds = {"Name": registry_eval(NM, reg_name), "Operator": registry_eval(OPS, reg_op),
             "Key": registry_eval(KY, reg_key), "Fingerprint": fp, "Key+Fingerprint": kf}
    labs = {"Name": NM, "Operator": OPS, "Key": KY, "Fingerprint": fp, "Key+Fingerprint": kf}
    for s in SCHEMES:
        pr = preds[s]
        acc = float((pr == A).mean())
        # continuity: per agent, share of its events that land in its single largest cluster
        cont = []
        for a in range(N):
            m = A == a
            cont.append(np.bincount(labs[s][m] - labs[s][m].min()).max() / m.sum())
        im = VIC >= 0
        framed = float((pr[im] == VIC[im]).mean()) if im.any() else float("nan")
        out[s] = dict(acc=acc, cont=float(np.mean(cont)),
                      frag=float(len(np.unique(labs[s])) / N), framed=framed)
    return out

def sweep(param, values, seeds=30, **fixed):
    res = {}
    for v in values:
        rows = [run(1000 + s, **{**fixed, param: v}) for s in range(seeds)]
        res[str(v)] = {s: {m: [row[s][m] for row in rows] for m in ["acc", "cont", "frag", "framed"]}
                       for s in SCHEMES}
        print(param, v, {s: round(np.mean(res[str(v)][s]["acc"]), 3) for s in SCHEMES}, flush=True)
    return res

if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    R = {"defaults": DEF, "seeds": seeds}
    R["default_run"] = sweep("rho", [DEF["rho"]], seeds)
    R["rename"] = sweep("rho", [0.0, 0.02, 0.05, 0.1, 0.2, 0.4], seeds)
    R["impersonation"] = sweep("imp", [0.0, 0.05, 0.1, 0.2, 0.4], seeds)
    R["clones"] = sweep("clone", [0.0, 0.05, 0.1, 0.2, 0.3], seeds)
    R["noise"] = sweep("sigma", [0.25, 0.5, 1.0, 2.0], seeds)
    R["rotation"] = sweep("kappa", [0.0, 0.02, 0.05, 0.1], seeds)
    json.dump(R, open("results.json", "w"))
