"""Model 2 - machine-learning investigation (notebook 05).

Feature table of the team's feature notebook, Isolation Forest, K-Means and the
review of ML candidates against PLC evidence.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, average_precision_score, silhouette_score
from sklearn.preprocessing import StandardScaler




FEATURES_A = ["log_duration_ratio", "robust_z_clipped", "time_since_cycle_start_s", "idle_before_s",
              "n_parallel_active", "rack_x", "rack_y", "is_siemens"]


FEATURES_B = ["time_since_cycle_start_s", "idle_before_s", "n_parallel_active", "rack_x", "rack_y",
              "is_siemens"]


def activation_features(acts: pd.DataFrame, cycles: pd.DataFrame, p0: float) -> pd.DataFrame:
    """Feature table of notebooks/03_feature_dataset.ipynb, extracted unchanged.

    One row per activation (all of them, including start-up and runs open at the
    recording borders). Definitions as in that notebook:
    * cycle by the activation's midpoint; cycle 1 starts at t = 0 (start-up);
    * median / MAD from complete runs in production; log ratio and robust z only
      for complete runs of outputs with >= 3 such runs (else NaN; robust z NaN if MAD = 0);
    * idle time since the previous run of the same output (NaN for the first);
    * other outputs ON at the midpoint.
    Notebook 05 checks that this equals the table built by the team's feature-notebook code.
    """
    a = acts[["plc", "signal", "start", "end", "open_at_start", "open_at_end"]].copy()
    a["duration_s"] = a["end"] - a["start"]
    a = a.sort_values(["start", "plc", "signal"]).reset_index(drop=True)
    starts = cycles["start_s"].to_numpy(dtype=float).copy()
    starts[0] = 0.0
    ends = np.append(starts[1:], cycles["end_s"].iloc[-1])
    a["midpoint"] = (a["start"] + a["end"]) / 2
    a["cycle"] = np.searchsorted(starts, a["midpoint"].to_numpy(), side="right")
    c = cycles.assign(cycle_start=starts, cycle_end=ends).set_index("cycle")
    a["crosses_cycle_boundary"] = ((a["start"] < a["cycle"].map(c["cycle_start"]) - 1e-9)
                                   | (a["end"] > a["cycle"].map(c["cycle_end"]) + 1e-9))
    a["rack_x"] = a["cycle"].map(c["rack_x"])
    a["rack_y"] = a["cycle"].map(c["rack_y"])
    a["time_since_cycle_start_s"] = a["start"] - a["cycle"].map(c["cycle_start"])
    a["in_production_window"] = a["start"] >= p0
    ref = a[a["in_production_window"] & ~a["open_at_start"] & ~a["open_at_end"]]
    stats = (ref.groupby(["plc", "signal"])["duration_s"]
             .agg(n_signal_acts="size", median_duration_s="median",
                  mad_duration_s=lambda x: (x - x.median()).abs().median()).reset_index())
    a = a.merge(stats, on=["plc", "signal"], how="left", validate="many_to_one")
    a["n_signal_acts"] = a["n_signal_acts"].fillna(0).astype(int)
    enough = (a["n_signal_acts"] >= 3) & ~a["open_at_start"] & ~a["open_at_end"]
    scale = (1.4826 * a["mad_duration_s"]).where(lambda x: x > 0)
    a["log_duration_ratio"] = np.where(enough, np.log(a["duration_s"].clip(lower=1e-3) / a["median_duration_s"]),
                                       np.nan)
    a["robust_z"] = np.where(enough, (a["duration_s"] - a["median_duration_s"]) / scale, np.nan)
    a = a.sort_values(["plc", "signal", "start"])
    a["idle_before_s"] = a["start"] - a.groupby(["plc", "signal"])["end"].shift()
    a = a.sort_values(["start", "plc", "signal"]).reset_index(drop=True)
    s_, e_, m_ = a["start"].to_numpy(), a["end"].to_numpy(), a["midpoint"].to_numpy()
    a["n_parallel_active"] = ((s_[None, :] <= m_[:, None]) & (m_[:, None] < e_[None, :])).sum(axis=1) - 1
    return a.drop(columns="midpoint")


def build_features(acts: pd.DataFrame, cycles: pd.DataFrame, flag_keys: set, p0: float,
                   min_n: int = 5) -> pd.DataFrame:
    """Model-2 input: the team's feature table (notebook 05, part 1), modelling-stage choices applied.

    Rows: complete runs in production of outputs with >= ``min_n`` runs (same
    population as the rule). Missing values, which the feature notebook deliberately leaves
    to the modelling stage: robust z with MAD = 0 -> 0 and clipped to +/-20; idle
    time of the first run -> median idle time of that output.
    """
    a = activation_features(acts, cycles, p0)
    a = a[a["in_production_window"] & ~a["open_at_start"] & ~a["open_at_end"] & (a["n_signal_acts"] >= min_n)].copy()
    a["duration"] = a["duration_s"]
    a["fence_s"] = a.groupby("signal")["duration_s"].transform(
        lambda x: x.quantile(0.75) + 3.0 * (x.quantile(0.75) - x.quantile(0.25)))
    a["robust_z_clipped"] = a["robust_z"].clip(-20, 20).fillna(0.0)
    a["idle_before_s"] = a["idle_before_s"].fillna(a.groupby("signal")["idle_before_s"].transform("median")).fillna(0)
    a["is_siemens"] = (a["plc"] == "Siemens").astype(int)
    a["rule_flag"] = [k in flag_keys for k in zip(a["signal"], a["start"])]
    # stop-cut flags of notebook 03 (needed to explain why a run is unusual, not model input)
    a = a.merge(acts[["signal", "start", "station", "cut_by_stop", "resumed_after_stop"]], on=["signal", "start"],
                how="left", validate="one_to_one")
    return a.sort_values("start").reset_index(drop=True)


def _if_scores(X: np.ndarray, seed: int, contamination="auto") -> tuple[np.ndarray, np.ndarray]:
    m = IsolationForest(n_estimators=300, random_state=seed, contamination=contamination).fit(X)
    return -m.score_samples(X), m.predict(X) == -1


def isolation_forest_check(feat: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seeds, top_k = cfg["ml"]["seeds"], cfg["ml"]["top_k"]
    y = feat["rule_flag"].to_numpy()
    rows, sweep, ranks = [], [], {}
    for name, cols in (("A_duration_aware", FEATURES_A), ("B_context_only", FEATURES_B)):
        X = StandardScaler().fit_transform(feat[cols].to_numpy(float))
        tops, scores_all = [], []
        for sd in seeds:
            sc, _ = _if_scores(X, sd)
            scores_all.append(sc)
            tops.append(set(np.argsort(-sc)[:top_k]))
            rows.append({"feature_set": name, "seed": sd, "average_precision_vs_rule": average_precision_score(y, sc),
                         f"rule_flags_in_top{top_k}": int(y[list(tops[-1])].sum())})
        jac = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
        mean_score = np.mean(scores_all, axis=0)
        ranks[name] = mean_score
        for c in cfg["ml"]["contamination_sweep"]:
            _, flagged = _if_scores(X, seeds[0], c)
            sweep.append({"feature_set": name, "contamination": c, "flagged": int(flagged.sum()),
                          "rule_flags_covered": int((flagged & y).sum()), "rule_flags_total": int(y.sum()),
                          "flagged_not_rule": int((flagged & ~y).sum())})
        rows.append({"feature_set": name, "seed": "all", "top_k_jaccard_mean": float(np.mean(jac)),
                     "top_k_jaccard_min": float(np.min(jac))})
    for name, sc in ranks.items():
        feat[f"if_score_{name}"] = sc
        feat[f"if_rank_{name}"] = (-sc).argsort().argsort() + 1
    res = pd.DataFrame(rows)
    summ = {}
    for name in ranks:
        r = res[(res["feature_set"] == name) & (res["seed"] != "all")]
        s = res[(res["feature_set"] == name) & (res["seed"] == "all")].iloc[0]
        summ[name] = {"average_precision_mean": float(r["average_precision_vs_rule"].mean()),
                      "average_precision_min": float(r["average_precision_vs_rule"].min()),
                      "rule_flags_in_top_k_mean": float(r[f"rule_flags_in_top{top_k}"].mean()),
                      "top_k_jaccard_mean": float(s["top_k_jaccard_mean"]),
                      "top_k_jaccard_min": float(s["top_k_jaccard_min"])}
    summ["base_rate"] = float(y.mean())
    return res, pd.DataFrame(sweep), summ


def kmeans_check(feat: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    X = StandardScaler().fit_transform(feat[FEATURES_A].to_numpy(float))
    y = feat["rule_flag"].to_numpy()
    rows = []
    for k in cfg["ml"]["kmeans_k"]:
        labels = [KMeans(n_clusters=k, n_init=10, random_state=sd).fit_predict(X) for sd in cfg["ml"]["seeds"]]
        sil = [silhouette_score(X, lab) for lab in labels]
        ari = [adjusted_rand_score(labels[0], lab) for lab in labels[1:]]
        lab = labels[0]
        best = max(range(k), key=lambda c: (y[lab == c].sum(), -np.sum(lab == c)))
        rows.append({"k": k, "silhouette_mean": float(np.mean(sil)), "silhouette_min": float(np.min(sil)),
                     "ari_vs_seed0_min": float(np.min(ari)),
                     "best_cluster_size": int((lab == best).sum()),
                     "rule_flags_in_best_cluster": int(y[lab == best].sum()),
                     "best_cluster_purity": float(y[lab == best].mean())})
    out = pd.DataFrame(rows)
    b = out.loc[out["silhouette_mean"].idxmax()]
    return out, {"best_k_by_silhouette": int(b["k"]), "best_silhouette": float(b["silhouette_mean"]),
                 "best_k_ari_min": float(b["ari_vs_seed0_min"]),
                 "best_k_flag_cluster_purity": float(b["best_cluster_purity"]),
                 "best_k_flags_in_cluster": int(b["rule_flags_in_best_cluster"])}


def kmeans_long_cluster(feat: pd.DataFrame, cfg: dict, k: int) -> pd.Series:
    """Membership of the cluster that holds most rule overruns (k, first seed).

    Used only to report agreement per event; not a detection result.
    """
    X = StandardScaler().fit_transform(feat[FEATURES_A].to_numpy(float))
    lab = KMeans(n_clusters=k, n_init=10, random_state=cfg["ml"]["seeds"][0]).fit_predict(X)
    y = feat["rule_flag"].to_numpy()
    best = max(range(k), key=lambda c: (y[lab == c].sum(), -np.sum(lab == c)))
    return pd.Series(lab == best, index=feat.index)


UNEXPLAINED = "UNEXPLAINED: longer than normal, not flagged - review trace"


def explanation(r, min_excess_s: float) -> str:
    """Why an activation looks unusual - and whether that could be a new loss.

    A performance loss needs lost time, so an activation can only be a new loss
    candidate if it lasted longer than its normal range (above the Tukey fence),
    by at least the detection limit, and is not explained by a stop window or by
    the rule itself.
    """
    if r["rule_flag"]:
        return "rule overrun (also flagged by the engineering rule)"
    if r.get("cut_by_stop", False):
        return "truncated by a stop window (short by construction, no lost time)"
    if r.get("resumed_after_stop", False):
        return "restart directly after a stop window (part of that stop)"
    if r["duration"] <= r["fence_s"]:
        return ("within the normal range of this output (below its Tukey fence); ranked for its context "
                "(cycle position, parallel activity)")
    if r["duration"] - r["median_duration_s"] < min_excess_s:
        return "mild overrun below the detection limit (listed in sheet slow_steps)"
    return UNEXPLAINED


def explain_rows(feat: pd.DataFrame, rank_col: str, n: int, min_excess_s: float) -> pd.DataFrame:
    """Top-ranked activations with the reason they rank high."""
    top = feat.sort_values(rank_col).head(n).copy()
    top["explanation"] = top.apply(explanation, axis=1, min_excess_s=min_excess_s)
    return top[["start", "end", "plc", "signal", "duration", "median_duration_s", "fence_s",
                "log_duration_ratio", "rule_flag", rank_col, "explanation"]]
