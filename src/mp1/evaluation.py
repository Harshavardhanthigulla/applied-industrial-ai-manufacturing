"""Cross-checking and evaluation (notebook 06).

Cycle-level effect (serial chain, position-adjusted reference), performance,
time-grid sensitivity, evidence register and candidate register.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd

import dataclasses

from .data import STOP_TOL_S, activations, mark_stop_cuts, overlaps, pallet_cycles, stop_windows
from .model1 import detect
from .model2 import UNEXPLAINED



# --------------------------------------------------------------------------
# Pallet cycles, event effect and performance estimate
# --------------------------------------------------------------------------
def exact_permutation_p(durations: np.ndarray, is_event: np.ndarray) -> tuple[float, float, int]:
    """One-sided exact permutation test: mean(event) - mean(clean)."""
    obs = durations[is_event].mean() - durations[~is_event].mean()
    k = int(is_event.sum())
    diffs = []
    for chosen in combinations(range(len(durations)), k):
        m = np.zeros(len(durations), bool)
        m[list(chosen)] = True
        diffs.append(durations[m].mean() - durations[~m].mean())
    diffs = np.array(diffs)
    return float(np.mean(diffs >= obs - 1e-9)), float(obs), len(diffs)


def cycle_effect(cycles: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    c = cycles.copy()
    c["events"] = [" ".join(events.loc[events["cycle"] == n, "event"]) for n in c["cycle"]]
    done = c[c["complete"]].copy()
    clean = done[done["events"] == ""]
    hit = done[done["events"] != ""]
    base = float(clean["duration_s"].median())
    done["excess_vs_clean_median_s"] = done["duration_s"] - base
    p, obs, n = exact_permutation_p(done["duration_s"].to_numpy(), (done["events"] != "").to_numpy())
    info = {"clean_cycles": clean["cycle"].tolist(), "event_cycles": hit["cycle"].tolist(),
            "clean_min_s": float(clean["duration_s"].min()), "clean_max_s": float(clean["duration_s"].max()),
            "clean_median_s": base, "event_cycle_durations_s": hit["duration_s"].round(2).tolist(),
            "mean_difference_s": obs, "permutation_p": p, "permutation_splits": n,
            "event_cycles_extra_s": float((hit["duration_s"] - base).sum()),
            "every_event_cycle_longer_than_every_clean": bool(hit["duration_s"].min() > clean["duration_s"].max())}
    return done, info


def _clipped_on_time(acts: pd.DataFrame, signals: list[str], a: float, e: float) -> float:
    x = acts[acts["signal"].isin(signals)]
    return float((x["end"].clip(upper=e) - x["start"].clip(lower=a)).clip(lower=0).sum())


def _union_length(iv: list[tuple[float, float]]) -> float:
    total, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return total + (cur[1] - cur[0] if cur else 0.0)


VERTICAL = ["WH.Q5_vertical_downward", "WH.Q6_vertical_upward"]


def chain_coverage(done: pd.DataFrame, acts: pd.DataFrame) -> pd.DataFrame:
    """Share of each pallet cycle in which a warehouse or robot output is ON.

    If the Beckhoff warehouse + robot outputs cover the whole cycle, the pallet
    cycle is one serial chain (no idle time on it) and the Siemens stations
    work in parallel, off that chain.
    """
    b = acts[acts["plc"] == "Beckhoff"]
    rows = []
    for c in done.itertuples():
        iv = [(max(s, c.start_s), min(e, c.end_s)) for s, e in zip(b["start"], b["end"])
              if e > c.start_s and s < c.end_s]
        busy = _union_length(iv)
        rows.append({"cycle": c.cycle, "duration_s": c.duration_s, "chain_busy_s": busy,
                     "chain_idle_s": c.duration_s - busy})
    return pd.DataFrame(rows)


def chain_handover(done: pd.DataFrame, acts: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    """Hand-over delays on the serial chain, per cycle (measured).

    * robot pick start after belt-forward start (does the robot wait for the belt?)
    * belt-backward end -> next warehouse output (does the warehouse wait for the belt?)
    These show which belt overruns lay on the chain and which ran in parallel.
    """
    rows = []
    for c in done.itertuples():
        win = acts[(acts["start"] >= c.start_s) & (acts["start"] < c.end_s)]
        bf = win[win["signal"] == "WH.Q1_beltForward"].sort_values("start").iloc[0]
        bb = win[win["signal"] == "WH.Q2_beltBackward"].sort_values("start").iloc[0]
        m = moves[(moves["cycle"] == c.cycle) & (moves["move_type"] == "pallet -> oven")].iloc[0]
        nxt = acts[acts["signal"].isin(VERTICAL) & (acts["start"] >= bb["start"])].sort_values("start").iloc[0]
        rows.append({"cycle": c.cycle, "events": c.events,
                     "belt_forward_s": bf["duration"], "robot_pick_after_belt_start_s": m["start_s"] - bf["start"],
                     "belt_backward_s": bb["duration"], "warehouse_after_belt_backward_end_s": nxt["start"] - bb["end"]})
    return pd.DataFrame(rows)


def event_chain_effect(events: pd.DataFrame, handover: pd.DataFrame, tol: float = STOP_TOL_S) -> list[str]:
    """Did the event lengthen the serial warehouse + robot chain? (measured rule)

    * stop window: no actuator switches -> the whole line, including the chain, halts;
    * belt forward overrun: the chain is delayed only if the robot pick started
      later than in every clean cycle;
    * belt backward overrun: the warehouse starts its next step at the belt's
      end, so the chain waited for it;
    * Siemens step without a stop: off the chain; its effect is not separable.
    """
    clean = handover[handover["events"] == ""]
    pick_max = float(clean["robot_pick_after_belt_start_s"].max())
    wait_max = float(clean["warehouse_after_belt_backward_end_s"].max())
    h = handover.set_index("cycle")
    out = []
    for e in events.itertuples():
        sig = e.signals.split()
        if isinstance(e.stop_id, str) and e.stop_id:
            out.append("line halted (stop window)")
        elif e.cycle not in h.index:
            out.append("not measurable (cycle not complete)")
        elif "WH.Q1_beltForward" in sig:
            late = h.loc[e.cycle, "robot_pick_after_belt_start_s"] - pick_max
            out.append(f"chain delayed: robot pick {late:.1f} s later than in any clean cycle" if late > tol
                       else "ran in parallel with the robot pick (no chain delay)")
        elif "WH.Q2_beltBackward" in sig:
            out.append("chain delayed: warehouse waited for the belt" if
                       h.loc[e.cycle, "warehouse_after_belt_backward_end_s"] <= wait_max + tol
                       else "no chain delay")
        else:
            out.append("off the warehouse-robot chain (effect not separable)")
    return out


def cycle_reference(done: pd.DataFrame, travel: pd.DataFrame, moves: pd.DataFrame, acts: pd.DataFrame,
                    events: pd.DataFrame, stops: pd.DataFrame, flag_keys: set) -> tuple[pd.DataFrame, dict]:
    """Position-adjusted normal reference time for every complete pallet cycle.

    The pallet cycle is one serial warehouse + robot chain (`chain_coverage`).
    Its parts that depend on where the pallet is stored are taken from
    measured models instead of one fixed reference:

    * horizontal travel: constant-speed model of `warehouse_travel` (predicted
      from the travel distances, so an overrun or a stop does not enter it),
    * robot moves: median duration of the same move type and target lane,
    * vertical travel: measured motor ON time (no physical model exists; used
      only because no vertical run is flagged or cut by a stop - checked),
    * everything else (cantilever, belt, robot positioning, hand-overs) is the
      fixed part F, estimated as the median over the clean cycles.

    reference = F + horizontal(pred) + robot(ref) + vertical(meas).
    This is a *normal* reference (what the line demonstrably needs at these
    rack positions and lanes), not an ideal (design) cycle time.
    """
    rmed = moves.groupby(["move_type", "lane"])["duration_s"].median()
    vert = acts[acts["signal"].isin(VERTICAL)]
    vert_bad = vert[[k in flag_keys or cut for k, cut in zip(zip(vert["signal"], vert["start"]), vert["cut_by_stop"])]]
    rows = []
    for c in done.itertuples():
        t = travel[travel["cycle"] == c.cycle]
        m = moves[moves["cycle"] == c.cycle]
        lane = ", ".join(x for x in m["lane"].fillna("") if x)
        st = stops.assign(a=stops["start_s"].clip(lower=c.start_s), b=stops["end_s"].clip(upper=c.end_s))
        rows.append({
            "cycle": c.cycle, "rack_x": c.rack_x, "rack_y": c.rack_y, "lane": lane, "events": c.events,
            "observed_s": c.duration_s,
            "horizontal_pred_s": float(t["predicted_s"].sum()), "horizontal_obs_s": float(t["duration_s"].sum()),
            "robot_ref_s": float(sum(rmed[(r.move_type, r.lane if isinstance(r.lane, str) else "")]
                                     for r in m.itertuples())),
            "robot_obs_s": float(m["duration_s"].sum()),
            "vertical_obs_s": _clipped_on_time(acts, VERTICAL, c.start_s, c.end_s),
            "vertical_runs_flagged_or_cut": int(((vert_bad["start"] < c.end_s) & (vert_bad["end"] > c.start_s)).sum()),
            "stop_in_cycle_s": float((st["b"] - st["a"]).clip(lower=0).sum()),
            "equipment_lost_in_cycle_s": float(events.loc[events["cycle"] == c.cycle, "equipment_lost_time_s"].sum()),
        })
    out = pd.DataFrame(rows)
    out["position_dependent_s"] = out["horizontal_pred_s"] + out["robot_ref_s"] + out["vertical_obs_s"]
    out["fixed_part_s"] = out["observed_s"] - out["position_dependent_s"]
    clean = out["events"] == ""
    f_clean = out.loc[clean, "fixed_part_s"]
    F = float(f_clean.median())
    out["reference_s"] = F + out["position_dependent_s"]
    out["reference_low_s"] = float(f_clean.min()) + out["position_dependent_s"]
    out["reference_high_s"] = float(f_clean.max()) + out["position_dependent_s"]
    out["cycle_excess_s"] = out["observed_s"] - out["reference_s"]
    # leave-one-out check on the clean cycles: predict each from the others
    out["loo_residual_s"] = np.nan
    for i in out.index[clean]:
        f_other = float(f_clean.drop(i).median())
        out.loc[i, "loo_residual_s"] = out.loc[i, "observed_s"] - (f_other + out.loc[i, "position_dependent_s"])
    single = float(out.loc[clean, "observed_s"].median())
    out["excess_vs_clean_median_s"] = out["observed_s"] - single
    hit = out[~clean]
    info = {
        "fixed_part_s": F, "fixed_part_clean_min_s": float(f_clean.min()), "fixed_part_clean_max_s": float(f_clean.max()),
        "clean_observed_range_s": float(np.ptp(out.loc[clean, "observed_s"])),
        "clean_fixed_part_range_s": float(np.ptp(f_clean)),
        "clean_loo_max_abs_residual_s": float(out["loo_residual_s"].abs().max()),
        "vertical_runs_flagged_or_cut": int(out["vertical_runs_flagged_or_cut"].sum()),
        "event_cycles_excess_s": float(hit["cycle_excess_s"].sum()),
        "event_cycles_excess_low_s": float((hit["observed_s"] - hit["reference_high_s"]).sum()),
        "event_cycles_excess_high_s": float((hit["observed_s"] - hit["reference_low_s"]).sum()),
        "event_cycles_equipment_lost_s": float(hit["equipment_lost_in_cycle_s"].sum()),
        "event_cycles_stop_s": float(hit["stop_in_cycle_s"].sum()),
        "single_reference_s": single,
    }
    return out, info


def performance_estimates(done: pd.DataFrame, info: dict, n_boot: int, seed: int,
                          ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Performance of the complete cycles under several reference times.

    Performance = sum of reference times / operating time; with one reference
    for all cycles this is parts x reference / operating time. Operating time =
    span of the complete cycles (all stops inside them are treated as minor
    stops, i.e. performance losses, see docs/methodology.md).

    * method 'single reference' (kept for comparison): one cycle time for all
      cycles - fastest clean cycle, median clean cycle and its bootstrap.
    * method 'position-adjusted' (`cycle_reference`): every cycle against the
      normal time at its own rack position and lane; range from the smallest
      and largest fixed part of the clean cycles.
    Neither is an ideal (design) cycle time; see docs/methodology.md.
    """
    parts = len(done)
    op = float(done["duration_s"].sum())
    clean = done.loc[done["events"] == "", "duration_s"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(clean, size=(n_boot, len(clean)), replace=True), axis=1)
    rows = [
        ("single reference", "fastest clean cycle (upper bound of a single ideal time)", parts * clean.min()),
        ("single reference", "median clean cycle (typical, not ideal)", parts * np.median(clean)),
        ("single reference", "bootstrap 2.5% of median clean cycle", parts * np.percentile(boot, 2.5)),
        ("single reference", "bootstrap 97.5% of median clean cycle", parts * np.percentile(boot, 97.5)),
    ]
    if ref is not None:
        rows += [
            ("position-adjusted", "normal reference, median fixed part (primary)", ref["reference_s"].sum()),
            ("position-adjusted", "normal reference, smallest clean fixed part", ref["reference_low_s"].sum()),
            ("position-adjusted", "normal reference, largest clean fixed part", ref["reference_high_s"].sum()),
        ]
    out = pd.DataFrame(rows, columns=["method", "reference_basis", "reference_time_total_s"])
    out["mean_reference_cycle_s"] = out["reference_time_total_s"] / parts
    out["parts"] = parts
    out["operating_time_s"] = op
    out["performance"] = out["reference_time_total_s"] / op
    out["lost_time_s"] = op - out["reference_time_total_s"]
    return out


# --------------------------------------------------------------------------
# Notebook 05: evidence register and candidate register
# --------------------------------------------------------------------------
STOP_SIGNALS = {"protective stop (light curtain)": "A201.0, M0.3, M24.3, VG.safeCurtainShort, E200.0 (reset)",
                "HMI stop": "Visual.start, Visual.Stoplight"}


def evidence_register(events: pd.DataFrame, attr: pd.DataFrame, flags: pd.DataFrame, feat: pd.DataFrame,
                      quiet: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """One row per final event: what detected it, what independently confirms it,
    which other methods agree, how sensitive it is, and the final decision.

    Independent evidence = a signal that did NOT define the detection (stop
    states, light barriers, encoder, motor-vs-request). Agreement of Model 2 or
    of another rule is listed separately and is not counted as confirmation.
    """
    at = attr.set_index("event")
    rows = []
    for e in events.itertuples():
        f = flags[flags["event"] == e.event]
        keys = set(zip(f["signal"], f["start"]))
        mine = feat[[k in keys for k in zip(feat["signal"], feat["start"])]]
        rank_a = int(mine["if_rank_A_duration_aware"].min()) if len(mine) else None
        rank_b = int(mine["if_rank_B_context_only"].min()) if len(mine) else None
        in_km = bool(mine["kmeans_long_cluster"].any()) if len(mine) else False
        q = quiet[[overlaps(a, b, e.start_s, e.end_s) for a, b in zip(quiet["start_s"], quiet["end_s"])]]
        ok_grids = [int(r.grid_ms) for r in grid.itertuples()
                    if r.status == "ok" and e.event in str(r.raw_events_found).split()]
        indep = [x for x in (e.standstill_basis if isinstance(e.standstill_basis, str) else "",
                             f"expected: {at.loc[e.event, 'expected_transition']}; "
                             f"observed: {at.loc[e.event, 'observed']}") if x and "nan" not in x]
        supported = e.worst_case_lost_time_s > 0 and bool(indep)
        decision = ("evidence-supported performance loss" if supported and e.sweep_presence >= 0.9 else
                    "evidence-supported performance loss, threshold-sensitive" if supported else "uncertain")
        rows.append({
            "event": e.event, "start_s": e.start_s, "end_s": e.end_s,
            "equipment_lost_time_s": e.equipment_lost_time_s, "timing_uncertainty_s": e.boundary_uncertainty_s,
            "equipment": e.component, "location_confidence": e.location_confidence,
            "detection_method": f"Model 1 step-overrun rule on {e.signals}",
            "supporting_signals": e.signals + (f"; stop: {STOP_SIGNALS.get(e.stop_mechanism, '')}"
                                               if isinstance(e.stop_id, str) and e.stop_id else ""),
            "independent_evidence": " | ".join(indep),
            "model2_isolation_forest": f"rank A {rank_a}, rank B {rank_b} of {len(feat)}",
            "model2_kmeans_long_run_cluster": in_km,
            "line_quiet_check": f"{len(q)} long quiet period(s) inside" if len(q) else "no long quiet period",
            "serial_chain_effect": e.serial_chain_effect,
            "threshold_sweep_presence": e.sweep_presence,
            "found_on_grids_up_to_ms": max(ok_grids) if ok_grids else None,
            "existence_confidence": e.existence_confidence,
            "remaining_uncertainty": ("stop vs. slow motion not separable (no motion sensor); "
                                      if e.loss_category.startswith("minor stop or") else "")
                                     + str(at.loc[e.event, "root_cause"]),
            "final_decision": decision,
        })
    return pd.DataFrame(rows)


def candidate_register(flags: pd.DataFrame, mild: pd.DataFrame, ml_top: pd.DataFrame, quiet: pd.DataFrame,
                       limit_s: float, unexplained_label: str, km_members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every candidate the methods produced and what became of it.

    status: evidence-supported loss | abnormal signal behaviour, not a loss |
            anomaly, not a loss | uncertain | explained
    """
    rows = [{"source": "Model 1 rule (flagged activation)", "signal": r.signal, "start_s": r.start,
             "end_s": r.end, "excess_s": r.excess, "status": "evidence-supported loss",
             "reason": f"part of event {r.event}; excess >= {limit_s:.2f} s, positive after timing uncertainty"}
            for r in flags.itertuples()]
    rows += [{"source": "Model 1 rule (above fence, below detection limit)", "signal": r.signal, "start_s": r.start,
              "end_s": r.end, "excess_s": r.excess, "status": "abnormal signal behaviour, not a loss",
              "reason": f"excess {r.excess:.2f} s < detection limit {limit_s:.2f} s"
                        + (" (larger than its own edge uncertainty)" if r.excess_beyond_uncertainty else
                           " (within its own edge uncertainty)")}
             for r in mild.itertuples()]
    for r in ml_top[~ml_top["rule_flag"]].drop_duplicates(["signal", "start"]).itertuples():
        rows.append({"source": f"Model 2 Isolation Forest top 20 ({r.feature_set})", "signal": r.signal,
                     "start_s": r.start, "end_s": r.end, "excess_s": r.duration - r.median_duration_s,
                     "status": "uncertain" if r.explanation == unexplained_label else "anomaly, not a loss",
                     "reason": r.explanation})
    if km_members is not None:
        rows += [{"source": "Model 2 K-Means cluster of long runs (not rule-flagged)", "signal": r.signal,
                  "start_s": r.start, "end_s": r.end, "excess_s": r.duration - r.median_duration_s,
                  "status": "uncertain" if r.explanation == unexplained_label else "anomaly, not a loss",
                  "reason": r.explanation} for r in km_members.itertuples()]
    rows += [{"source": "Line-quiet check (no actuator edge)", "signal": "all actuators", "start_s": r.start_s,
              "end_s": r.end_s, "excess_s": np.nan, "status": "explained", "reason": r.explained_by}
             for r in quiet.itertuples()]
    return pd.DataFrame(rows)


def grid_bundle(data: dict, grid_ms: int) -> dict:
    """Copy of the data bundle with every PLC frame on a regular grid.

    The value at a grid tick is the last logged row at or before that tick
    (forward fill). Nothing is interpolated: boolean states cannot be averaged.
    """
    out = dict(data)
    step = grid_ms / 1000.0
    for plc in ("Siemens", "Beckhoff"):
        d = data[plc]
        f = d.frame
        ticks = np.arange(f["t"].iloc[0], f["t"].iloc[-1] + 1e-9, step)
        pos = np.searchsorted(f["t"].to_numpy(), ticks, side="right") - 1
        g = f.iloc[pos].reset_index(drop=True).copy()
        g["t"] = ticks
        g["timestamp"] = (data["t0_ms"] + np.round(ticks * 1000)).astype("int64")
        out[plc] = dataclasses.replace(d, frame=g)
    return out


def _events(flags: pd.DataFrame) -> pd.DataFrame:
    if flags.empty:
        return pd.DataFrame(columns=["start_s", "end_s", "lost_s", "signals"])
    return (flags.groupby("event")
            .agg(start_s=("start", "min"), end_s=("end", "max"), lost_s=("excess", "max"),
                 signals=("signal", lambda s: " ".join(sorted(s))))
            .reset_index(drop=True))


def run_on(data: dict, cfg: dict, min_excess_s: float, tol: float) -> dict:
    """Accepted preparation + Model-1 detection on a (possibly gridded) bundle."""
    rc = cfg["rule"]
    stops = stop_windows(data, cfg, tol)
    cycles = pallet_cycles(data, cfg)
    acts = mark_stop_cuts(activations(data, cfg, cycles), stops)
    flags, _ = detect(acts, rc["tukey_k"], min_excess_s, rc["min_activations"], rc["merge_gap_s"])
    return {"stops": stops, "cycles": cycles, "acts": acts, "events": _events(flags), "flags": flags}


def event_sensitivity(data: dict, cfg: dict, raw: dict, grids_ms: list[int], min_excess_s: float,
                      tol: float, event_names: list[str]) -> pd.DataFrame:
    """Compare the raw-time result with the result on each regular grid.

    ``raw`` is ``run_on(data, ...)``; ``event_names`` labels its events (E1..).
    Columns: what is lost or shifted by the grid at every level of the analysis
    (rows, stop windows, cycles, flagged activations, events, lost time, equipment).
    """
    rev = raw["events"].reset_index(drop=True).copy()
    rev["event"] = event_names
    rstops, rcyc = raw["stops"], raw["cycles"]
    rows = []
    for g in grids_ms:
        row = {"grid_ms": g}
        try:
            res = run_on(grid_bundle(data, g), cfg, min_excess_s, tol)
        except Exception as exc:                  # a grid can lose a pulse the pipeline needs
            row["status"] = f"failed: {type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        ev = res["events"]
        matched, lost_diff, same_equipment = [], [], []
        for r in rev.itertuples():
            hit = ev[[overlaps(r.start_s, r.end_s, a, b) for a, b in zip(ev["start_s"], ev["end_s"])]]
            if len(hit):
                matched.append(r.event)
                lost_diff.append(abs(hit["lost_s"].max() - r.lost_s))
                same_equipment.append(set(" ".join(hit["signals"]).split()) == set(r.signals.split()))
        extra = sum(not any(overlaps(a, b, x, y) for x, y in zip(rev["start_s"], rev["end_s"]))
                    for a, b in zip(ev["start_s"], ev["end_s"]))
        st = res["stops"]
        stop_shift = (max(abs(st["start_s"].to_numpy() - rstops["start_s"].to_numpy()).max(),
                          abs(st["end_s"].to_numpy() - rstops["end_s"].to_numpy()).max())
                      if len(st) == len(rstops) else np.nan)
        cyc = res["cycles"]
        cyc_shift = (abs(cyc["duration_s"].to_numpy() - rcyc["duration_s"].to_numpy()).max()
                     if len(cyc) == len(rcyc) else np.nan)
        row.update({
            "status": "ok",
            "flagged_activations": int(len(res["flags"])), "events": int(len(ev)),
            "raw_events_found": " ".join(matched),
            "raw_events_missed": " ".join(e for e in rev["event"] if e not in matched),
            "extra_events": int(extra),
            "max_lost_time_change_s": float(max(lost_diff)) if lost_diff else np.nan,
            "same_equipment_flagged": bool(all(same_equipment)) if same_equipment else False,
            "stop_windows": int(len(st)), "max_stop_boundary_shift_s": stop_shift,
            "pallet_cycles": int(len(cyc)), "max_cycle_duration_change_s": cyc_shift,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Quantified results by level (equipment, incident, cycle, performance).
# Every value is derived from tables computed above; nothing is assumed.
# Equipment seconds diagnose WHERE a step took too long; the cycle excess measures the
# additional production time. They are never added to each other or across equipment.
# --------------------------------------------------------------------------
NOT_AVAILABLE = "not available"


def _pct(num, den) -> float:
    """num / den in %, or NaN when the denominator is missing or not positive."""
    return float(100.0 * num / den) if pd.notna(den) and den > 0 else np.nan


def equipment_quantification(flags: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """One row per flagged activation (equipment-level overrun) against the baseline of its own output.

    observed = ON duration; median normal = median ON duration of the same output (complete runs in
    production); excess = observed - median (the rule's lost time). The excess is also given over the longest
    normal run of the same output (stricter baseline that already includes the farthest rack position).
    """
    rows = []
    for e, g in flags.sort_values("start").groupby("event", sort=False):
        for order, r in enumerate(g.sort_values("overrun_start_s").itertuples(), start=1):
            hit = [(s.stop_id, min(r.overrun_end_s, s.end_s) - max(r.overrun_start_s, s.start_s))
                   for s in stops.itertuples()]
            hit = [(sid, ov) for sid, ov in hit if ov > 0]
            rows.append({
                "event": e, "order_in_incident": order, "plc": r.plc, "signal": r.signal, "station": r.station,
                "observed_s": r.duration, "median_normal_s": r.median, "longest_normal_s": r.longest_normal_s,
                "excess_s": r.excess,
                "excess_relative_to_median_normal_duration_pct": _pct(r.excess, r.median),
                "excess_over_longest_normal_s": r.above_longest_normal_s,
                "excess_relative_to_longest_normal_duration_pct": _pct(r.above_longest_normal_s, r.longest_normal_s),
                "worst_case_excess_s": r.worst_case_excess_s,
                "overrun_start_s": r.overrun_start_s, "overrun_end_s": r.overrun_end_s,
                "stop_window_overlapped": " ".join(sid for sid, _ in hit),
                "overrun_inside_stop_s": float(sum(ov for _, ov in hit)),
            })
    return pd.DataFrame(rows)


def incident_quantification(events: pd.DataFrame, equipment: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """One row per event (incident): its equipment overruns in chronological order and how they relate.

    No additive incident value is assigned: an incident with several equipment overruns keeps one value per
    equipment. A stop window's duration is the measured standstill, not excess production time.
    """
    st = stops.set_index("stop_id")
    rows = []
    for ev in events.itertuples():
        g = equipment[equipment["event"] == ev.event].sort_values("overrun_start_s")
        seq = " -> ".join(f"{r.signal} {r.excess_s:.2f} s from {r.overrun_start_s:.2f} s" for r in g.itertuples())
        if len(g) == 1:
            relation = "single equipment overrun"
        else:
            a, b = g.iloc[0], g.iloc[1]
            ov = min(a["overrun_end_s"], b["overrun_end_s"]) - max(a["overrun_start_s"], b["overrun_start_s"])
            relation = (f"overlapping overruns ({ov:.2f} s overlap)" if ov > 0 else
                        f"sequential overruns ({-ov:.2f} s apart): {a['signal']} first, then {b['signal']}")
        has_stop = isinstance(ev.stop_id, str) and ev.stop_id in st.index
        rows.append({
            "event": ev.event, "cycle": int(ev.cycle), "n_equipment_overruns": int(len(g)),
            "equipment_sequence": seq, "overrun_relation": relation,
            "stop_window": ev.stop_id if has_stop else "",
            "measured_standstill_s": float(st.loc[ev.stop_id, "duration_s"]) if has_stop else np.nan,
            "incident_value": ("one equipment overrun (value at equipment level)" if len(g) == 1 else
                               "no additive value: each equipment overrun is quantified separately"),
        })
    return pd.DataFrame(rows)


def cycle_quantification(cref: pd.DataFrame, cycles: pd.DataFrame, events: pd.DataFrame,
                         incidents: pd.DataFrame) -> pd.DataFrame:
    """Cycle-level excess of every complete cycle in s and in % of its position-adjusted reference, with the
    range over the clean-cycle fixed part, plus the incomplete cycle(s) with events (no reference there)."""
    out = cref[["cycle", "events", "observed_s", "reference_s", "reference_low_s", "reference_high_s",
                "cycle_excess_s"]].copy()
    out["events"] = out["events"].fillna("")
    out["cycle_excess_low_s"] = out["observed_s"] - out["reference_high_s"]
    out["cycle_excess_high_s"] = out["observed_s"] - out["reference_low_s"]
    out["cycle_excess_pct"] = [_pct(x, r) for x, r in zip(out["cycle_excess_s"], out["reference_s"])]
    out["cycle_excess_low_pct"] = [_pct(x, r) for x, r in zip(out["cycle_excess_low_s"], out["reference_high_s"])]
    out["cycle_excess_high_pct"] = [_pct(x, r) for x, r in zip(out["cycle_excess_high_s"], out["reference_low_s"])]
    out["status"] = np.where(out["events"] == "", "clean cycle (defines the reference)", "complete cycle with events")
    extra = []
    for c in sorted(set(events["cycle"]) - set(cref["cycle"])):
        evs = events[events["cycle"] == c]["event"].tolist()
        inc = incidents[incidents["event"].isin(evs)]
        extra.append({"cycle": int(c), "events": " ".join(evs),
                      "observed_s": float(cycles.loc[cycles["cycle"] == c, "duration_s"].iloc[0]),
                      "status": "incomplete cycle (recording ends): no reference, no cycle excess; measured "
                                "standstill " + "; ".join(f"{r.event} {r.stop_window} {r.measured_standstill_s:.2f} s"
                                                         for r in inc.itertuples() if r.stop_window)})
    return pd.concat([out, pd.DataFrame(extra)], ignore_index=True)


def event_cycles_total(cyc: pd.DataFrame) -> dict:
    """Total excess of the complete cycles with events. Cycles are disjoint time spans, so their excesses add."""
    e = cyc[cyc["status"] == "complete cycle with events"]
    return {"cycles": [int(c) for c in e["cycle"]],
            "excess_s": float(e["cycle_excess_s"].sum()), "excess_low_s": float(e["cycle_excess_low_s"].sum()),
            "excess_high_s": float(e["cycle_excess_high_s"].sum()), "reference_s": float(e["reference_s"].sum()),
            "excess_pct": _pct(e["cycle_excess_s"].sum(), e["reference_s"].sum()),
            "excess_low_pct": _pct(e["cycle_excess_low_s"].sum(), e["reference_high_s"].sum()),
            "excess_high_pct": _pct(e["cycle_excess_high_s"].sum(), e["reference_low_s"].sum())}


def ml_quantification(if_sweep: pd.DataFrame, if_sum: dict, mlcmp: pd.DataFrame, km_sum: dict,
                      km_members: pd.DataFrame, candidates: pd.DataFrame, unexplained_label: str) -> pd.DataFrame:
    """Model 2 counts, each traced to an evaluation output (column `source`). Anomaly scores are rankings, so
    no lost seconds or percentages are derived from them."""
    rows = []

    def add(method, metric, value, source):
        rows.append({"method": method, "metric": metric, "value": value, "source": source})

    for fs, g in if_sweep.groupby("feature_set", sort=False):
        m = f"Isolation Forest {fs}"
        for r in g.itertuples():
            add(m, f"anomalies at contamination {r.contamination:g} (count set by the parameter)", int(r.flagged),
                "ml_contamination_sweep.flagged")
            add(m, f"anomalies at contamination {r.contamination:g} that are rule flags (of {int(r.rule_flags_total)})",
                int(r.rule_flags_covered), "ml_contamination_sweep.rule_flags_covered")
        s = if_sum[fs]
        add(m, "average precision vs. rule flags (mean over seeds)", round(float(s["average_precision_mean"]), 4),
            "ml_isolation_forest.average_precision_vs_rule (mean over seeds)")
        add(m, "rule flags in the top 10 (mean over seeds)", round(float(s["rule_flags_in_top_k_mean"]), 2),
            "ml_isolation_forest.rule_flags_in_top10 (mean over seeds)")
        top = mlcmp[mlcmp["feature_set"] == fs]
        is_rule = top["explanation"].str.startswith("rule overrun")
        add(m, "top-20 items that are rule overruns (overlap with Model 1)", int(is_rule.sum()), "ml_vs_rule.explanation")
        add(m, "additional candidates (top-20 items not flagged by the rule)", int((~is_rule).sum()),
            "ml_vs_rule.explanation")
        add(m, "additional candidates longer than normal and unexplained",
            int((top["explanation"] == unexplained_label).sum()), "ml_vs_rule.explanation")
    if_c = candidates[candidates["source"].str.startswith("Model 2 Isolation Forest")]
    add("Isolation Forest A + B", "distinct additional candidates (an activation in both top-20 lists counted once)",
        int(len(if_c)), "candidate_register (Isolation Forest rows)")
    add("Isolation Forest", "base rate (rule flags / scored activations)", round(float(if_sum["base_rate"]), 4),
        "rule flags / model rows (notebook 05)")
    add("K-Means", f"rule overruns in the cluster of long runs (k = {km_sum['best_k_by_silhouette']})",
        int(km_sum["best_k_flags_in_cluster"]), "ml_kmeans.rule_flags_in_best_cluster (best k)")
    add("K-Means", "additional candidates (cluster members not flagged by the rule)", int(len(km_members)),
        "km_members")
    add("K-Means", "additional candidates longer than normal and unexplained",
        int((km_members["explanation"] == unexplained_label).sum()) if len(km_members) else 0,
        "km_members.explanation")
    ml_c = candidates[candidates["source"].str.startswith("Model 2")]
    add("Model 2 (all)", "independently verified additional losses",
        int((ml_c["status"] == "evidence-supported loss").sum()), "candidate_register.status")
    add("Model 2 (all)", "lost seconds / percentage from anomaly scores",
        NOT_AVAILABLE + ": anomaly scores are rankings, not durations; seconds need physical evidence", "definition")
    return pd.DataFrame(rows)


def _ml_value(ml_q: pd.DataFrame, method_start: str, metric_start: str):
    m = ml_q[ml_q["method"].str.startswith(method_start) & ml_q["metric"].str.startswith(metric_start)]
    return m["value"].tolist()


def results_by_level(equipment: pd.DataFrame, incidents: pd.DataFrame, cyc_total: dict, perf: pd.DataFrame,
                     ml_q: pd.DataFrame) -> pd.DataFrame:
    """The quantities kept strictly apart. No row adds equipment seconds; official OEE is not computable."""
    pos = perf[perf["method"] == "position-adjusted"].set_index("reference_basis")["performance"]
    single = perf[perf["method"] == "single reference"].set_index("reference_basis")["performance"]
    primary = pos[pos.index.str.contains("primary")].iloc[0]
    fastest = single[single.index.str.startswith("fastest")].iloc[0]
    multi = incidents[incidents["n_equipment_overruns"] > 1]
    # incidents outside the complete event cycles = the incomplete final cycle (only standstill measurable)
    final = incidents[~incidents["cycle"].isin(cyc_total["cycles"]) & (incidents["stop_window"] != "")]
    verified = _ml_value(ml_q, "Model 2", "independently verified")[0]
    eq_pct = equipment["excess_relative_to_median_normal_duration_pct"]
    return pd.DataFrame([
        {"level": "1 equipment", "quantity": "excess of each flagged step over the median normal duration of its "
                                              "own output",
         "value": f"{equipment['excess_s'].min():.2f}-{equipment['excess_s'].max():.2f} s per overrun "
                  f"({len(equipment)} overruns); excess relative to median normal duration "
                  f"{eq_pct.min():.0f}-{eq_pct.max():.0f} %",
         "basis": "equipment_quantification",
         "limitation": "diagnoses where a step took too long; stations work in parallel, so these seconds are "
                       "not additive and are not production-line loss"},
        {"level": "2 incident", "quantity": "equipment overruns grouped per event, in chronological order",
         "value": f"{len(incidents)} incidents; " + "; ".join(f"{r.event}: {r.overrun_relation}"
                                                             for r in multi.itertuples()),
         "basis": "incident_quantification", "limitation": "no additive incident value is assigned"},
        {"level": "3 cycle", "quantity": "additional production-cycle time (observed - position-adjusted reference)",
         "value": f"{cyc_total['excess_s']:.2f} s ({cyc_total['excess_low_s']:.2f}-{cyc_total['excess_high_s']:.2f} s)"
                  f" = {cyc_total['excess_pct']:.1f} % ({cyc_total['excess_low_pct']:.1f}-"
                  f"{cyc_total['excess_high_pct']:.1f} %) of the reference of cycles "
                  + ", ".join(map(str, cyc_total["cycles"])),
         "basis": "cycle_quantification (range over the clean-cycle fixed part)",
         "limitation": "complete cycles only; the reference rests on 4 clean cycles"},
        {"level": "3 cycle (incomplete)", "quantity": "measured standstill in the final, incomplete cycle",
         "value": "; ".join(f"{r.event} {r.stop_window} {r.measured_standstill_s:.2f} s" for r in final.itertuples()),
         "basis": "stops (stop-window duration)",
         "limitation": "measured standstill duration, not automatically excess production time (no reference for "
                       "an incomplete cycle)"},
        {"level": "4 estimated performance", "quantity": "sum of reference times / operating time of the complete "
                                                         "cycles",
         "value": f"{100 * primary:.1f} % ({100 * pos.min():.1f}-{100 * pos.max():.1f} %)",
         "basis": "performance (position-adjusted; range over the clean-cycle fixed part)",
         "limitation": "relative to demonstrated normal operation: an upper bound of OEE performance"},
        {"level": "5 official OEE performance", "quantity": "theoretical cycle time x parts / up time",
         "value": f"{NOT_AVAILABLE}: theoretical (ideal) cycle time not given (A4); with one ideal time per part "
                  f"it is at most {100 * fastest:.1f} %",
         "basis": "performance (fastest clean cycle = upper bound of a single ideal time)",
         "limitation": "needs the design cycle time"},
        {"level": "Model 2", "quantity": "independently verified additional losses from machine learning",
         "value": f"{verified}; anomaly scores give no seconds or percentages",
         "basis": "ml_quantification", "limitation": "anomalies are candidates, not confirmed losses"},
    ])


def method_comparison(ctx: dict, equipment: pd.DataFrame, cyc_total: dict, ml_q: pd.DataFrame,
                      levels: pd.DataFrame) -> pd.DataFrame:
    """One row per method: purpose, finding, quantified loss and limitation (all values from the run)."""
    quiet, lvl = ctx["quiet"], levels.set_index("level")["value"]
    v = lambda method, metric: _ml_value(ml_q, method, metric)  # noqa: E731
    unexpl_if = sum(v("Isolation Forest", "additional candidates longer"))
    return pd.DataFrame([
        {"method": "Model 1 - engineering rule + PLC evidence",
         "purpose": "find steps far longer than normal and confirm them with independent signals",
         "finding": f"{len(equipment)} equipment overruns in {equipment['event'].nunique()} incidents",
         "quantified_loss": lvl["1 equipment"],
         "limitation": "equipment level only; stop vs slow not separable for belt and small conveyor"},
        {"method": "Model 2 - Isolation Forest (sets A, B)", "purpose": "look for abnormal activations the rule misses",
         "finding": f"additional candidates in the top 20: A {v('Isolation Forest A', 'additional candidates (top')[0]}, "
                    f"B {v('Isolation Forest B', 'additional candidates (top')[0]}; longer than normal and "
                    f"unexplained: {unexpl_if}",
         "quantified_loss": "none: anomaly scores are rankings, not seconds",
         "limitation": "anomaly count set by contamination; agreement with the rule is not validation"},
        {"method": "Model 2 - K-Means", "purpose": "test for natural groups of activations",
         "finding": f"cluster of long runs: {v('K-Means', 'rule overruns in the cluster')[0]} rule overruns + "
                    f"{v('K-Means', 'additional candidates (cluster')[0]} other runs; unexplained: "
                    f"{v('K-Means', 'additional candidates longer')[0]}",
         "quantified_loss": "none", "limitation": "weak overall structure (low silhouette)"},
        {"method": "Line-quiet check (H8)", "purpose": "find whole-line halts the duration rule cannot see",
         "finding": f"{len(quiet)} long quiet periods, {int((quiet['explained_by'] == 'NOT EXPLAINED').sum())} "
                    "unexplained",
         "quantified_loss": "none beyond the events", "limitation": "limit taken from the 4 clean cycles"},
        {"method": "Cycle-level evaluation (H7)", "purpose": "measure the additional production time",
         "finding": f"cycles {', '.join(map(str, cyc_total['cycles']))} longer than their position-adjusted "
                    f"reference; permutation p = {ctx['eff']['permutation_p']:.3f}",
         "quantified_loss": lvl["3 cycle"], "limitation": "complete cycles only; E7, E8 lie in the incomplete cycle"},
        {"method": "Performance estimate", "purpose": "express the cycle-level loss as performance",
         "finding": "performance relative to demonstrated normal operation",
         "quantified_loss": lvl["4 estimated performance"],
         "limitation": "upper bound; official OEE performance not computable (no ideal cycle time)"},
    ])


def quantified_results(ctx: dict) -> dict:
    """All quantification tables of one run (used by the evaluation stage, notebook 06)."""
    eq = equipment_quantification(ctx["flags"], ctx["stops"])
    inc = incident_quantification(ctx["events"], eq, ctx["stops"])
    cyc = cycle_quantification(ctx["cref"], ctx["cycles"], ctx["events"], inc)
    tot = event_cycles_total(cyc)
    mlq = ml_quantification(ctx["if_sweep"], ctx["ml"]["isolation_forest"], ctx["mlcmp"], ctx["ml"]["kmeans"],
                            ctx["km_members"], ctx["candidates"], UNEXPLAINED)
    lv = results_by_level(eq, inc, tot, ctx["perf"], mlq)
    return {"equipment": eq, "incidents": inc, "cycles": cyc, "event_cycles_total": tot, "ml": mlq,
            "levels": lv, "methods": method_comparison(ctx, eq, tot, mlq, lv)}
