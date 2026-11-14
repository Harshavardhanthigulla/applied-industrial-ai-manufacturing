"""Model 1 - engineering hypotheses and rule-based detection (notebook 04).

Step-overrun rule (Tukey fence + detection limit), independent PLC evidence,
event consolidation and confidence, physical attribution, reduced-speed tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy import stats as sps

from .data import STOP_TOL_S, cut_by_stop, edge_times, on_intervals, overlaps, resumed_after_stop, step_before


# --------------------------------------------------------------------------
# R1: step overrun
# --------------------------------------------------------------------------
def scored_activations(acts: pd.DataFrame) -> pd.DataFrame:
    """Activations usable for duration statistics: in production, complete."""
    return acts[acts["in_production"] & ~acts["open_at_start"] & ~acts["open_at_end"]].copy()


def normal_durations(acts: pd.DataFrame) -> pd.DataFrame:
    g = acts.groupby("signal")["duration"]
    st = pd.DataFrame({"n": g.size(), "median": g.median(), "q1": g.quantile(0.25),
                       "q3": g.quantile(0.75), "max": g.max(), "min": g.min()})
    st["iqr"] = st["q3"] - st["q1"]
    return st


def flag_long(acts: pd.DataFrame, stats: pd.DataFrame, k: float, min_excess_s: float,
              min_n: int) -> pd.DataFrame:
    d = acts.join(stats[["n", "median", "q3", "iqr"]], on="signal")
    d["fence"] = d["q3"] + k * d["iqr"]
    d["excess"] = d["duration"] - d["median"]
    hit = (d["n"] >= min_n) & (d["duration"] > d["fence"]) & (d["excess"] >= min_excess_s)
    return d[hit].sort_values("start").reset_index(drop=True)


def group_events(flags: pd.DataFrame, merge_gap_s: float = 0.0) -> list[int]:
    """Event number per flagged activation: overlapping (or closer than
    ``merge_gap_s``) activations share a number."""
    numbers, n, end = [], 0, -np.inf
    for s, e in zip(flags["start"], flags["end"]):
        if s > end + merge_gap_s:
            n += 1
            end = e
        end = max(end, e)
        numbers.append(n)
    return numbers


def detect(acts: pd.DataFrame, k: float, min_excess_s: float, min_n: int,
           merge_gap_s: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = scored_activations(acts)
    stats = normal_durations(sc)
    flags = flag_long(sc, stats, k, min_excess_s, min_n)
    flags["event"] = [f"E{n}" for n in group_events(flags, merge_gap_s)]
    return flags, stats


def add_flag_evidence(flags: pd.DataFrame, acts: pd.DataFrame, data: dict) -> pd.DataFrame:
    """Timing uncertainty, worst case, comparison with the longest normal run."""
    f = flags.copy()
    tt = {p: data[p].frame["t"].to_numpy() for p in ("Siemens", "Beckhoff")}
    f["start_uncertainty_s"] = [step_before(tt[p], s) for p, s in zip(f["plc"], f["start"])]
    f["end_uncertainty_s"] = [step_before(tt[p], e) for p, e in zip(f["plc"], f["end"])]
    f["edge_uncertainty_s"] = f["start_uncertainty_s"] + f["end_uncertainty_s"]
    f["worst_case_excess_s"] = f["excess"] - f["edge_uncertainty_s"]
    sc = scored_activations(acts)
    key = set(zip(f["signal"], f["start"]))
    normal = sc[[k not in key for k in zip(sc["signal"], sc["start"])]]
    longest = normal.groupby("signal")["duration"].max()
    f["longest_normal_s"] = f["signal"].map(longest)
    f["above_longest_normal_s"] = f["duration"] - f["longest_normal_s"]
    f["overrun_start_s"] = f["start"] + f["median"]
    f["overrun_end_s"] = f["end"]
    return f


def threshold_sweep(acts: pd.DataFrame, ks, excesses, merge_gaps, min_n,
                    reference_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Number of flagged activations and events for every parameter combination.

    ``reference_events`` (frozen baseline) lets each row list which baseline
    events are still found."""
    rows = []
    for k in ks:
        for m in excesses:
            for g in merge_gaps:
                f, _ = detect(acts, k, m, min_n, g)
                found = []
                if reference_events is not None and len(f):
                    for _, ev in reference_events.iterrows():
                        if any(overlaps(a, b, ev["start_s"], ev["end_s"])
                               for a, b in zip(f["start"], f["end"])):
                            found.append(ev["event"])
                rows.append({"tukey_k": k, "min_excess_s": m, "merge_gap_s": g,
                             "flagged_activations": len(f),
                             "events": f["event"].nunique() if len(f) else 0,
                             "baseline_events_found": " ".join(found)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Standstill / slow evidence
# --------------------------------------------------------------------------
PAIRS = {
    "WH.Q3_horRack": "WH.Q3_horizontal_towards_rack",
    "WH.Q4_horBelt": "WH.Q4_horizontal_towards_conveyor_belt",
    "WH.Q5_verDown": "WH.Q5_vertical_downward",
    "WH.Q6_verUP": "WH.Q6_vertical_upward",
    "WH.Q7_cantForw": "WH.Q7_cantilever_forward",
    "WH.Q8_cantBack": "WH.Q8_cantilever_backward",
}


HOR_ENCODERS = ["WH.B1_encoder_horizontal_impuls1", "WH.B2_encoder_horizontal_impuls2"]


def paired_output_windows(data: dict, p0: float) -> pd.DataFrame:
    """Windows in which the request signal of a pair is ON but the motor is OFF."""
    b = data["Beckhoff"].frame
    rows = []
    for req, mot in PAIRS.items():
        for s, e in on_intervals(b["t"], b[req] & ~b[mot]):
            if s >= p0:
                rows.append({"request": req, "motor": mot, "start_s": s, "end_s": e,
                             "duration_s": e - s})
    return pd.DataFrame(rows)


def encoder_silence(data: dict, acts: pd.DataFrame) -> pd.DataFrame:
    b = data["Beckhoff"].frame
    pulses = np.sort(np.concatenate([edge_times(b["t"], b[c].to_numpy()) for c in HOR_ENCODERS]))
    mv = acts[acts["signal"].isin(["WH.Q3_horRack", "WH.Q4_horBelt"]) & acts["in_production"]].copy()

    def longest(s, e):
        inside = pulses[(pulses > s) & (pulses < e)]
        return float(np.diff(np.concatenate(([s], inside, [e]))).max())

    mv["longest_encoder_silence_s"] = [longest(s, e) for s, e in zip(mv["start"], mv["end"])]
    return mv[["signal", "start", "end", "duration", "longest_encoder_silence_s"]]


def belt_arrivals(data: dict, acts: pd.DataFrame) -> pd.DataFrame:
    """Warehouse belt: time from belt start to the last light-barrier change."""
    b = data["Beckhoff"].frame
    barrier = np.sort(np.concatenate([edge_times(b["t"], b[c].to_numpy()) for c in
                                      ("WH.I2_Light_barrier_inside", "WH.I3_Light_barrier_outside")]))
    belt = acts[acts["signal"].isin(["WH.Q1_beltForward", "WH.Q2_beltBackward"]) & acts["in_production"]].copy()
    last = [barrier[(barrier >= s) & (barrier <= e)] for s, e in zip(belt["start"], belt["end"])]
    belt["barrier_reached_after_s"] = [x.max() - s if len(x) else np.nan for x, s in zip(last, belt["start"])]
    belt["belt_off_after_barrier_s"] = [e - x.max() if len(x) else np.nan for x, e in zip(last, belt["end"])]
    return belt[["signal", "start", "end", "duration", "barrier_reached_after_s", "belt_off_after_barrier_s"]]


def barrier_changes_with_belt_off(data: dict, p0: float) -> pd.DataFrame:
    b = data["Beckhoff"].frame
    on = (b["WH.Q1_beltForward"] | b["WH.Q2_beltBackward"]).to_numpy()
    both_off = ~on & ~np.r_[True, on[:-1]]
    rows = []
    for c in ("WH.I2_Light_barrier_inside", "WH.I3_Light_barrier_outside"):
        v = b[c].to_numpy()
        ch = np.r_[False, v[1:] != v[:-1]]
        for t in b["t"].to_numpy()[ch & both_off]:
            if t >= p0:
                rows.append({"barrier": c, "t_s": t})
    return pd.DataFrame(rows)


def small_conveyor_arrivals(data: dict, acts: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Small conveyor A0.7: time until the part breaks light barrier E1.0.

    A run without any E1.0 edge is checked against the logging gaps: if a
    Siemens gap longer than a normal E1.0 pulse lies inside the run, the
    missing edge is explained by logging, otherwise the part did not arrive.
    """
    s = data["Siemens"].frame
    e10 = edge_times(s["t"], s["E1.0"].to_numpy())
    conv = acts[(acts["signal"] == "A0.7") & acts["in_production"]].copy()
    sg = gaps[gaps["plc"] == "Siemens"]
    first, gap_inside = [], []
    for a, e in zip(conv["start"], conv["end"]):
        inside = e10[(e10 >= a) & (e10 <= e)]
        first.append(inside.min() - a if len(inside) else np.nan)
        g = sg[(sg["gap_start_s"] >= a) & (sg["gap_end_s"] <= e + 1.0)]
        gap_inside.append(float(g["gap_s"].max()) if len(g) else 0.0)
    conv["part_at_barrier_after_s"] = first
    conv["largest_siemens_gap_in_run_s"] = gap_inside
    conv["no_arrival_explained_by_gap"] = conv["part_at_barrier_after_s"].isna() & (conv["largest_siemens_gap_in_run_s"] > 0.4)
    return conv[["signal", "start", "end", "duration", "open_at_end", "part_at_barrier_after_s",
                 "largest_siemens_gap_in_run_s", "no_arrival_explained_by_gap"]]


def barrier_edges_without_drive(data: dict, p0: float) -> pd.DataFrame:
    """Edges of the conveyor light barriers E1.0 / E0.1 while neither the small
    conveyor (A0.7) nor the long conveyor (A0.0) runs: the part moved without
    its drive (e.g. by hand)."""
    s = data["Siemens"].frame
    drive = (s["A0.7"] | s["A0.0"]).to_numpy()
    off = ~drive & ~np.r_[True, drive[:-1]]
    rows = []
    for c in ("E1.0", "E0.1"):
        v = s[c].to_numpy()
        ch = np.r_[False, v[1:] != v[:-1]]
        for t in s["t"].to_numpy()[ch & off]:
            if t >= p0:
                rows.append({"barrier": c, "t_s": t})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# R2: inactivity (line-wide and per station)
# --------------------------------------------------------------------------
def actuator_edge_times(data: dict, actuators: dict, p0: float, station: str | None = None) -> np.ndarray:
    out = []
    for plc, sigs in actuators.items():
        d = data[plc].frame
        doc = data["doc"][plc].set_index("signal")
        use = [c for c in sigs if station is None or (c in doc.index and doc.loc[c, "station"] == station)]
        if not use:
            continue
        v = d[use].astype(int).to_numpy()
        ch = np.r_[False, (np.abs(np.diff(v, axis=0)).sum(axis=1) > 0)]
        out.append(d["t"].to_numpy()[ch])
    t = np.sort(np.concatenate(out)) if out else np.array([])
    return t[t >= p0]


def quiet_periods(times: np.ndarray) -> pd.DataFrame:
    q = pd.DataFrame({"start_s": times[:-1], "end_s": times[1:]})
    q["duration_s"] = q["end_s"] - q["start_s"]
    return q


def quiet_completeness(data: dict, actuators: dict, p0: float, cycles_done: pd.DataFrame,
                       events: pd.DataFrame, acts: pd.DataFrame, flag_keys: set) -> tuple[pd.DataFrame, float]:
    """Every period without any actuator edge (both PLCs) that is longer than
    the longest such period inside clean cycles must lie inside an event or
    inside a movement of normal length. Returns the table and the limit."""
    t = actuator_edge_times(data, actuators, p0)
    q = quiet_periods(t)
    clean = cycles_done[cycles_done["events"] == ""]
    in_clean = np.zeros(len(q), bool)
    for _, c in clean.iterrows():
        in_clean |= ((q["start_s"] >= c["start_s"]) & (q["end_s"] <= c["end_s"])).to_numpy()
    limit = float(q.loc[in_clean, "duration_s"].max())
    sc = scored_activations(acts)
    normal = sc[[k not in flag_keys for k in zip(sc["signal"], sc["start"])]]
    longest = normal.groupby("signal")["duration"].max()
    long_q = q[q["duration_s"] > limit].copy()

    def why(a, b):
        for _, ev in events.iterrows():
            if overlaps(a, b, ev["start_s"], ev["end_s"]):
                return f"inside event {ev['event']}"
        cov = acts[(acts["start"] <= a) & (acts["end"] >= b)]
        for _, r in cov.iterrows():
            if r["signal"] in longest.index and r["duration"] <= longest[r["signal"]]:
                return f"normal move {r['signal']} ({r['duration']:.2f} s <= {longest[r['signal']]:.2f} s)"
        return "NOT EXPLAINED"

    long_q["explained_by"] = [why(a, b) for a, b in zip(long_q["start_s"], long_q["end_s"])]
    return long_q.reset_index(drop=True), limit


def describe_signal(data: dict, plc: str, signal: str, dictionary: pd.DataFrame) -> str:
    row = dictionary[(dictionary["plc"] == plc) & (dictionary["signal"] == signal)]
    if row.empty:
        return signal
    r = row.iloc[0]
    text = r["description"] if plc == "Siemens" and r["description"] else r["interpretation"]
    text = str(text).replace("(output)", "").strip()
    return f"{r['station']}: {text} ({signal})"


def event_presence(sweep_flags: dict, events: pd.DataFrame) -> pd.Series:
    """Share of sweep settings in which each event is found again."""
    found = {e: 0 for e in events["event"]}
    for _, flags in sweep_flags.items():
        for _, ev in events.iterrows():
            if any(overlaps(a, b, ev["start_s"], ev["end_s"]) for a, b in zip(flags["start"], flags["end"])):
                found[ev["event"]] += 1
    return pd.Series({k: v / len(sweep_flags) for k, v in found.items()})


def consolidate(flags: pd.DataFrame, data: dict, dictionary: pd.DataFrame, stops: pd.DataFrame,
                drive_off: pd.DataFrame, enc: pd.DataFrame, cycles: pd.DataFrame,
                min_excess_s: float, halt: pd.DataFrame) -> pd.DataFrame:
    """One row per event with timing, uncertainty, stop evidence and category."""
    starts = cycles["start_s"].to_numpy()
    rows = []
    for name, g in flags.groupby("event", sort=False):
        a, b = g["start"].min(), g["end"].max()
        hit = stops[[overlaps(a, b, x, y) for x, y in zip(stops["start_s"], stops["end_s"])]]
        off = drive_off[[overlaps(a, b, x, y) for x, y in zip(drive_off["start_s"], drive_off["end_s"])]] \
            if len(drive_off) else drive_off
        silent = enc[(enc["start"] >= a) & (enc["start"] <= b) & (enc["longest_encoder_silence_s"] >= min_excess_s)]
        halted = False
        if len(hit):
            halted = bool((halt.set_index("stop_id").loc[hit["stop_id"], "actuator_edges_inside"] == 0).all())
        standstill = bool(len(off) or len(silent) or halted)
        first_abnormal = min(list(g["start"] + g["median"]) + list(off["start_s"] if len(off) else []))
        if len(hit):
            st = hit.iloc[0]
            lead = st["start_s"] - first_abnormal
            mech = "protective stop (light curtain)" if st["kind"].startswith("protective") else "HMI stop"
            category = "minor stop"
        else:
            st, lead, mech = None, np.nan, "step overrun, no stop signal"
            category = "minor stop" if standstill else "minor stop or reduced speed (not separable)"
        rows.append({
            "event": name,
            "cycle": int(np.searchsorted(starts, a, side="right")),
            "start_s": a, "end_s": b,
            "overrun_start_s": float((g["start"] + g["median"]).min()),
            "overrun_end_s": b,
            "equipment_lost_time_s": float(g["excess"].max()),
            "worst_case_lost_time_s": float(g["worst_case_excess_s"].max()),
            "boundary_uncertainty_s": float(g["edge_uncertainty_s"].max()),
            "n_flagged_activations": len(g),
            "signals": " ".join(g["signal"]),
            "equipment": "; ".join(describe_signal(data, p, s, dictionary) for p, s in zip(g["plc"], g["signal"])),
            "stop_id": st["stop_id"] if st is not None else "",
            "stop_mechanism": mech,
            "stop_start_s": st["start_s"] if st is not None else np.nan,
            "stop_end_s": st["end_s"] if st is not None else np.nan,
            "stop_duration_s": st["duration_s"] if st is not None else np.nan,
            "abnormal_before_stop_s": lead,
            "standstill_proven": standstill,
            "standstill_basis": "; ".join(filter(None, [
                "motor output OFF while request ON" if len(off) else "",
                "horizontal encoder silent" if len(silent) else "",
                "no actuator switched during the stop window" if halted else ""])),
            "loss_category": category,
        })
    return pd.DataFrame(rows)


def assign_confidence(events: pd.DataFrame, presence: pd.Series, flags: pd.DataFrame,
                      limit_s: float) -> pd.DataFrame:
    """Confidence that the event is a real loss (not a measurement artefact).

    high   : worst-case excess > 0, longer than the longest normal run by more
             than the detection limit, found in >= 90 % of sweep settings
    medium : real after uncertainty, but sensitive to the threshold
    low    : otherwise
    """
    e = events.copy()
    e["sweep_presence"] = e["event"].map(presence)
    above = flags.groupby("event")["above_longest_normal_s"].max()
    worst = flags.groupby("event")["worst_case_excess_s"].max()
    conf = []
    for _, r in e.iterrows():
        ok_real = worst[r["event"]] > 0 and above[r["event"]] >= limit_s
        if ok_real and r["sweep_presence"] >= 0.9:
            conf.append("high")
        elif worst[r["event"]] > 0:
            conf.append("medium")
        else:
            conf.append("low")
    e["existence_confidence"] = conf
    return e


PAIR_NORMAL_LAG_S = 0.26          # measured, see paired_output_windows


STATION_OF_PREFIX = {"WH.": "Warehouse", "VG.": "Robot"}


def _station(row_signal: str, plc: str, data: dict) -> str:
    doc = data["doc"][plc].set_index("signal")
    return doc.loc[row_signal, "station"] if row_signal in doc.index else ""


def stations_active(acts: pd.DataFrame, a: float, b: float) -> str:
    """Stations with an actuator edge (start or end of an activation) in [a, b)."""
    s1 = acts[(acts["start"] >= a) & (acts["start"] < b)]["station"]
    s2 = acts[(acts["end"] >= a) & (acts["end"] < b)]["station"]
    return ", ".join(sorted(set(s1) | set(s2))) or "none"


def first_resumed(acts: pd.DataFrame, t: float) -> str:
    nxt = acts[acts["start"] > t].sort_values("start").drop_duplicates("station")
    return "; ".join(f"{r.station} {r.signal} @{r.start:.2f}s" for r in nxt.head(4).itertuples())


def expected_vs_observed(flag: pd.Series, belt: pd.DataFrame, conv: pd.DataFrame,
                         drive_off: pd.DataFrame, stops: pd.DataFrame) -> tuple[str, str]:
    sig = flag["signal"]
    if sig in ("WH.Q1_beltForward", "WH.Q2_beltBackward"):
        normal = belt[(belt["signal"] == sig) & (belt["duration"] < flag["fence"])]["barrier_reached_after_s"]
        mine = belt[(belt["signal"] == sig) & (np.isclose(belt["start"], flag["start"]))]["barrier_reached_after_s"]
        return (f"pallet reaches the belt light barrier within {normal.max():.2f} s of belt start",
                f"reached after {mine.iloc[0]:.2f} s; belt ON {flag['duration']:.2f} s")
    if sig == "A0.7":
        normal = conv[conv["duration"] < flag["fence"]]["part_at_barrier_after_s"].dropna()
        mine = conv[np.isclose(conv["start"], flag["start"])]["part_at_barrier_after_s"]
        return (f"part at light barrier E1.0 within {normal.max():.2f} s",
                f"part at E1.0 after {mine.iloc[0]:.2f} s")
    if sig == "A1.0":
        return (f"drill OFF after its normal {flag['median']:.2f} s (drill timer, M6.4)",
                f"drill ON {flag['duration']:.2f} s, switched off only when the stop began")
    if sig in ("WH.Q3_horRack", "WH.Q4_horBelt", "WH.Q7_cantForw", "WH.Q8_cantBack"):
        off = drive_off[(drive_off["request"] == sig) & (drive_off["start_s"] >= flag["start"] - 0.01)
                        & (drive_off["start_s"] <= flag["end"])]
        if len(off):
            o = off.iloc[0]
            return (f"motor output follows the request within {PAIR_NORMAL_LAG_S} s",
                    f"motor output OFF for {o['duration_s']:.2f} s ({o['start_s']:.2f}-{o['end_s']:.2f} s) "
                    f"while the request stayed ON")
    return (f"step ends after its normal {flag['median']:.2f} s",
            f"ON {flag['duration']:.2f} s")


def attribute(events: pd.DataFrame, flags: pd.DataFrame, acts: pd.DataFrame, data: dict,
              stops: pd.DataFrame, drive_off: pd.DataFrame, belt: pd.DataFrame,
              conv: pd.DataFrame, no_drive: pd.DataFrame, belt_off_barrier: pd.DataFrame,
              mw22: pd.DataFrame, tol: float, cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ev in events.iterrows():
        g = flags[flags["event"] == ev["event"]].sort_values("overrun_start_s")
        lead = g.iloc[0]
        a = float(ev["overrun_start_s"])
        exp, obs = expected_vs_observed(lead, belt, conv, drive_off, stops)
        phase = mw22[(mw22["start_s"] <= a) & (mw22["end_s"] > a)]
        evidence, precursors = [], []

        # ----- precursor candidates before a stop -------------------------
        stop_kind = ev["stop_mechanism"]
        if stop_kind.startswith("protective") or stop_kind == "HMI stop":
            s0, s1 = ev["stop_start_s"], ev["stop_end_s"]
            if ev["abnormal_before_stop_s"] > tol:
                precursors.append(
                    (s0 - ev["abnormal_before_stop_s"], _station(lead["signal"], lead["plc"], data),
                     lead["signal"],
                     f"{lead['signal']} abnormal {ev['abnormal_before_stop_s']:.2f} s before the stop "
                     f"({obs})"))
            # small-conveyor runs cut by this stop without part arrival
            cut = conv[(conv["end"] >= s0 - tol) & (conv["end"] <= s0 + tol)
                       & conv["part_at_barrier_after_s"].isna() & ~conv["no_arrival_explained_by_gap"]]
            normal_arr = conv["part_at_barrier_after_s"].dropna()
            normal_arr = normal_arr[normal_arr <= 1.5].max()
            for r in cut.itertuples():
                overdue = (s0 - r.start) - normal_arr
                precursors.append((r.start + normal_arr, "Drilling station", "A0.7",
                                   f"small conveyor A0.7 ran {s0 - r.start:.2f} s without the part reaching "
                                   f"E1.0 (normal <= {normal_arr:.2f} s): {overdue:.2f} s overdue at the stop"))
                later = acts[(acts["signal"] == "A0.7") & (acts["start"] > s0)]
                if later.empty:
                    evidence.append("A0.7 never runs again until the end of the recording")
            md = no_drive[(no_drive["t_s"] >= s0 - tol) & (no_drive["t_s"] <= s1 + tol)]
            if len(md):
                evidence.append(f"{len(md)} edges of light barriers {', '.join(sorted(md['barrier'].unique()))} "
                                f"while both conveyors were OFF ({md['t_s'].min():.2f}-{md['t_s'].max():.2f} s): "
                                f"the part moved without its drive (consistent with manual handling)")
            if stop_kind.startswith("protective"):
                evidence.append(f"protective stop {s0:.2f}-{s1:.2f} s: A201.0 LOW, red lamp M0.3 and "
                                f"long-curtain SR bit M24.3 ON, ended by HMI reset E200.0")
            else:
                evidence.append(f"HMI stop {s0:.2f}-{s1:.2f} s: Visual.start LOW, Visual.Stoplight HIGH")
            affected = [f"{r.signal} (overdue {r.excess:.2f} s)" for r in g.itertuples()
                        if overlaps(r.overrun_start_s, r.end, s0, s1)]
            if affected:
                evidence.append("held by the stop: " + ", ".join(affected))
        else:
            # ----- overrun without stop signal -----------------------------
            evidence.append(f"expected: {exp}; observed: {obs}")
            if lead["signal"].startswith("WH.Q") and "belt" in lead["signal"]:
                cyc_bar = belt_off_barrier[(belt_off_barrier["barrier"] == "WH.I3_Light_barrier_outside")]
                c = cycles[cycles["cycle"] == ev["cycle"]].iloc[0]
                cyc_rows = cyc_bar[(cyc_bar["t_s"] >= c["start_s"]) & (cyc_bar["t_s"] < c["end_s"])]
                if len(cyc_rows):
                    evidence.append("outside barrier WH.I3 changed while the belt was OFF at "
                                    + ", ".join(f"{x:.2f}" for x in cyc_rows["t_s"]) + " s")

        # ----- first deviation = earliest abnormal moment ------------------
        cands = [(a, f"{lead['signal']} passed its normal {lead['median']:.2f} s")]
        off = drive_off[(drive_off["start_s"] >= ev["start_s"] - 0.01) & (drive_off["start_s"] <= ev["end_s"])]
        for o in off.itertuples():
            cands.append((o.start_s + PAIR_NORMAL_LAG_S,
                          f"{o.motor} still OFF {PAIR_NORMAL_LAG_S} s after {o.request} switched ON"))
        for p in precursors:
            cands.append((p[0], p[3].split(":")[0]))
        if stop_kind.startswith("protective") or stop_kind == "HMI stop":
            cands.append((ev["stop_start_s"], f"{stop_kind} began"))
        t_first, what_first = min(cands, key=lambda c: c[0])
        resume_ref = ev["stop_end_s"] if not np.isnan(ev["stop_end_s"]) else ev["end_s"]

        # ----- decision rules ------------------------------------------------
        if stop_kind.startswith("protective"):
            if precursors:
                precursors.sort()                       # earliest abnormal moment first
                _, st, sig, _ = precursors[0]
                component = f"{st} ({sig}) - abnormal before the protective stop"
                category = "Station/component candidate"
                # documented Siemens outputs -> medium; Beckhoff pair reading -> low
                loc_conf = "medium" if sig.startswith("A") else "low"
                evidence = [p[3] for p in precursors] + evidence
            else:
                component = "line-wide protective stop (safety light curtain); no precursor"
                category = "Unattributable"
                loc_conf = "n/a"
            root = "protective stop by the light curtain (documented); what interrupted the curtain is not recorded"
            root_conf = "high for the stop mechanism; trigger unknown"
        elif stop_kind == "HMI stop":
            if any("manual" in x for x in evidence):
                component = "Drilling -> sorting transfer (small conveyor A0.7 / barriers E1.0, E0.1)"
                category = "Station/component candidate"
                loc_conf = "medium"
                root = "operator stop at the HMI; part moved through the transfer barriers with conveyors OFF"
                root_conf = "medium (manual handling is an inference from barrier edges)"
            else:
                component = "operator stop (HMI)"
                category = "Unattributable"
                loc_conf = "n/a"
                root = "operator stop at the HMI; reason not recorded"
                root_conf = "high for the stop mechanism; reason unknown"
        else:
            component = f"{_station(lead['signal'], lead['plc'], data)} ({lead['signal']})"
            category = "Station/component candidate"
            loc_conf = "high" if lead["signal"] in ("WH.Q1_beltForward", "WH.Q2_beltBackward", "A0.7") else "medium"
            root = "unknown - no stop, alarm or interlock signal is recorded"
            root_conf = "n/a"

        rows.append({
            "event": ev["event"], "start_s": ev["start_s"], "end_s": ev["end_s"],
            "phase_at_overrun_start": (f"MW22={int(phase['value'].iloc[0])} ({phase['phase'].iloc[0]})"
                                       if len(phase) else ""),
            "stations_active_5s_before": stations_active(acts, a - 5, a),
            "first_deviation_s": t_first,
            "first_deviation": f"{what_first} at {t_first:.2f} s",
            "expected_transition": exp, "observed": obs,
            "resumed_after": first_resumed(acts, resume_ref),
            "stop_mechanism": stop_kind,
            "attribution_category": category, "component": component,
            "location_confidence": loc_conf, "root_cause": root, "root_cause_confidence": root_conf,
            "evidence": " | ".join(evidence),
        })
    return pd.DataFrame(rows)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------
# Repeated steps: dispersion and drift
# --------------------------------------------------------------------------
def step_dispersion(acts: pd.DataFrame, flag_keys: set, min_n: int = 5) -> pd.DataFrame:
    """Per output: spread of normal durations and drift over the recording.

    Excluded: flagged overruns, runs cut by a stop or restarted after one
    (their duration is truncated), and runs open at the recording borders.
    Drift test: Spearman correlation of duration with order, Holm-adjusted
    over all outputs; a drift is reported only if it is also larger than the
    timing resolution (0.2 s) between the two halves.
    """
    sc = acts[acts["in_production"] & ~acts["open_at_start"] & ~acts["open_at_end"]
              & ~acts["cut_by_stop"] & ~acts["resumed_after_stop"]]
    sc = sc[[k not in flag_keys for k in zip(sc["signal"], sc["start"])]]
    rows = []
    for sig, g in sc.groupby("signal"):
        if len(g) < min_n:
            continue
        g = g.sort_values("start")
        d = g["duration"].to_numpy()
        half = len(d) // 2
        rho, p = sps.spearmanr(np.arange(len(d)), d) if np.ptp(d) > 0 else (0.0, 1.0)
        rows.append({
            "plc": g["plc"].iloc[0], "signal": sig, "station": g["station"].iloc[0], "n": len(d),
            "median_s": np.median(d), "iqr_s": np.subtract(*np.percentile(d, [75, 25])),
            "min_s": d.min(), "max_s": d.max(),
            "robust_cv": (np.subtract(*np.percentile(d, [75, 25])) / 1.349) / np.median(d)
            if np.median(d) > 0 else np.nan,
            "first_half_median_s": np.median(d[:half]), "second_half_median_s": np.median(d[half:]),
            "half_change_s": np.median(d[half:]) - np.median(d[:half]),
            "spearman_rho_vs_time": rho, "spearman_p": p,
        })
    out = pd.DataFrame(rows).sort_values(["plc", "signal"]).reset_index(drop=True)
    out["spearman_p_holm"] = holm(out["spearman_p"].fillna(1.0).to_numpy())
    out["drift_detected"] = (out["spearman_p_holm"] < 0.05) & (out["half_change_s"].abs() > 0.2)
    return out


# --------------------------------------------------------------------------
# Warehouse horizontal axis: duration vs distance
# --------------------------------------------------------------------------
MOTORS = {"WH.Q3_horizontal_towards_rack": "towards rack",
          "WH.Q4_horizontal_towards_conveyor_belt": "towards belt"}


def warehouse_travel(data: dict, cycles: pd.DataFrame, stops: pd.DataFrame, p0: float) -> pd.DataFrame:
    """Horizontal motor runs with their travel distance.

    Position model (measured, see docs/methodology.md section 9): the belt is
    at x = 0 and a rack column at x = desiredPalletX. In each cycle the axis
    moves from the previous rack column to the new one, then to the belt, and
    after processing back to the rack column to store the pallet. Runs that
    overlap a stop window are excluded (they are split by the stop).
    """
    b = data["Beckhoff"].frame
    runs = []
    for sig, direction in MOTORS.items():
        for s, e in on_intervals(b["t"], b[sig]):
            runs.append((s, e, sig, direction))
    runs.sort()
    starts = cycles["start_s"].to_numpy()
    pos = None
    pending = None          # (from_x, to_x) of a run cut by a stop
    last_cycle = None
    rows = []
    for s, e, sig, direction in runs:
        # A move can be logged up to one Beckhoff step before the pallet
        # request change that triggers it (e.g. 98.39 s vs 98.50 s).
        c = int(np.searchsorted(starts, s + STOP_TOL_S, side="right"))
        rack = float(cycles.loc[cycles["cycle"] == c, "rack_x"].iloc[0])
        first_in_cycle = c != last_cycle
        last_cycle = c
        cut = cut_by_stop(s, e, stops)
        resumed = pending is not None and resumed_after_stop(s, stops)
        if resumed:
            frm, target = pending
            pending = None
        else:
            frm = pos
            if direction == "towards rack":
                target = rack
            else:
                target = rack if (first_in_cycle and pos is not None and rack < pos) else 0.0
        if cut:
            pending = (frm, target)
        else:
            pos = target
        rows.append({"signal": sig, "direction": direction, "start_s": s, "end_s": e,
                     "duration_s": e - s, "cycle": c, "from_x": frm, "to_x": target,
                     "distance": abs(target - frm) if frm is not None else np.nan,
                     "in_production": s >= p0, "cut_by_stop": cut, "resumed_after_stop": resumed})
    out = pd.DataFrame(rows)
    ok = out["in_production"] & ~out["cut_by_stop"] & ~out["resumed_after_stop"] & out["distance"].gt(0)
    x, y = out.loc[ok, "distance"].to_numpy(), out.loc[ok, "duration_s"].to_numpy()
    fit = sps.linregress(x, y)
    out["predicted_s"] = fit.intercept + fit.slope * out["distance"]
    out["residual_s"] = out["duration_s"] - out["predicted_s"]
    out["used_in_fit"] = ok
    out.attrs["fit"] = {"slope_s_per_unit": fit.slope, "intercept_s": fit.intercept,
                        "r2": fit.rvalue ** 2, "n": int(ok.sum()),
                        "speed_units_per_s": 1.0 / fit.slope,
                        "max_abs_residual_s": float(np.abs(out.loc[ok, "residual_s"]).max())}
    return out


# --------------------------------------------------------------------------
# Robot moves and drilling station
# --------------------------------------------------------------------------
def robot_summary(moves: pd.DataFrame) -> pd.DataFrame:
    g = moves.groupby(["move_type", "lane"])["duration_s"]
    out = g.agg(["count", "median", "min", "max"]).reset_index()
    out["range_s"] = out["max"] - out["min"]
    return out


def drilling_summary(dcyc: pd.DataFrame, acts: pd.DataFrame, flag_keys: set) -> dict:
    drill = acts[(acts["signal"] == "A1.0") & acts["in_production"]]
    normal = drill[[k not in flag_keys for k in zip(drill["signal"], drill["start"])]]
    return {"oven_phase_min_s": float(dcyc["oven_phase_s"].min()),
            "oven_phase_max_s": float(dcyc["oven_phase_s"].max()),
            "oven_phases": int(len(dcyc)),
            "drill_normal_min_s": float(normal["duration"].min()),
            "drill_normal_max_s": float(normal["duration"].max()),
            "part_interval_min_s": float(dcyc["part_interval_s"].min()),
            "part_interval_max_s": float(dcyc["part_interval_s"].max())}


