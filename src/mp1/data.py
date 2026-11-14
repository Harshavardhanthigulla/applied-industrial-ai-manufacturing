"""Data understanding and preparation (notebooks 02 and 03).

Loading and cleaning of the raw PLC files, signal dictionary, time base (sampling, gaps,
edges, clock check, grid loss), production start, stop windows, pallet cycles,
activations and stop-cut flags. Reusable equivalents of the cells in notebooks 02-03.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import re




# --------------------------------------------------------------------------
# Basic helpers used everywhere
# --------------------------------------------------------------------------
def on_intervals(t, x) -> list[tuple[float, float]]:
    """(start, end) of every period in which boolean ``x`` is True.

    start = time of the first True sample; end = time of the first False
    sample afterwards, or the last sample if still True at the end.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=bool).astype(int)
    change = np.diff(np.concatenate(([0], x, [0])))
    starts = np.flatnonzero(change == 1)
    ends = np.minimum(np.flatnonzero(change == -1), len(t) - 1)
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


def interval_table(t, x) -> pd.DataFrame:
    """Like ``on_intervals`` but also marks periods cut by the recording borders."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=bool).astype(int)
    change = np.diff(np.concatenate(([0], x, [0])))
    starts = np.flatnonzero(change == 1)
    ends_idx = np.flatnonzero(change == -1)
    out = pd.DataFrame({
        "start": t[starts],
        "end": t[np.minimum(ends_idx, len(t) - 1)],
        "open_at_start": starts == 0,
        "open_at_end": ends_idx >= len(t),
        "start_idx": starts,
        "end_idx": np.minimum(ends_idx, len(t) - 1),
    })
    out["duration"] = out["end"] - out["start"]
    return out


def edge_times(t, x) -> np.ndarray:
    """Times at which ``x`` changes value (time of the first new sample)."""
    x = np.asarray(x)
    return np.asarray(t, dtype=float)[1:][x[1:] != x[:-1]]


def step_before(t: np.ndarray, time: float) -> float:
    """Logging step that ends at the sample logged at ``time``.

    An edge logged at ``time`` happened somewhere in (time - step, time], so
    this is the timing uncertainty of that edge.
    """
    t = np.asarray(t, dtype=float)
    i = int(np.clip(np.searchsorted(t, time), 1, len(t) - 1))
    return float(t[i] - t[i - 1])


def overlaps(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


# --------------------------------------------------------------------------
# Sampling and gaps
# --------------------------------------------------------------------------
def sampling_stats(plcs: dict) -> pd.DataFrame:
    rows = []
    for name in ("Siemens", "Beckhoff"):
        t = plcs[name].frame["t"].to_numpy()
        dt = np.diff(t)
        rows.append({
            "plc": name, "rows": len(t), "first_s": t[0], "last_s": t[-1],
            "duration_s": t[-1] - t[0],
            "min_step_s": dt.min(), "p05_step_s": np.percentile(dt, 5),
            "median_step_s": np.median(dt), "mean_step_s": dt.mean(),
            "p95_step_s": np.percentile(dt, 95), "p99_step_s": np.percentile(dt, 99),
            "max_step_s": dt.max(),
            "steps_over_0p2s": int((dt > 0.2).sum()),
            "steps_over_0p5s": int((dt > 0.5).sum()),
            "steps_over_1s": int((dt > 1.0).sum()),
            "sum_of_steps_over_0p5s": float(dt[dt > 0.5].sum()),
        })
    return pd.DataFrame(rows)


def gap_table(plcs: dict, min_gap_s: float, tolerance_s: float) -> pd.DataFrame:
    """Logging gaps longer than ``min_gap_s`` and whether the other PLC also
    has a gap (> 2 x its median step) overlapping the same time."""
    gaps = {}
    medians = {}
    for name in ("Siemens", "Beckhoff"):
        t = plcs[name].frame["t"].to_numpy()
        dt = np.diff(t)
        medians[name] = np.median(dt)
        gaps[name] = (t[:-1], t[1:], dt)
    rows = []
    for name, other in (("Siemens", "Beckhoff"), ("Beckhoff", "Siemens")):
        s, e, d = gaps[name]
        os_, oe, od = gaps[other]
        other_big = od > 2 * medians[other]
        for a, b, g in zip(s[d > min_gap_s], e[d > min_gap_s], d[d > min_gap_s]):
            hit = other_big & (os_ <= b + tolerance_s) & (oe >= a - tolerance_s)
            rows.append({
                "plc": name, "gap_start_s": a, "gap_end_s": b, "gap_s": g,
                "other_plc_gap": bool(hit.any()),
                "other_plc_gap_s": float(od[hit].max()) if hit.any() else 0.0,
            })
    return pd.DataFrame(rows).sort_values("gap_start_s").reset_index(drop=True)


def resample_loss(plcs: dict, grids_ms: list[int]) -> pd.DataFrame:
    """How many boolean transitions survive on a regular last-value grid.

    For each grid the value of every boolean signal at each grid tick is the
    last logged value at or before the tick (what pandas ``asfreq``/``ffill``
    or ``merge_asof`` would produce). Lost transitions are short pulses that
    fall between two ticks. Edge time error is the shift of surviving edges.
    """
    rows = []
    for g in grids_ms:
        step = g / 1000.0
        total = kept = lost_pulses = 0
        shift = []
        for name in ("Siemens", "Beckhoff"):
            d = plcs[name]
            t = d.frame["t"].to_numpy()
            ticks = np.arange(t[0], t[-1] + 1e-9, step)
            pos = np.searchsorted(t, ticks, side="right") - 1
            for sig in d.signals:
                v = d.frame[sig]
                if v.dtype != bool or v.nunique() < 2:
                    continue
                v = v.to_numpy()
                raw_edges = int((v[1:] != v[:-1]).sum())
                r = v[pos]
                grid_edges_idx = np.flatnonzero(r[1:] != r[:-1]) + 1
                total += raw_edges
                kept += len(grid_edges_idx)
                lost_pulses += (raw_edges - len(grid_edges_idx)) // 2
                raw_t = edge_times(t, v)
                if len(grid_edges_idx) and len(raw_t):
                    gt = ticks[grid_edges_idx]
                    j = np.clip(np.searchsorted(raw_t, gt, side="right") - 1, 0, len(raw_t) - 1)
                    shift.extend(gt - raw_t[j])
        rows.append({
            "grid_ms": g, "raw_transitions": total, "grid_transitions": kept,
            "lost_transitions": total - kept,
            "lost_pct": 100.0 * (total - kept) / total,
            "lost_pulses": lost_pulses,
            "median_edge_shift_s": float(np.median(shift)) if shift else np.nan,
            "max_edge_shift_s": float(np.max(shift)) if shift else np.nan,
        })
    return pd.DataFrame(rows)


def shortest_pulses(plcs: dict, n: int = 10) -> pd.DataFrame:
    """Shortest logged ON or OFF periods of boolean signals (event-time)."""
    rows = []
    for name in ("Siemens", "Beckhoff"):
        d = plcs[name]
        t = d.frame["t"].to_numpy()
        for sig in d.signals:
            v = d.frame[sig]
            if v.dtype != bool or v.nunique() < 2:
                continue
            e = edge_times(t, v.to_numpy())
            if len(e) > 1:
                w = np.diff(e)
                k = int(np.argmin(w))
                rows.append({"plc": name, "signal": sig, "shortest_period_s": w[k],
                             "at_s": e[k]})
    return pd.DataFrame(rows).sort_values("shortest_period_s").head(n).reset_index(drop=True)


# --------------------------------------------------------------------------
# Cross-PLC clock check
# --------------------------------------------------------------------------
def cross_plc_sync(plcs: dict, sie_signal: str, bec_signal: str) -> pd.DataFrame:
    """Edges of the documented Siemens->Beckhoff safety signal in both logs.

    ``A201.0`` is documented as "Safetydata from curtain to beckhoff"; the
    Beckhoff input ``VG.safeCurtainShort`` receives it. The same physical edge
    therefore appears in both files.
    """
    s = plcs["Siemens"].frame
    b = plcs["Beckhoff"].frame
    es = edge_times(s["t"], s[sie_signal].to_numpy())
    eb = edge_times(b["t"], b[bec_signal].to_numpy())
    if len(es) != len(eb):
        raise ValueError("handshake signals have different edge counts")
    out = pd.DataFrame({"siemens_edge_s": es, "beckhoff_edge_s": eb})
    out["direction"] = ["falling" if not s[sie_signal].to_numpy()[np.searchsorted(s["t"], x)]
                        else "rising" for x in es]
    out["siemens_step_s"] = [step_before(s["t"].to_numpy(), x) for x in es]
    out["beckhoff_step_s"] = [step_before(b["t"].to_numpy(), x) for x in eb]
    out["difference_s"] = out["beckhoff_edge_s"] - out["siemens_edge_s"]
    # Each logged edge lies in (logged - step, logged]; the true offset
    # interval implied by each pair:
    out["offset_low_s"] = out["difference_s"] - out["beckhoff_step_s"]
    out["offset_high_s"] = out["difference_s"] + out["siemens_step_s"]
    return out


def sync_summary(sync: pd.DataFrame) -> dict:
    """Offset consistent with all handshake edges (interval intersection)."""
    lo = float(sync["offset_low_s"].max())
    hi = float(sync["offset_high_s"].min())
    return {
        "n_edges": int(len(sync)),
        "max_abs_difference_s": float(sync["difference_s"].abs().max()),
        "median_difference_s": float(sync["difference_s"].median()),
        "feasible_offset_low_s": lo,
        "feasible_offset_high_s": hi,
        "zero_offset_feasible": bool(lo <= 0.0 <= hi),
        "differences_change_sign": bool((sync["difference_s"] > 0).any()
                                        and (sync["difference_s"] < 0).any()),
    }


BOOL_MAP = {"True": True, "False": False, True: True, False: False}


@dataclass
class PLCData:
    """Analysis-ready recording of one PLC."""

    name: str
    frame: pd.DataFrame                 # timestamp, t, one column per signal
    signals: list[str]
    raw_shape: tuple[int, int]          # rows, signal columns (as in the file)
    excluded_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped_columns: list[str] = field(default_factory=list)


def _read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", low_memory=False)


def load_plc(path, name: str) -> PLCData:
    """Read one raw PLC file.

    * Drops the empty trailing column produced by the trailing ';' of every
      line (all values NaN, header 'Unnamed: n').
    * Excludes rows that contain missing signal values and records them.
      In this recording that is only the last Siemens row, in which 118 of
      142 signals are empty (an incomplete final logger write).
    * Converts the 'True'/'False' strings to booleans.
    """
    raw = _read_raw(path)
    empty = [c for c in raw.columns if c.startswith("Unnamed") and raw[c].isna().all()]
    raw = raw.drop(columns=empty)
    signals = [c for c in raw.columns if c != "timestamp"]
    raw_shape = (len(raw), len(signals))

    n_missing = raw[signals].isna().sum(axis=1)
    bad = n_missing > 0
    excluded = pd.DataFrame({
        "plc": name,
        "row_index": raw.index[bad],
        "timestamp": raw.loc[bad, "timestamp"].to_numpy(),
        "missing_values": n_missing[bad].to_numpy(),
        "reason": "incomplete row (missing signal values); excluded from analysis",
    })
    df = raw.loc[~bad].copy()
    for c in signals:
        if df[c].dtype == object:
            mapped = df[c].map(BOOL_MAP)
            if mapped.notna().all():
                df[c] = mapped.astype(bool)
            else:
                df[c] = pd.to_numeric(df[c])
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{name}: timestamps are not monotonic")
    return PLCData(name, df.reset_index(drop=True), signals, raw_shape,
                   excluded, empty)


def load_dictionary(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the Siemens and Beckhoff sheets with harmonised column names."""
    sie = pd.read_excel(path, sheet_name="Siemens").rename(columns={
        "Variablenname": "signal", "Datentyp": "data_type",
        "Kommentar": "description", "Station": "station", "Symbol": "symbol"})
    bec = pd.read_excel(path, sheet_name="Beckhoff").rename(columns={
        "Variablenname / Beschreibung": "signal", "Station": "station"})
    for d in (sie, bec):
        d["station"] = d["station"].astype(str).str.strip()
    # 'Robt' is an obvious typo of 'Robot' in the source sheet; it is kept in
    # station_raw and corrected in station.
    bec["station_raw"] = bec["station"]
    bec["station"] = bec["station"].replace({"Robt": "Robot"})
    sie["station_raw"] = sie["station"]
    return sie, bec


def load_all(cfg: dict) -> dict:
    """Load both PLCs, add the common time origin and return a bundle."""
    p = cfg["paths"]
    sie = load_plc(p["raw_siemens"], "Siemens")
    bec = load_plc(p["raw_beckhoff"], "Beckhoff")
    t0 = int(sie.frame["timestamp"].iloc[0])
    for d in (sie, bec):
        d.frame.insert(1, "t", (d.frame["timestamp"] - t0) / 1000.0)
    sie_doc, bec_doc = load_dictionary(p["variable_file"])
    return {"Siemens": sie, "Beckhoff": bec, "t0_ms": t0,
            "doc": {"Siemens": sie_doc, "Beckhoff": bec_doc}}


# Interpretations that the analysis relies on, or that are easy to misread
# from the signal names. Each "measured" claim is shown in notebook 02 and listed in docs/methodology.md.
INTERPRETATIONS: dict[str, tuple[str, str]] = {
    # --- Siemens: safety / stop evidence -----------------------------------
    "A201.0": ("Safety light-curtain status sent to Beckhoff; LOW = protective stop. "
               "Its 7 edges reappear in VG.safeCurtainShort (clock check).", "documented"),
    "M0.3": ("Signal lamp 'red' segment; ON exactly while A201.0 is LOW.", "measured"),
    "E200.0": ("HMI reset button; every curtain stop ends with a press.", "documented"),
    "AW256": ("Data word that drives the Balluff signal lamp. Changes at every oven "
              "blink sequence (normal operation) and during stops. NOT a fault or "
              "motion signal.", "documented"),
    "MW26": ("Documented as 'oven time x 2', but every logged value is a combination "
             "of single bits (0x4000..0xB000) and bit 0x2000 is set exactly during the "
             "curtain stops. Behaves as a status bit-field here; not used as a "
             "process parameter.", "measured"),
    "MD28": ("Documented 'drilling time'. Changes only for the one long drill run "
             "(385.07-390.06 s) and back. Units/encoding not verifiable; not used "
             "as commanded-dwell evidence.", "measured"),
    "MW32": ("Documented 'not in time format'. Same change window as MD28. Read "
             "byte-swapped the normal value is 1000 (would match a 1 s drill time) - "
             "hypothesis only.", "measured"),
    "MW22": ("Drilling-station cycle register. Repeats 1 -> 2|3 -> 0 -> 1 once per "
             "part. State 2|3 lasts 7.4-7.5 s and covers the oven/burn phase; state 0 "
             "~1.2 s; state 1 is the variable remainder (transfer, drill, waiting).",
             "measured"),
    "MW20": ("Counted oven-light blinks (0-2) during each oven run.", "documented"),
    "I200.7": ("Sheet says 'Not used' but ON exactly during the first curtain stop.",
               "measured"),
    "M0.1": ("Sheet says 'Not used' but ON exactly during the second curtain stop.",
             "measured"),
    "M24.4": ("Sheet says 'Not used' but ON during start-up and inside the third "
              "curtain stop.", "measured"),
    "A1.0": ("Drill motor. Normal ON time 0.94-1.05 s.", "documented"),
    "M6.4": ("Keeps drill on / drill timer running (documented); tracks A1.0.",
             "documented"),
    "A0.7": ("Small conveyor of the drilling station; carries the part to light "
             "barrier E1.0 and on to the sorting station.", "documented"),
    "E1.0": ("Light barrier on the small conveyor; LOW while a part is in the beam.",
             "documented"),
    "A0.0": ("Long conveyor of the sorting station.", "documented"),
    # --- Beckhoff ---------------------------------------------------------
    "VG.safeCurtainShort": ("Receives A201.0 from Siemens (same edges within 0.09 s).",
                            "measured"),
    "Visual.start": ("HMI run state; LOW once (468.5-473.5 s) = HMI stop.", "name reading"),
    "Visual.Stoplight": ("HMI stop lamp; HIGH exactly while Visual.start is LOW.",
                         "measured"),
    "WH.desiredPalletX": ("Requested rack column of the next pallet; a change starts a "
                          "pallet cycle.", "name reading"),
    "WH.desiredPalletY": ("Requested rack row of the next pallet.", "name reading"),
    "WH.isEmpty": ("Pallet on the belt is empty: set during robot pick 1, cleared at the end of "
                   "robot put-down 2. Start-up cycle: cleared at the next pallet request (no part "
                   "to return). Final cycle: set from the request on (an empty pallet is fetched).",
                   "measured"),
    "WH.Q1_beltForward": ("Warehouse belt outwards; runs until the pallet reaches a light "
                          "barrier.", "measured"),
    "WH.Q2_beltBackward": ("Warehouse belt inwards; runs until the pallet reaches a light "
                           "barrier.", "measured"),
    "WH.Q3_horRack": ("Horizontal axis towards rack: request (stays ON while the motor "
                      "output of the pair is OFF during stops).", "measured"),
    "WH.Q3_horizontal_towards_rack": ("Horizontal axis towards rack: motor output "
                                      "(encoder moves only while it is ON).", "measured"),
    "WH.B1_encoder_horizontal_impuls1": ("Horizontal-axis encoder; pulses while the axis "
                                         "moves. Pulse counts are unreliable at ~11 Hz "
                                         "logging; only presence/absence is used.",
                                         "measured"),
    "VG.Q7_compressorON": ("Robot vacuum compressor; ON for one pick-and-place move. "
                           "Two moves per pallet cycle.", "measured"),
}


def _siemens_role(name: str, dtype: str, desc: str) -> tuple[str, str]:
    d = str(desc).strip().lower()
    if d == "not used":
        return "Unused or unknown", "sheet says 'Not used'"
    if name.startswith(("A200", "A201")):
        return "Interface", "A-output with station 'Beckhoff' (data to Beckhoff)"
    if name.startswith("AW"):
        return "Command", "output word (indicator lamp)"
    if name.startswith("A"):
        return "Command", "S7 output (A = Ausgang)"
    if name.startswith(("E200", "EW256", "EW258")):
        return "Interface", "input from the Beckhoff HMI"
    if name.startswith(("E", "I", "EW")):
        return "Sensor", "S7 input (E = Eingang)"
    if name == "MW16":
        return "Sensor", "analog colour-sensor value copied to memory"
    if name in ("MW26", "MD28", "MW32"):
        return "Parameter", "documented time/scaling value"
    if name.startswith(("MW", "MD", "C")):
        return "State", "memory word / counter"
    if name.startswith("M"):
        return "State", "S7 memory bit (M = Merker)"
    return "Unused or unknown", "no rule"


def _beckhoff_role(name: str) -> tuple[str, str]:
    if ".Q" in name:
        return "Command", "IEC 61131-3 output (Q)"
    if re.search(r"\.(I\d|B\d|A\d)", name):
        return "Sensor", "input (I), encoder (B) or trail sensor (A)"
    if "safeCurtain" in name:
        return "Interface", "safety data received from Siemens"
    if "desiredPallet" in name:
        return "Parameter", "requested rack position"
    if name.startswith("Visual."):
        return "State", "HMI state (station Visualization)"
    if name in ("WH.isEmpty", "VG.atHome", "VG.error_robot"):
        return "State", "status flag"
    return "Unused or unknown", "no rule"


def _behaviour(series: pd.Series, t: np.ndarray) -> dict:
    v = series.to_numpy()
    changes = np.flatnonzero(v[1:] != v[:-1]) + 1
    out = {
        "n_changes": int(len(changes)),
        "constant": bool(len(changes) == 0),
        "first_change_s": float(t[changes[0]]) if len(changes) else np.nan,
        "last_change_s": float(t[changes[-1]]) if len(changes) else np.nan,
        "n_distinct": int(pd.Series(v).nunique()),
        "frac_true": float(v.mean()) if series.dtype == bool else np.nan,
        "min_value": float(np.min(v.astype(float))),
        "max_value": float(np.max(v.astype(float))),
    }
    if series.dtype == bool and len(changes):
        e = t[changes]
        out["shortest_period_s"] = float(np.diff(e).min()) if len(e) > 1 else np.nan
    else:
        out["shortest_period_s"] = np.nan
    return out


def build_dictionary(data: dict) -> pd.DataFrame:
    rows = []
    sie_doc = data["doc"]["Siemens"].set_index("signal")
    bec_doc = data["doc"]["Beckhoff"].set_index("signal")
    for plc, doc in (("Siemens", sie_doc), ("Beckhoff", bec_doc)):
        d = data[plc]
        t = d.frame["t"].to_numpy()
        for sig in d.signals:
            meta = doc.loc[sig] if sig in doc.index else None
            in_doc = meta is not None
            if plc == "Siemens":
                dtype = meta["data_type"] if in_doc else ""
                desc = meta["description"] if in_doc else ""
                role, basis = _siemens_role(sig, dtype, desc)
                symbol = meta["symbol"] if in_doc else ""
            else:
                dtype = str(d.frame[sig].dtype)
                desc = ""
                symbol = ""
                role, basis = _beckhoff_role(sig)
            beh = _behaviour(d.frame[sig], t)
            interp, status = INTERPRETATIONS.get(sig, ("", ""))
            if not interp:
                if plc == "Siemens" and desc and str(desc).strip().lower() != "not used":
                    interp, status = str(desc).strip(), "documented"
                elif plc == "Beckhoff":
                    interp, status = "meaning from the signal name only", "name reading"
                else:
                    interp, status = "", "unknown"
            if beh["constant"]:
                note = "constant over the whole recording; carries no information here"
            elif role == "Unused or unknown":
                note = "documented as unused but changes; treated as unknown"
            else:
                note = ""
            rows.append({
                "plc": plc, "signal": sig, "in_dictionary": in_doc,
                "station": meta["station"] if in_doc else "",
                "station_raw": meta["station_raw"] if in_doc else "",
                "data_type": dtype, "description": desc, "symbol": symbol,
                "role": role, "role_basis": basis, **beh,
                "interpretation": interp, "evidence_status": status, "note": note,
            })
    return pd.DataFrame(rows)


MW22_PHASE = {0: "transition (vacuum arm down/up)", 1: "transfer, turntable, drill, conveyor, waiting",
              2: "oven phase", 3: "oven phase (variant)"}


def recording_end(data: dict) -> float:
    return float(max(data["Siemens"].frame["t"].iloc[-1], data["Beckhoff"].frame["t"].iloc[-1]))


def production_start(data: dict, cfg: dict) -> float:
    """First press of the documented HMI reset button (E200.0)."""
    s = data["Siemens"].frame
    presses = on_intervals(s["t"], s[cfg["phases"]["production_start_signal"]])
    return presses[0][0]


def hmi_reset_times(data: dict, cfg: dict) -> list[float]:
    s = data["Siemens"].frame
    return [a for a, _ in on_intervals(s["t"], s[cfg["phases"]["production_start_signal"]])]


def stop_windows(data: dict, cfg: dict, sync_tol_s: float) -> pd.DataFrame:
    """Protective (curtain) stops and HMI stops, with their confirmations."""
    s, b = data["Siemens"].frame, data["Beckhoff"].frame
    ph = cfg["phases"]
    p0 = production_start(data, cfg)
    presses = hmi_reset_times(data, cfg)
    lamp = on_intervals(s["t"], s["M0.3"])
    sr_long = on_intervals(s["t"], s["M24.3"])
    bec_curtain = on_intervals(b["t"], ~b[ph["curtain_signal_beckhoff"]].astype(bool))
    rows = []

    def matched(a, e, windows):
        return any(abs(x - a) <= sync_tol_s and abs(y - e) <= sync_tol_s for x, y in windows)

    for a, e in on_intervals(s["t"], ~s[ph["curtain_signal"]].astype(bool)):
        reset = min(presses, key=lambda r: abs(r - e))
        rows.append({
            "kind": "protective stop (light curtain)" if a >= p0 else "start-up (curtain not yet reset)",
            "start_s": a, "end_s": e, "duration_s": e - a,
            "source_signal": f"Siemens {ph['curtain_signal']} LOW",
            "confirm_red_lamp_M0.3": matched(a, e, lamp),
            "confirm_long_curtain_SR_M24.3": matched(a, e, sr_long),
            "confirm_beckhoff_receiver": matched(a, e, bec_curtain),
            "ended_by_reset_press": abs(reset - e) <= sync_tol_s,
            "reset_press_s": reset,
        })
    stoplight = on_intervals(b["t"], b["Visual.Stoplight"])
    for a, e in on_intervals(b["t"], ~b[ph["hmi_run_signal"]].astype(bool)):
        rows.append({
            "kind": "HMI stop", "start_s": a, "end_s": e, "duration_s": e - a,
            "source_signal": f"Beckhoff {ph['hmi_run_signal']} LOW",
            "confirm_red_lamp_M0.3": False, "confirm_long_curtain_SR_M24.3": False,
            "confirm_beckhoff_receiver": matched(a, e, stoplight),
            "ended_by_reset_press": False, "reset_press_s": np.nan,
        })
    out = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)
    out.insert(0, "stop_id", [f"S{i}" for i in range(len(out))])
    return out


def mw22_phases(data: dict, cfg: dict) -> pd.DataFrame:
    s = data["Siemens"].frame
    reg = cfg["phases"]["drilling_cycle_register"]
    v = s[reg].to_numpy()
    t = s["t"].to_numpy()
    change = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
    starts = t[change]
    ends = np.r_[starts[1:], t[-1]]
    out = pd.DataFrame({"value": v[change].astype(int), "start_s": starts, "end_s": ends})
    out["duration_s"] = out["end_s"] - out["start_s"]
    out["open_at_end"] = False
    out.loc[out.index[-1], "open_at_end"] = True
    out["open_at_start"] = False
    out.loc[out.index[0], "open_at_start"] = True
    out["phase"] = out["value"].map(MW22_PHASE)
    return out


def drilling_cycles(phases: pd.DataFrame) -> pd.DataFrame:
    """One row per oven phase (MW22 in {2,3}); part-to-part interval."""
    ov = phases[phases["value"].isin([2, 3])].copy()
    # merge the 2->3 sequence of the first part into one oven phase
    rows = []
    for _, r in ov.iterrows():
        if rows and abs(r["start_s"] - rows[-1]["end_s"]) < 1e-9:
            rows[-1]["end_s"] = r["end_s"]
            rows[-1]["values"] += f"{r['value']}"
        else:
            rows.append({"start_s": r["start_s"], "end_s": r["end_s"], "values": f"{r['value']}"})
    out = pd.DataFrame(rows)
    out["oven_phase_s"] = out["end_s"] - out["start_s"]
    out["part_interval_s"] = out["start_s"].diff()
    out.insert(0, "part", range(1, len(out) + 1))
    return out


def pallet_cycles(data: dict, cfg: dict) -> pd.DataFrame:
    """A pallet cycle starts at every change of the requested rack position."""
    b = data["Beckhoff"].frame
    req = cfg["phases"]["pallet_request_signals"]
    new = b[req].diff().ne(0).any(axis=1)
    cyc = b.loc[new, ["t"] + req].reset_index(drop=True)
    cyc.columns = ["start_s", "rack_x", "rack_y"]
    cyc.insert(0, "cycle", range(1, len(cyc) + 1))
    cyc["end_s"] = np.r_[cyc["start_s"].to_numpy()[1:], recording_end(data)]
    cyc["duration_s"] = cyc["end_s"] - cyc["start_s"]
    p0 = production_start(data, cfg)
    last = cyc["cycle"].max()
    s = data["Siemens"].frame
    oven_in = [a for a, _ in on_intervals(s["t"], s["A1.1"])]
    robot = [a for a, _ in on_intervals(b["t"], b["VG.Q7_compressorON"])]
    belt_f = [a for a, _ in on_intervals(b["t"], b["WH.Q1_beltForward"])]
    belt_b = [a for a, _ in on_intervals(b["t"], b["WH.Q2_beltBackward"])]

    def count(times, a, e):
        return int(sum(a <= x < e for x in times))

    cyc["robot_moves"] = [count(robot, a, e) for a, e in zip(cyc["start_s"], cyc["end_s"])]
    cyc["belt_forward"] = [count(belt_f, a, e) for a, e in zip(cyc["start_s"], cyc["end_s"])]
    cyc["belt_backward"] = [count(belt_b, a, e) for a, e in zip(cyc["start_s"], cyc["end_s"])]
    cyc["oven_runs"] = [count(oven_in, a, e) for a, e in zip(cyc["start_s"], cyc["end_s"])]
    roles = []
    for _, c in cyc.iterrows():
        if c["start_s"] < p0:
            roles.append("start-up (contains homing; first part only)")
        elif c["cycle"] == last:
            roles.append("final (truncated by end of recording; no new part)")
        else:
            roles.append("complete")
    cyc["role"] = roles
    cyc["complete"] = cyc["role"] == "complete"
    return cyc


def actuator_lists(data: dict) -> dict:
    """Actuator outputs used for activation analysis (documented / IEC naming)."""
    sdoc = data["doc"]["Siemens"]
    sie = sdoc.loc[sdoc["signal"].str.match(r"^A\d") &
                   sdoc["station"].isin(["Sorting station", "Drilling station"]), "signal"].tolist()
    bec = [c for c in data["Beckhoff"].signals if ".Q" in c]
    return {"Siemens": sie, "Beckhoff": bec}


def station_of(data: dict, plc: str, signal: str) -> str:
    doc = data["doc"][plc].set_index("signal")
    return doc.loc[signal, "station"] if signal in doc.index else ""


def activations(data: dict, cfg: dict, cycles: pd.DataFrame) -> pd.DataFrame:
    """One row per ON-period of each actuator output (station activity)."""
    p0 = production_start(data, cfg)
    acts = actuator_lists(data)
    frames = []
    for plc, signals in acts.items():
        d = data[plc].frame
        for sig in signals:
            it = interval_table(d["t"], d[sig])
            if it.empty:
                continue
            it.insert(0, "signal", sig)
            it.insert(0, "plc", plc)
            it["station"] = station_of(data, plc, sig)
            frames.append(it)
    out = pd.concat(frames, ignore_index=True)
    out["in_production"] = out["start"] >= p0
    starts = cycles["start_s"].to_numpy()
    out["cycle"] = np.searchsorted(starts, out["start"].to_numpy(), side="right")
    out["cycle"] = out["cycle"].clip(lower=1)
    return out.sort_values(["start", "plc", "signal"]).reset_index(drop=True)


def robot_moves(data: dict, cycles: pd.DataFrame) -> pd.DataFrame:
    """Robot pick-and-place moves; move type from the lane barrier it clears.

    A sorting-lane barrier (E0.3 white, E0.4 red, E0.5 blue) is LOW while a
    finished part waits in the lane. If it goes HIGH during a move, that move
    took the part from the lane (type 'lane -> pallet'); otherwise the move is
    'pallet -> oven'.
    """
    b, s = data["Beckhoff"].frame, data["Siemens"].frame
    lanes = {"E0.3": "white", "E0.4": "red", "E0.5": "blue"}
    rises = {c: [a for _, a in on_intervals(s["t"], ~s[c].astype(bool))] for c in lanes}
    rows = []
    for a, e in on_intervals(b["t"], b["VG.Q7_compressorON"]):
        lane = [lanes[c] for c, r in rises.items() if any(a <= x <= e + 0.5 for x in r)]
        rows.append({"start_s": a, "end_s": e, "duration_s": e - a,
                     "move_type": "lane -> pallet" if lane else "pallet -> oven",
                     "lane": lane[0] if lane else ""})
    out = pd.DataFrame(rows)
    out["cycle"] = np.searchsorted(cycles["start_s"].to_numpy(), out["start_s"].to_numpy(), side="right")
    return out


# One logging step of either PLC: edges of the same stop can be logged this
# far apart on the two PLCs (measured offset <= 0.09 s, median steps 0.08/0.09 s).
STOP_TOL_S = 0.25


def cut_by_stop(s: float, e: float, stops: pd.DataFrame) -> bool:
    """The run ended at (or contains) the start of a stop window."""
    return bool(((stops["start_s"] > s) & (stops["start_s"] <= e + STOP_TOL_S)
                 & (stops["start_s"] >= e - STOP_TOL_S)).any())


def resumed_after_stop(s: float, stops: pd.DataFrame) -> bool:
    return bool((np.abs(stops["end_s"] - s) <= STOP_TOL_S).any())


def mark_stop_cuts(acts: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Flag activations truncated by a stop, or restarted right after one."""
    a = acts.copy()
    a["cut_by_stop"] = [cut_by_stop(s, e, stops) for s, e in zip(a["start"], a["end"])]
    a["resumed_after_stop"] = [resumed_after_stop(s, stops) for s in a["start"]]
    return a


def stop_halt_check(data: dict, actuators: dict, stops: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Actuator switching inside each stop window (edges more than ``tol``
    after the stop start and before its end). Zero = the line was halted."""
    rows = []
    for _, st in stops.iterrows():
        n, sigs = 0, set()
        for plc, names in actuators.items():
            d = data[plc].frame
            w = d[(d["t"] > st["start_s"] + tol) & (d["t"] < st["end_s"] - tol)]
            prev = d[d["t"] <= st["start_s"] + tol].tail(1)
            block = pd.concat([prev, w])[names].astype(int)
            ch = block.diff().abs().iloc[1:]
            n += int(ch.to_numpy().sum())
            sigs |= set(ch.columns[ch.sum() > 0])
        rows.append({"stop_id": st["stop_id"], "actuator_edges_inside": n,
                     "signals_switching_inside": " ".join(sorted(sigs))})
    return pd.DataFrame(rows)
