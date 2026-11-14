"""Figures of the analysis (the same functions are copied into the notebooks).

One style for every figure: 12 x 5.4 in, 200 dpi, fonts >= 12 pt, a title that states the
takeaway, direct value labels, units on the axes. Colour is never the only carrier of meaning:
events are always labelled E1-E8.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

COLORS = {"stop": "#D64545", "overrun": "#F0A030", "normal": "#9AA5B1", "accent": "#1F7A8C",
          "navy": "#1F2A5A", "light": "#E6E9ED", "muted": "#5B6675"}
FIGSIZE = (12, 5.4)
DPI = 200
STYLE = {
    "figure.figsize": FIGSIZE, "figure.dpi": 100, "savefig.dpi": DPI, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 13, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 13, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5B6675", "axes.labelcolor": "#1F2A5A", "text.color": "#1F2A5A",
    "xtick.color": "#1F2A5A", "ytick.color": "#1F2A5A", "legend.frameon": False,
}
# Plain names of the signals shown in the figures
PLAIN = {
    "WH.Q3_horRack": "Warehouse axis to rack (request)",
    "WH.Q3_horizontal_towards_rack": "Warehouse axis to rack (motor)",
    "WH.Q1_beltForward": "Warehouse belt outwards",
    "WH.Q2_beltBackward": "Warehouse belt inwards",
    "WH.Q7_cantForw": "Cantilever forward (request)",
    "WH.Q7_cantilever_forward": "Cantilever forward (motor)",
    "WH.Q8_cantBack": "Cantilever backward (request)",
    "WH.Q8_cantilever_backward": "Cantilever backward (motor)",
    "WH.I2_Light_barrier_inside": "Belt light barrier inside",
    "WH.I3_Light_barrier_outside": "Belt light barrier outside",
    "A1.7": "Drilling vacuum suction",
    "A1.0": "Drill",
    "A0.7": "Small conveyor",
    "E1.0": "Small-conveyor barrier (part)",
    "A201.0": "Safety curtain stop",
    "Visual.start": "HMI stop",
}
MOTOR_OF = {"WH.Q3_horRack": "WH.Q3_horizontal_towards_rack", "WH.Q7_cantForw": "WH.Q7_cantilever_forward",
            "WH.Q8_cantBack": "WH.Q8_cantilever_backward"}


def apply_style():
    plt.rcParams.update(STYLE)


def _title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=15, fontweight="bold")


def _save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _on_spans(t, x):
    """(start, length) of every ON period of boolean x on time axis t."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=bool).astype(int)
    change = np.diff(np.concatenate(([0], x, [0])))
    starts = np.flatnonzero(change == 1)
    ends = np.minimum(np.flatnonzero(change == -1), len(t) - 1)
    return [(float(t[a]), float(t[b] - t[a])) for a, b in zip(starts, ends)]


def _event_label(event, eq):
    """'E6' + the equipment excess of each overrun of that event (never added)."""
    vals = eq.loc[eq["event"] == event, "excess_s"]
    return f"{event}\n" + " / ".join(f"{v:.1f}" for v in vals) + " s"


def _note_panel(ax, text):
    ax.axis("off")
    ax.text(0.0, 0.5, text, fontsize=12, va="center", ha="left", transform=ax.transAxes, linespacing=1.5,
            bbox={"fc": COLORS["light"], "ec": COLORS["muted"], "boxstyle": "round,pad=0.6"})


# --------------------------------------------------------------------------
# F8 time base
# --------------------------------------------------------------------------
def time_base(data, gaps, limit, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
    med = {}
    for name, col in (("Siemens", COLORS["accent"]), ("Beckhoff", COLORS["overrun"])):
        d = np.diff(data[name].frame["t"].to_numpy())
        med[name] = np.median(d)
        a.hist(d, bins=np.logspace(np.log10(0.06), np.log10(2.2), 50), color=col, alpha=0.75,
               label=f"{name} (median {1000 * med[name]:.0f} ms)")
    a.axvline(limit, color=COLORS["stop"], lw=2)
    a.text(limit * 0.92, a.get_ylim()[1] * 0.55, f"detection limit\n{limit:.2f} s", color=COLORS["stop"],
           ha="right", fontsize=12)
    a.set_xscale("log")
    a.set_xlabel("logging step (s, log scale)")
    a.set_ylabel("number of steps")
    a.legend(loc="center")
    a.set_title("A  Logging step distribution")
    for name, col, m in (("Siemens", COLORS["accent"], "o"), ("Beckhoff", COLORS["overrun"], "s")):
        g = gaps[gaps["plc"] == name]
        b.scatter(g["gap_start_s"], g["gap_s"], color=col, marker=m, s=70, label=name, zorder=3)
    b.axhline(limit, color=COLORS["stop"], lw=2)
    top = gaps.loc[gaps["gap_s"].idxmax()]
    b.annotate(f"largest gap {top['gap_s']:.2f} s", (top["gap_start_s"], top["gap_s"]),
               xytext=(top["gap_start_s"] + 40, top["gap_s"] + 0.25), fontsize=12,
               arrowprops={"arrowstyle": "-", "color": COLORS["muted"]})
    b.set_ylim(0.4, limit + 0.6)
    b.set_xlabel("time of the gap (s)")
    b.set_ylabel("gap length (s)")
    b.legend(loc="upper left", ncol=2)
    b.set_title("B  Logging gaps > 0.5 s")
    _title(fig, f"Irregular logging (median {1000 * med['Siemens']:.0f} / {1000 * med['Beckhoff']:.0f} ms), gaps up "
                f"to {limit:.2f} s: smaller overruns are not counted")
    _save(fig, path)


# --------------------------------------------------------------------------
# F1 event timeline
# --------------------------------------------------------------------------
def event_timeline(events, stops, eq, p0, end, path):
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    y0, y1 = 0.0, 1.0
    ax.axvspan(0, p0, color=COLORS["light"], zorder=0)
    ax.text(p0 + 3, 0.5, "start-up\n(excluded)", ha="left", va="center", fontsize=12, color=COLORS["muted"])
    for s in stops[stops["start_s"] >= p0].itertuples():
        ax.fill_between([s.start_s, s.end_s], y0, y1, color=COLORS["stop"], lw=0, zorder=2)
    over = events[events["stop_mechanism"].str.startswith("step overrun")]
    for e in over.itertuples():
        ax.fill_between([e.overrun_start_s, e.overrun_end_s], y0, y1, color=COLORS["overrun"], lw=0, zorder=2)
    last, top = {}, 0
    for e in events.sort_values("overrun_start_s").itertuples():
        x = (e.overrun_start_s + e.overrun_end_s) / 2
        lvl = 0
        while lvl in last and x - last[lvl] < 48:
            lvl += 1
        last[lvl] = x
        top = max(top, lvl)
        ax.annotate(_event_label(e.event, eq), (x, y1), xytext=(x, y1 + 0.35 + 0.95 * lvl), ha="center",
                    va="bottom", fontsize=12, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "lw": 1})
    ax.set_ylim(-0.1, y1 + 0.35 + 0.95 * (top + 1) + 0.2)
    ax.set_xlim(0, end)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time since the start of the recording (s); labels = equipment excess per overrun")
    ax.legend(handles=[Patch(color=COLORS["stop"], label="line-wide stop (curtain or HMI)"),
                       Patch(color=COLORS["overrun"], label="step overrun"),
                       Patch(color=COLORS["light"], label="start-up")], loc="upper left",
              bbox_to_anchor=(0.08, 1.0), ncol=1)
    n_stop = int(len(stops[stops["start_s"] >= p0]))
    _title(fig, f"{len(events)} events in production ({p0:.2f}-{end:.2f} s): {n_stop} line-wide stops (red), "
                f"{len(over)} step overruns (amber)")
    _save(fig, path)


# --------------------------------------------------------------------------
# F2 attribution
# --------------------------------------------------------------------------
def attribution_bars(eq, events, path):
    mech = events.set_index("event")["stop_mechanism"]
    rows = eq.copy()
    rows["is_stop"] = ~rows["event"].map(mech).str.startswith("step overrun")
    rows = rows.sort_values(["station", "event", "order_in_incident"], ascending=[False, True, True])
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    y, ticks, labels, group_y = 0, [], [], []
    for station, g in rows.groupby("station", sort=False):
        group_y.append((station, y))
        for r in g.itertuples():
            ax.barh(y, r.excess_s, color=COLORS["stop"] if r.is_stop else COLORS["overrun"], height=0.7)
            ax.text(r.excess_s + 0.15, y, f"{r.excess_s:.1f} s", va="center", fontsize=12)
            ticks.append(y)
            labels.append(f"{r.event}  {PLAIN.get(r.signal, r.signal)}")
            y += 1
        y += 0.8
    for station, gy in group_y:
        ax.text(-0.02, gy - 0.75, station, transform=ax.get_yaxis_transform(), ha="right", fontsize=13,
                fontweight="bold", color=COLORS["accent"])
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, rows["excess_s"].max() * 1.15)
    ax.set_xlabel("equipment excess over the normal (median) duration (s) - equipment seconds, not additive")
    ax.legend(handles=[Patch(color=COLORS["stop"], label="event with a line-wide stop"),
                       Patch(color=COLORS["overrun"], label="step overrun, no stop signal")], loc="lower right")
    n_st = rows.groupby("station")["event"].size().sort_values(ascending=False)
    _title(fig, "Where steps took too long: " + " and ".join(f"{n} overruns at the {s.lower()}"
                                                            for s, n in n_st.items()))
    _save(fig, path)


# --------------------------------------------------------------------------
# F3 cycle effect
# --------------------------------------------------------------------------
def cycle_effect_figure(cyc, total, path):
    c = cyc[cyc["reference_s"].notna()].sort_values("cycle")
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    x = c["cycle"].to_numpy()
    base = np.minimum(c["observed_s"], c["reference_s"])
    exc = np.clip(c["cycle_excess_s"], 0, None)
    ax.bar(x, base, color=COLORS["normal"], width=0.62, label="normal reference (position-adjusted)")
    ax.bar(x, exc, bottom=base, color=COLORS["stop"], width=0.62, label="excess over the reference")
    for r in c.itertuples():
        if r.cycle_excess_s < 0:
            ax.hlines(r.reference_s, r.cycle - 0.31, r.cycle + 0.31, color=COLORS["navy"], lw=2)
        txt = f"{r.observed_s:.1f} s"
        if r.events:
            txt = f"{r.events}\n+{r.cycle_excess_s:.1f} s\n" + txt
        ax.text(r.cycle, max(r.observed_s, r.reference_s) + 1, txt, ha="center", va="bottom", fontsize=12,
                fontweight="bold" if r.events else "normal")
    ax.set_xticks(x)
    ax.set_xlabel("pallet cycle (complete cycles); dark line = reference where the cycle was faster")
    ax.set_ylabel("cycle time (s)")
    ax.set_ylim(0, c["observed_s"].max() * 1.42)
    ax.legend(loc="upper left", ncol=2)
    clean = c.loc[c["events"] == "", "cycle_excess_s"].abs().max()
    _title(fig, f"Cycles {', '.join(map(str, total['cycles']))} took {total['excess_s']:.1f} s longer than their "
                f"normal reference ({total['excess_pct']:.1f} %); clean cycles within {clean:.1f} s")
    _save(fig, path)


# --------------------------------------------------------------------------
# F4 evidence figures
# --------------------------------------------------------------------------
def _signal_rows(ax, data, rows, a, b, highlight):
    for y, (plc, sig, label, invert) in enumerate(rows):
        d = data[plc].frame
        w = d[(d["t"] >= a - 20) & (d["t"] <= b + 20)]
        if sig == "encoder":
            for enc in ("WH.B1_encoder_horizontal_impuls1", "WH.B2_encoder_horizontal_impuls2"):
                v = w[enc].to_numpy()
                e = w["t"].to_numpy()[1:][v[1:] != v[:-1]]
                ax.vlines(e[(e >= a) & (e <= b)], y - 0.3, y + 0.3, color=COLORS["navy"], lw=1)
            continue
        x = ~w[sig].astype(bool) if invert else w[sig].astype(bool)
        col = COLORS["stop"] if sig == "A201.0" else COLORS["overrun"] if sig in highlight else COLORS["accent"]
        ax.broken_barh([s for s in _on_spans(w["t"], x) if s[0] + s[1] >= a and s[0] <= b], (y - 0.3, 0.6),
                       facecolors=col)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(a, b)
    ax.set_xlabel("time since the start of the recording (s)")


def evidence_stop(data, events, stops, paired, path, event="E7"):
    ev = events.set_index("event").loc[event]
    st = stops.set_index("stop_id").loc[ev["stop_id"]]
    pw = paired[(paired["start_s"] <= ev["end_s"]) & (paired["end_s"] >= ev["start_s"])
                & (paired["request"] == "WH.Q3_horRack")].iloc[0]
    a, b = ev["start_s"] - 3, ev["end_s"] + 3
    rows = [("Beckhoff", "WH.Q3_horRack", "Axis request", False),
            ("Beckhoff", "WH.Q3_horizontal_towards_rack", "Axis motor output", False),
            ("Beckhoff", "encoder", "Axis encoder pulses", False),
            ("Siemens", "A201.0", "Safety curtain stop", True)]
    fig, (ax, nb) = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained", width_ratios=[2.6, 1])
    ax.axvspan(st["start_s"], st["end_s"], color=COLORS["stop"], alpha=0.12, zorder=0)
    _signal_rows(ax, data, rows, a, b, {"WH.Q3_horRack"})
    _note_panel(nb, f"Expected:\nmotor follows the request\nwithin 0.26 s\n\nObserved:\nmotor OFF for "
                    f"{pw['duration_s']:.2f} s\nwhile requested,\nencoder silent,\ninside curtain stop "
                    f"{ev['stop_id']}\n({st['duration_s']:.2f} s, shaded)")
    _title(fig, f"{event}: the axis was requested, but its motor stayed off for {pw['duration_s']:.1f} s "
                f"during curtain stop {ev['stop_id']}")
    _save(fig, path)


def evidence_overrun(data, belt, flags, path, event="E3"):
    f = flags[flags["event"] == event].iloc[0]
    sig = f["signal"]
    run = belt[(belt["signal"] == sig) & (np.isclose(belt["start"], f["start"]))].iloc[0]
    keys = set(zip(flags["signal"], flags["start"].round(6)))
    flagged = np.array([(s, round(t, 6)) in keys for s, t in zip(belt["signal"], belt["start"])])
    normal = belt[(belt["signal"] == sig).to_numpy() & ~flagged]
    a, b = f["start"] - 3, f["end"] + 3
    rows = [("Beckhoff", sig, PLAIN.get(sig, sig) + " (output)", False),
            ("Beckhoff", "WH.I2_Light_barrier_inside", PLAIN["WH.I2_Light_barrier_inside"], False),
            ("Beckhoff", "WH.I3_Light_barrier_outside", PLAIN["WH.I3_Light_barrier_outside"], False)]
    fig, (ax, nb) = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained", width_ratios=[2.6, 1])
    ax.axvspan(f["overrun_start_s"], f["overrun_end_s"], color=COLORS["overrun"], alpha=0.15, zorder=0)
    _signal_rows(ax, data, rows, a, b, {sig})
    _note_panel(nb, f"Expected:\npallet at the barrier after\n{normal['barrier_reached_after_s'].min():.2f}-"
                    f"{normal['barrier_reached_after_s'].max():.2f} s (normal runs)\n\nObserved:\n"
                    f"{run['barrier_reached_after_s']:.2f} s; belt ON {run['duration']:.2f} s\nno stop signal\n"
                    "(shaded = beyond the\nnormal duration)")
    _title(fig, f"{event}: the belt ran {run['duration']:.1f} s instead of about {normal['duration'].median():.1f} s "
                "before the pallet reached its barrier")
    _save(fig, path)


# --------------------------------------------------------------------------
# F5 robustness
# --------------------------------------------------------------------------
def robustness(sweep, grid, n_events, limit, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
    s = sweep[sweep["merge_gap_s"] == 0.0]
    both = s.groupby("min_excess_s")["events"].apply(lambda v: (v == n_events).all())
    plateau = both[both].index
    a.axvspan(plateau.min(), plateau.max(), color=COLORS["accent"], alpha=0.15)
    a.text(plateau.max() + 0.1, n_events + 1.2, f"{n_events} events for\n{plateau.min():.1f}-{plateau.max():.1f} s",
           ha="left", fontsize=12, color=COLORS["accent"], fontweight="bold")
    for k, col, m in ((1.5, COLORS["overrun"], "s"), (3.0, COLORS["navy"], "o")):
        p = s[s["tukey_k"] == k]
        a.plot(p["min_excess_s"], p["events"], marker=m, color=col, lw=2, ms=8, label=f"Tukey k = {k:g}")
    a.axvline(limit, color=COLORS["stop"], ls="--", lw=1.5)
    a.text(limit + 0.08, s["events"].max() + 0.5, f"used: {limit:.2f} s", color=COLORS["stop"], fontsize=12)
    a.set_ylim(0, s["events"].max() + 2)
    a.set_xlabel("minimum excess over the median (s)")
    a.set_ylabel("events found")
    a.legend(loc="lower right")
    a.set_title("A  Threshold sweep")
    g = grid[grid["status"] == "ok"].copy()
    missed = g["raw_events_missed"].fillna("").astype(str).str.split().str.len()
    g["found"] = n_events - missed
    xs = np.arange(len(g))
    b.bar(xs - 0.2, g["found"], width=0.4, color=COLORS["accent"], label=f"events found (of {n_events})")
    b.bar(xs + 0.2, g["extra_events"], width=0.4, color=COLORS["stop"], label="false extra events")
    for x, f, e in zip(xs, g["found"], g["extra_events"]):
        b.text(x - 0.2, f + 0.15, str(int(f)), ha="center", fontsize=12)
        b.text(x + 0.2, e + 0.15, str(int(e)), ha="center", fontsize=12)
    b.set_xticks(xs)
    b.set_xticklabels([f"{int(v)}" for v in g["grid_ms"]])
    b.set_xlabel("regular time grid (ms)")
    b.set_ylabel("events")
    b.set_ylim(0, n_events + 3.5)
    b.legend(loc="upper left", ncol=2)
    b.set_title("B  Time-grid test")
    ok_to = int(g.loc[(g["found"] == n_events) & (g["extra_events"] == 0), "grid_ms"].max())
    _title(fig, f"The {n_events} events are robust: same count for thresholds {plateau.min():.1f}-"
                f"{plateau.max():.1f} s and for grids up to {ok_to} ms")
    _save(fig, path)


# --------------------------------------------------------------------------
# F6 method comparison
# --------------------------------------------------------------------------
def method_comparison_figure(ml_q, n_flags, path):
    def val(method, metric):
        m = ml_q[(ml_q["method"] == method) & ml_q["metric"].str.startswith(metric)]
        return int(float(m["value"].iloc[0]))
    rows = [("Rule (Model 1)", n_flags, 0, 0)]
    for lab, m in (("Isolation Forest A", "Isolation Forest A_duration_aware"),
                   ("Isolation Forest B", "Isolation Forest B_context_only")):
        rows.append((lab, val(m, "top-20 items that are rule overruns"), val(m, "additional candidates (top-20"),
                     val(m, "additional candidates longer")))
    rows.append(("K-Means", val("K-Means", "rule overruns in the cluster"), val("K-Means", "additional candidates (cluster"),
                 val("K-Means", "additional candidates longer")))
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    xs = np.arange(len(rows))
    bars = [(-0.27, 1, COLORS["accent"], f"rule-flagged overruns recovered (of {n_flags})"),
            (0.0, 2, COLORS["normal"], "extra candidates checked"),
            (0.27, 3, COLORS["stop"], "extra candidates confirmed as loss")]
    for off, i, col, lab in bars:
        vals = [r[i] for r in rows]
        ax.bar(xs + off, vals, width=0.27, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x + off, v + 0.3, str(v), ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, max(r[2] for r in rows) + 4)
    ax.set_ylabel("activations")
    ax.legend(loc="upper left", ncol=3)
    _title(fig, f"Machine learning re-finds up to {max(r[1] for r in rows[1:])} of {n_flags} rule overruns but "
                f"confirms {sum(r[3] for r in rows)} additional losses")
    _save(fig, path)


# --------------------------------------------------------------------------
# F7 performance bounds
# --------------------------------------------------------------------------
def performance_bounds(perf, path):
    pos = perf[perf["method"] == "position-adjusted"].set_index("reference_basis")["performance"] * 100
    sing = perf[perf["method"] == "single reference"].set_index("reference_basis")["performance"] * 100
    primary = pos[pos.index.str.contains("primary")].iloc[0]
    median = sing[sing.index.str.startswith("median")].iloc[0]
    fastest = sing[sing.index.str.startswith("fastest")].iloc[0]
    b_lo = sing[sing.index.str.contains("2.5%")].iloc[0]
    b_hi = sing[sing.index.str.contains("97.5%")].iloc[0]
    rows = [("vs position-adjusted normal\nreference (primary)", primary, pos.min(), pos.max(), COLORS["accent"],
             "range over the clean-cycle fixed part"),
            ("single reference:\nmedian clean cycle", median, b_lo, b_hi, COLORS["normal"], "bootstrap of the median"),
            ("single reference:\nfastest clean cycle", fastest, None, None, COLORS["overrun"],
             "upper bound for one ideal time per part")]
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for y, (lab, v, lo, hi, col, note) in enumerate(rows):
        if lo is not None:
            ax.hlines(y, lo, hi, color=col, lw=7, alpha=0.5)
            note = f"{note}: {lo:.1f}-{hi:.1f} %"
        ax.plot(v, y, "o", color=col, ms=14, zorder=3)
        ax.text(v, y - 0.22, f"{v:.1f} %", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.text(80.3, y + 0.3, note, fontsize=12, color=COLORS["muted"], va="center")
    ax.axvline(100, color=COLORS["navy"], lw=1.5, ls="--")
    ax.text(99.8, -0.6, "100 %", fontsize=12, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.8)
    ax.set_xlim(80, 101)
    ax.set_xlabel("performance (% of the operating time explained by the reference)")
    _title(fig, f"Performance {primary:.1f} % of demonstrated normal operation (upper bound);\n"
                "official OEE performance needs the ideal cycle time, which is not given")
    _save(fig, path)


# --------------------------------------------------------------------------
# kept figures, restyled: process flow, warehouse travel, event traces
# --------------------------------------------------------------------------
def process_flow(medians: dict, path):
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    w, h = 2.55, 1.75
    boxes = [
        (0.1, 2.9, "Warehouse", f"pallet out\nbelt {medians['belt_out']:.2f} s"),
        (3.1, 2.9, "Robot move 1", f"pallet -> oven\n{medians['robot_1']:.2f} s"),
        (6.1, 2.9, "Drilling station", f"oven {medians['oven']:.2f} s\ndrill {medians['drill']:.2f} s"),
        (9.1, 2.9, "Conveyor + sorting", f"small conveyor\n{medians['conveyor']:.2f} s, lane"),
        (9.1, 0.35, "Robot move 2", f"lane -> pallet\n{medians['robot_2']:.2f} s"),
        (4.6, 0.35, "Warehouse", f"belt in {medians['belt_in']:.2f} s\nstore pallet"),
    ]
    for x, y, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=COLORS["light"], ec=COLORS["accent"], lw=1.5))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + 0.65, body, ha="center", va="center", fontsize=12)
    arrows = [((2.65, 3.8), (3.1, 3.8)), ((5.65, 3.8), (6.1, 3.8)), ((8.65, 3.8), (9.1, 3.8)),
              ((10.4, 2.9), (10.4, 2.1)), ((9.1, 1.2), (7.15, 1.2)), ((4.6, 1.2), (1.4, 1.2)), ((1.4, 1.2), (1.4, 2.9))]
    for p, q in arrows:
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=COLORS["navy"], lw=1.5))
    ax.text(1.55, 1.95, "next pallet\nrequest", fontsize=12, color=COLORS["muted"])
    _title(fig, "One pallet cycle: a serial warehouse-robot chain (median step times in production)")
    _save(fig, path)


def travel_figure(w, path):
    fit = w.attrs["fit"]
    ok = w[w["used_in_fit"]]
    bad = w[w["cut_by_stop"] | w["resumed_after_stop"]]
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    xs = np.linspace(0, ok["distance"].max() * 1.05, 10)
    ax.plot(xs, fit["intercept_s"] + fit["slope_s_per_unit"] * xs, color=COLORS["muted"], lw=1.5, ls="--",
            label=f"constant speed {fit['speed_units_per_s']:.0f} units/s")
    for d, m, col in (("towards rack", "o", COLORS["accent"]), ("towards belt", "s", COLORS["navy"])):
        p = ok[ok["direction"] == d]
        ax.scatter(p["distance"], p["duration_s"], s=80, marker=m, color=col, label=d, zorder=3)
    ax.scatter(bad["distance"], bad["duration_s"], s=90, marker="x", color=COLORS["stop"],
               label="cut by / resumed after a stop (not used)")
    ax.set_xlabel("travel distance (rack units)")
    ax.set_ylabel("motor ON time (s)")
    ax.legend(loc="upper left")
    _title(fig, f"Axis travel time is linear in distance (R² = {fit['r2']:.3f}, n = {fit['n']}, largest residual "
                f"{fit['max_abs_residual_s']:.2f} s): no reduced speed")
    _save(fig, path)


def event_trace(ev, flags, data, stops, mw22, eq, path, pad=6.0):
    a, b = ev["start_s"] - pad, ev["end_s"] + pad
    sigs = list(dict.fromkeys(flags.loc[flags["event"] == ev["event"], "signal"]))
    tracks = []
    for s in sigs:
        plc = "Siemens" if s in data["Siemens"].signals else "Beckhoff"
        tracks.append((plc, s, PLAIN.get(s, s), False))
        if s in MOTOR_OF:
            tracks.append(("Beckhoff", MOTOR_OF[s], PLAIN.get(MOTOR_OF[s], MOTOR_OF[s]), False))
        if s in ("WH.Q1_beltForward", "WH.Q2_beltBackward"):
            tracks += [("Beckhoff", "WH.I2_Light_barrier_inside", PLAIN["WH.I2_Light_barrier_inside"], False),
                       ("Beckhoff", "WH.I3_Light_barrier_outside", PLAIN["WH.I3_Light_barrier_outside"], False)]
    tracks += [("Siemens", "A0.7", PLAIN["A0.7"], False), ("Siemens", "E1.0", PLAIN["E1.0"], True),
               ("Siemens", "A201.0", PLAIN["A201.0"], True), ("Beckhoff", "Visual.start", PLAIN["Visual.start"], True)]
    seen, uniq = set(), []
    for t in tracks:
        if t[1] in seen:
            continue
        seen.add(t[1])
        d = data[t[0]].frame
        w = d[(d["t"] >= a) & (d["t"] <= b)]
        x = ~w[t[1]].astype(bool) if t[3] else w[t[1]].astype(bool)
        if t[1] in sigs or x.any():          # keep only rows that are ON inside the window
            uniq.append(t)
    tracks = uniq
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for y, (plc, s, label, invert) in enumerate(tracks):
        d = data[plc].frame
        w = d[(d["t"] >= a - 30) & (d["t"] <= b + 30)]
        x = ~w[s].astype(bool) if invert else w[s].astype(bool)
        col = (COLORS["stop"] if s in ("A201.0", "Visual.start") else
               COLORS["overrun"] if s in sigs else COLORS["accent"])
        ax.broken_barh([p for p in _on_spans(w["t"], x) if p[0] + p[1] >= a and p[0] <= b], (y - 0.3, 0.6),
                       facecolors=col)
    ax.axvspan(ev["overrun_start_s"], ev["overrun_end_s"], color=COLORS["overrun"], alpha=0.13, zorder=0)
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels([t[2] for t in tracks])
    ax.set_ylim(len(tracks) - 0.4, -0.6)
    ax.set_xlim(a, b)
    ax.set_xlabel("time since the start of the recording (s); shaded = beyond the normal duration")
    vals = " / ".join(f"{v:.1f} s" for v in eq.loc[eq["event"] == ev["event"], "excess_s"])
    kind = "line-wide stop" if ev["loss_category"] == "minor stop" else "step overrun, stop or slow not separable"
    _title(fig, f"{ev['event']} ({kind}): equipment excess {vals}")
    _save(fig, path)
