"""Optional headless re-run of the analysis of notebooks 02-06: ``python run_mp1.py``.

Writes the same figures as the notebooks to reports/figures/ and the evidence tables to
reports/evidence_tables.xlsx; nothing else.

    prepare   notebooks 02-03  load, signal dictionary, time base, windows, cycles, activations
    model1    notebook 04      engineering rule, evidence, sensitivity, reduced speed, attribution
    model2    notebook 05      Isolation Forest and K-Means investigation
    evaluate  notebook 06      cycle-level effect, performance, time-grid check, registers
"""
from __future__ import annotations

import time
from io import StringIO

import pandas as pd

from . import data as dp
from . import evaluation as evl
from . import model1 as m1
from . import model2 as m2
from .config import TABLE_GROUPS, ensure_dirs, load_config


# --------------------------------------------------------------------------
# Stage 1: data understanding + preparation (notebooks 02-03)
# --------------------------------------------------------------------------
def prepare(cfg: dict, data: dict | None = None) -> dict:
    data = data or dp.load_all(cfg)
    sampling = dp.sampling_stats(data)
    gaps = dp.gap_table(data, cfg["timebase"]["gap_report_s"], cfg["timebase"]["gap_cooccur_tolerance_s"])
    sync = dp.cross_plc_sync(data, cfg["phases"]["curtain_signal"], cfg["phases"]["curtain_signal_beckhoff"])
    limit = float(sampling["max_step_s"].max())          # detection limit: largest logging step
    tol = float(sampling["median_step_s"].sum())         # cross-PLC tolerance: one median step of each PLC
    p0 = dp.production_start(data, cfg)
    stops = dp.stop_windows(data, cfg, tol)
    cycles = dp.pallet_cycles(data, cfg)
    acts = dp.mark_stop_cuts(dp.activations(data, cfg, cycles), stops)
    mw22 = dp.mw22_phases(data, cfg)
    rc = cfg["rule"]
    return {
        "cfg": cfg, "data": data,
        "dictionary": dp.build_dictionary(data),
        "sampling": sampling, "gaps": gaps,
        "resample": dp.resample_loss(data, cfg["timebase"]["resample_grids_ms"]),
        "pulses": dp.shortest_pulses(data), "sync": sync, "sync_summary": dp.sync_summary(sync),
        "limit": limit, "tol": tol,
        "min_excess": limit if rc["min_excess_s"] == "auto" else float(rc["min_excess_s"]),
        "p0": p0, "end": dp.recording_end(data), "stops": stops, "cycles": cycles, "acts": acts,
        "mw22": mw22, "dcyc": dp.drilling_cycles(mw22), "moves": dp.robot_moves(data, cycles),
        "actuators": dp.actuator_lists(data),
        "halt": dp.stop_halt_check(data, dp.actuator_lists(data), stops, tol),
    }


# --------------------------------------------------------------------------
# Stage 2: Model 1 - engineering hypotheses and rule-based detection (notebook 04)
# --------------------------------------------------------------------------
def model1(ctx: dict) -> dict:
    cfg, data, acts, stops, cycles, p0 = ctx["cfg"], ctx["data"], ctx["acts"], ctx["stops"], ctx["cycles"], ctx["p0"]
    rc, min_excess = cfg["rule"], ctx["min_excess"]
    # rule: Tukey fence + detection limit, overlapping flags -> one event
    flags, stats = m1.detect(acts, rc["tukey_k"], min_excess, rc["min_activations"], rc["merge_gap_s"])
    flags = m1.add_flag_evidence(flags, acts, data)
    flag_keys = set(zip(flags["signal"], flags["start"]))
    # independent PLC evidence (signals that did not define the detection)
    drive_off_all = m1.paired_output_windows(data, p0)
    drive_off = drive_off_all[drive_off_all["duration_s"] >= min_excess]
    enc = m1.encoder_silence(data, acts)
    belt = m1.belt_arrivals(data, acts)
    belt_off_bar = m1.barrier_changes_with_belt_off(data, p0)
    conv = m1.small_conveyor_arrivals(data, acts, ctx["gaps"])
    no_drive = m1.barrier_edges_without_drive(data, p0)
    # consolidation, confidence, sensitivity
    events = m1.consolidate(flags, data, ctx["dictionary"], stops, drive_off, enc, cycles, min_excess, ctx["halt"])
    sweep_flags = {(k, m): m1.detect(acts, k, m, rc["min_activations"], 0.0)[0]
                   for k in rc["sweep_k"] for m in rc["sweep_min_excess_s"]}
    events = m1.assign_confidence(events, m1.event_presence(sweep_flags, events), flags, min_excess)
    sweep = m1.threshold_sweep(acts, rc["sweep_k"], rc["sweep_min_excess_s"], rc["sweep_merge_gap_s"],
                                 rc["min_activations"], events)
    sc_all = m1.scored_activations(acts)
    alt_stats = m1.normal_durations(sc_all[~sc_all["cut_by_stop"] & ~sc_all["resumed_after_stop"]])
    alt_flags = m1.flag_long(sc_all, alt_stats, rc["tukey_k"], min_excess, rc["min_activations"])
    # activations above the fence but below the detection limit (rejected candidates)
    sc = sc_all.join(stats[["median", "q3", "iqr", "n"]], on="signal")
    sc["fence"] = sc["q3"] + rc["tukey_k"] * sc["iqr"]
    sc["excess"] = sc["duration"] - sc["median"]
    mild = sc[(sc["n"] >= rc["min_activations"]) & (sc["duration"] > sc["fence"]) & (sc["excess"] < min_excess)].copy()
    mild["edge_uncertainty_s"] = [dp.step_before(data[p].frame["t"].to_numpy(), s)
                                  + dp.step_before(data[p].frame["t"].to_numpy(), e)
                                  for p, s, e in zip(mild["plc"], mild["start"], mild["end"])]
    mild["excess_beyond_uncertainty"] = mild["excess"] > mild["edge_uncertainty_s"]
    # H9 reduced speed
    travel = m1.warehouse_travel(data, cycles, stops, p0)
    # physical attribution
    attr = m1.attribute(events, flags, acts, data, stops, drive_off, belt, conv, no_drive, belt_off_bar,
                        ctx["mw22"], ctx["tol"], cycles)
    events = events.merge(attr[["event", "attribution_category", "component", "location_confidence"]], on="event")
    ctx.update({
        "flags": flags, "stats": stats, "flag_keys": flag_keys, "drive_off_all": drive_off_all, "drive_off": drive_off,
        "enc": enc, "belt": belt, "belt_off_bar": belt_off_bar, "conv": conv, "no_drive": no_drive,
        "events": events, "sweep": sweep, "alt_flags": alt_flags, "mild": mild, "travel": travel, "attr": attr,
        "disp": m1.step_dispersion(acts, flag_keys),
        "robot": m1.robot_summary(ctx["moves"][ctx["moves"]["start_s"] >= p0]),
        "drilling": m1.drilling_summary(ctx["dcyc"], acts, flag_keys),
    })
    return ctx


# --------------------------------------------------------------------------
# Stage 3: Model 2 - machine-learning investigation (notebook 05)
# --------------------------------------------------------------------------
def model2(ctx: dict) -> dict:
    cfg, min_excess = ctx["cfg"], ctx["min_excess"]
    feat = m2.build_features(ctx["acts"], ctx["cycles"], ctx["flag_keys"], ctx["p0"], cfg["rule"]["min_activations"])
    if_res, if_sweep, if_sum = m2.isolation_forest_check(feat, cfg)
    km, km_sum = m2.kmeans_check(feat, cfg)
    feat["kmeans_long_cluster"] = m2.kmeans_long_cluster(feat, cfg, km_sum["best_k_by_silhouette"])
    top_a = m2.explain_rows(feat, "if_rank_A_duration_aware", 20, min_excess).assign(feature_set="A_duration_aware")
    top_b = m2.explain_rows(feat, "if_rank_B_context_only", 20, min_excess).assign(feature_set="B_context_only")
    mlcmp = pd.concat([top_a.rename(columns={"if_rank_A_duration_aware": "rank"}),
                       top_b.rename(columns={"if_rank_B_context_only": "rank"})])
    unexplained = int((mlcmp["explanation"] == m2.UNEXPLAINED).sum())
    km_members = feat[feat["kmeans_long_cluster"] & ~feat["rule_flag"]].copy()
    km_members["explanation"] = km_members.apply(m2.explanation, axis=1, min_excess_s=min_excess)         if len(km_members) else pd.Series(dtype=str)
    km_unexplained = int((km_members["explanation"] == m2.UNEXPLAINED).sum())
    a_s, b_s = if_sum["A_duration_aware"], if_sum["B_context_only"]
    # Decision rule fixed before running ML (docs/methodology.md section 6)
    reason = (
        f"Isolation Forest A (duration-aware) ranks on average {a_s['rule_flags_in_top_k_mean']:.1f} of the "
        f"{int(len(ctx['flags']))} rule overruns in its top {cfg['ml']['top_k']} (AP {a_s['average_precision_mean']:.2f}); "
        f"it uses the same relative-duration physics as the rule, so this is agreement, not validation. "
        f"Isolation Forest B (context only) reaches AP {b_s['average_precision_mean']:.2f} against a base rate of "
        f"{if_sum['base_rate']:.3f}. Among the top-20 of A and B together, {unexplained} activation(s) are "
        f"longer than normal without being flagged or explained by a stop. K-Means best silhouette "
        f"{km_sum['best_silhouette']:.2f} at k = {km_sum['best_k_by_silhouette']}; its cluster of long runs holds "
        f"{km_sum['best_k_flags_in_cluster']} rule overruns and {len(km_members)} other activation(s), "
        f"{km_unexplained} of them unexplained.")
    ctx.update({"feat": feat, "if_res": if_res, "if_sweep": if_sweep, "km": km, "mlcmp": mlcmp,
                "km_members": km_members, "ml": {
        "retained": bool(unexplained + km_unexplained > 0),
        "kmeans_cluster_other_members": int(len(km_members)), "kmeans_cluster_unexplained": km_unexplained,
        "decision": ("candidate new structure - review traces" if unexplained + km_unexplained else
                     "not retained: no verifiable additional loss beyond the engineering rule"),
        "reason": reason, "isolation_forest": if_sum, "kmeans": km_sum,
        "context_only_top20_unexplained_context": unexplained, "top_k": cfg["ml"]["top_k"]}})
    return ctx


# --------------------------------------------------------------------------
# Stage 4: cross-checking and evaluation (notebook 06)
# --------------------------------------------------------------------------
def evaluate(ctx: dict) -> dict:
    cfg, data, acts, stops, cycles = ctx["cfg"], ctx["data"], ctx["acts"], ctx["stops"], ctx["cycles"]
    events = ctx["events"]
    # H7: cycle-level effect; cycle reference model; performance
    done, eff = evl.cycle_effect(cycles, events)
    handover = evl.chain_handover(done, acts, ctx["moves"])
    cref, cref_info = evl.cycle_reference(done, ctx["travel"], ctx["moves"], acts, events, stops, ctx["flag_keys"])
    cref = cref.merge(evl.chain_coverage(done, acts)[["cycle", "chain_busy_s", "chain_idle_s"]], on="cycle")
    perf = evl.performance_estimates(done, eff, cfg["performance"]["bootstrap_resamples"],
                                    cfg["performance"]["bootstrap_seed"], cref)
    events = events.copy()
    events["serial_chain_effect"] = evl.event_chain_effect(events, handover)
    # H8: nothing missed (duration-independent check)
    quiet, quiet_limit = m1.quiet_completeness(data, ctx["actuators"], ctx["p0"], done, events, acts,
                                                 ctx["flag_keys"])
    # time-grid sensitivity of the whole result (data-preparation decision)
    raw = evl.run_on(data, cfg, ctx["min_excess"], ctx["tol"])
    grid = evl.event_sensitivity(data, cfg, raw, cfg["timebase"]["resample_grids_ms"], ctx["min_excess"],
                                 ctx["tol"], list(events["event"]))
    register = evl.evidence_register(events, ctx["attr"], ctx["flags"], ctx["feat"], quiet, grid)
    candidates = evl.candidate_register(ctx["flags"], ctx["mild"], ctx["mlcmp"], quiet, ctx["min_excess"],
                                       m2.UNEXPLAINED, ctx["km_members"])
    ctx.update({"events": events, "done": done, "eff": eff, "handover": handover, "cref": cref,
                "cref_info": {**cref_info, "chain_idle_max_s": float(cref["chain_idle_s"].max())},
                "perf": perf, "quiet": quiet, "quiet_limit": quiet_limit, "grid": grid,
                "register": register, "candidates": candidates})
    # quantified results by level (equipment, incident, cycle, performance; Model 2 counts)
    q = evl.quantified_results(ctx)
    ctx.update({"q_equipment": q["equipment"], "q_incidents": q["incidents"], "q_cycles": q["cycles"],
                "q_total": q["event_cycles_total"], "q_ml": q["ml"], "q_levels": q["levels"],
                "q_methods": q["methods"]})
    return ctx


# --------------------------------------------------------------------------
# Outputs: evidence workbook and figures
# --------------------------------------------------------------------------
TABLES = {  # table name -> ctx key (one sheet each in reports/evidence_tables.xlsx)
    "signal_dictionary": "dictionary", "sampling": "sampling", "gaps": "gaps", "sync_check": "sync",
    "shortest_pulses": "pulses", "resample_loss": "resample", "resample_event_sensitivity": "grid",
    "stops": "stops", "stop_halt_check": "halt", "cycles": "done", "drilling_cycles": "dcyc", "robot_moves": "moves",
    "normal_durations": "stats", "events_rule": "flags", "threshold_sweep": "sweep", "slow_steps": "mild",
    "step_dispersion": "disp", "warehouse_travel": "travel", "attribution": "attr",
    "paired_output_windows": "drive_off_all", "encoder_silence": "enc", "belt_arrivals": "belt",
    "barrier_changes_belt_off": "belt_off_bar", "small_conveyor_arrivals": "conv",
    "barrier_edges_without_drive": "no_drive", "ml_isolation_forest": "if_res", "ml_contamination_sweep": "if_sweep",
    "ml_kmeans": "km", "ml_vs_rule": "mlcmp", "line_quiet_periods": "quiet", "cycle_reference": "cref",
    "cycle_handover": "handover", "performance": "perf", "events_final": "events",
    "event_evidence_register": "register", "candidate_register": "candidates",
    "equipment_quantification": "q_equipment", "incident_quantification": "q_incidents",
    "cycle_quantification": "q_cycles", "ml_quantification": "q_ml", "results_by_level": "q_levels",
    "method_comparison": "q_methods",
}


def evidence_tables(ctx: dict) -> dict:
    """All evidence tables as DataFrames, rounded and typed exactly as they are stored."""
    T = {}
    for name, key in TABLES.items():
        df = ctx[key].reset_index() if name == "normal_durations" else ctx[key]
        T[name] = pd.read_csv(StringIO(df.round(4).to_csv(index=False)))
    return T


def write_evidence_workbook(T: dict, path) -> None:
    """One sheet per evidence table, in notebook order, plus an index sheet."""
    order = [(g, n) for g, names in TABLE_GROUPS.items() for n in names]
    index = pd.DataFrame([{"sheet": n, "notebook": g, "rows": len(T[n]), "columns": T[n].shape[1]} for g, n in order])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        index.to_excel(xw, sheet_name="index", index=False)
        for _, n in order:
            T[n].to_excel(xw, sheet_name=n, index=False)


def write_figures(ctx: dict, F) -> None:
    """The figures of notebooks 02, 04 and 06 (same functions, same files)."""
    from . import figures as fg
    fg.apply_style()
    data, events, flags, eq = ctx["data"], ctx["events"], ctx["flags"], ctx["q_equipment"]
    fg.time_base(data, ctx["gaps"], ctx["limit"], F / "time_base.png")
    fg.event_timeline(events, ctx["stops"], eq, ctx["p0"], ctx["end"], F / "event_timeline.png")
    fg.attribution_bars(eq, events, F / "attribution_bars.png")
    fg.evidence_stop(data, events, ctx["stops"], ctx["drive_off_all"], F / "evidence_stop.png")
    fg.evidence_overrun(data, ctx["belt"], flags, F / "evidence_overrun.png")
    fg.robustness(ctx["sweep"], ctx["grid"], len(events), ctx["limit"], F / "robustness.png")
    med, moves = ctx["stats"]["median"], ctx["moves"]
    fg.process_flow({"belt_out": med["WH.Q1_beltForward"], "belt_in": med["WH.Q2_beltBackward"],
                     "robot_1": float(moves.loc[moves["move_type"] == "pallet -> oven", "duration_s"].median()),
                     "robot_2": float(moves.loc[moves["move_type"] == "lane -> pallet", "duration_s"].median()),
                     "oven": float(ctx["dcyc"]["oven_phase_s"].median()), "drill": med["A1.0"],
                     "conveyor": med["A0.7"]}, F / "process_flow.png")
    fg.travel_figure(ctx["travel"], F / "warehouse_travel.png")
    for _, ev in events.iterrows():
        fg.event_trace(ev, flags, data, ctx["stops"], ctx["mw22"], eq, F / f"trace_{ev['event']}.png")
    fg.cycle_effect_figure(ctx["q_cycles"], ctx["q_total"], F / "cycle_effect.png")
    fg.method_comparison_figure(ctx["q_ml"], len(flags), F / "method_comparison.png")
    fg.performance_bounds(ctx["perf"], F / "performance_bounds.png")


def main(cfg: dict | None = None, make_figures: bool = True) -> dict:
    t_start = time.time()
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    ctx = evaluate(model2(model1(prepare(cfg))))
    R, F = cfg["paths"]["reports_dir"], cfg["paths"]["figures_dir"]
    write_evidence_workbook(evidence_tables(ctx), R / "evidence_tables.xlsx")
    if make_figures:
        write_figures(ctx, F)
    print(f"MP1 re-run finished in {time.time() - t_start:.1f} s -> {R / 'evidence_tables.xlsx'}, {F}")
    print(f"  events: {ctx['events']['event'].nunique()}  flagged activations: {len(ctx['flags'])}")
    return ctx


if __name__ == "__main__":
    main()
