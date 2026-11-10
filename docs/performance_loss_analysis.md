# Performance Loss Analysis – Explained

Companion to `notebooks/02_performance_loss_analysis.ipynb`
(same code as `scripts/performance_loss_analysis.py`).
Every number in this document is printed by the notebook. If the data
changes, re-run the notebook and update the numbers here.

---

## 1. The result in one paragraph

The machine produced 9 pallets in a 506 s recording. In the production
window (8.9 s to 505.8 s) we found **8 performance-loss events (E1–E8)**.
That is within the "1 to 10" given in the task. **4 events are short stops
with a visible cause:** 3 safety-curtain stops, each ended by the HMI reset
button, and 1 HMI stop. **4 events are slow transports without a cause
signal:** 3 on the warehouse belt and 1 on the drilling-station small
conveyor. Every pallet cycle that contains an event is longer than every
cycle without one (exact permutation test, p = 0.029).

| Event | Time (s) | Lost (s) | Equipment | What happened | Cause in data |
|---|---|---|---|---|---|
| E1 | 252.4–272.2 | 10.5 | Warehouse horizontal axis (+ drilling-station vacuum suction waits) | Axis stands still 11.8 s | Safety curtain 255.4–264.3, ended by reset. Axis already waited **3.0 s before** the curtain signal |
| E2 | 291.3–299.1 | 6.9 | Warehouse conveyor belt, forward | Pallet reaches barrier after 7.5 s (normal 0.4–0.6 s) | none |
| E3 | 314.8–321.6 | 6.2 | Warehouse conveyor belt, backward | Pallet reaches barrier after 6.7 s | none |
| E4 | 350.8–360.4 | 8.8 | Warehouse conveyor belt, forward | Pallet reaches barrier after 9.3 s | none |
| E5 | 363.1–371.1 | 2.2 | Drilling station, small conveyor (A0.7) | Part reaches barrier E1.0 after 2.56 s (normal 0.77–1.00 s) | none (weakest event) |
| E6 | 385.1–400.9 | 10.4 | Drilling station drill (A1.0) + warehouse cantilever | Drill ON 5.07 s (normal ~1 s), then cantilever stands still 10.3 s | Safety curtain 390.1–400.4, ended by reset. Drill was already **4.0 s overdue** |
| E7 | 453.0–463.3 | 7.2 | Warehouse horizontal axis | Axis stands still 7.6 s | Safety curtain 453.1–460.7, ended by reset |
| E8 | 468.5–476.5 | 4.9 | Warehouse cantilever, backward | Cantilever stands still 4.9 s | HMI stop 468.5–473.5 |

"Lost (s)" = how much longer than normal (median) the most delayed output
of the event was switched on.

---

## 2. Rules we followed (and why)

The task said: be accurate, use as few assumptions as possible, follow
CRISP-DM. We turned that into four rules:

1. **No hand-picked windows or thresholds.** Every limit is either
   *measured* from the data (for example the largest logging gap) or a
   *textbook rule* (Tukey's fence). Each one is printed in the notebook.
2. **Each statement is labelled with its source:**
   - **documented** – written in `Variablenerklärung.xlsx`
   - **measured** – computed from the recording
   - **reading of a name** – only the signal name suggests the meaning
     (this happens for Beckhoff, whose sheet lists only the station)
   - **open question** – the data cannot decide it
3. **Hypotheses first.** Test and decision rule are written down before
   the result is computed.
4. **The cleaned data is never changed.**

---

## 3. CRISP-DM: where each part of the work belongs

| Phase | What we did | Notebook section |
|---|---|---|
| 1 Business understanding | Task, OEE performance row, working definition, constraints, hypotheses | Phase 1 |
| 2 Data understanding | Logging resolution, documentation, production start, clock check (H1), cycle structure (H2) | Steps 2.1–2.7 |
| 3 Data preparation | Activation table: one row per ON-period of an actuator output | Step 3.1 |
| 4 Modelling | Statistical detection rule, grouping into events, sensitivity (H3) | Steps 4.1–4.2 |
| 5 Evaluation | Real or artefact (H4), standstill or slow (H5), cause (H6), effect on cycle time (H7), completeness (H8) | Steps 5.1–5.9 |
| 6 Deployment | CSV tables and figures in `results/`, this document | Phase 6 |

The "model" here is a transparent statistical rule, not machine learning.
Section 10 explains why that is the right first step and what it means
for choosing a model later.

---

## 4. Business understanding

### 4.1 OEE and the performance row

From the lecture slide:

```
OEE = Availability x Performance x Quality

Performance = (theoretical cycle time x number of parts) / up time
```

Performance losses are the **losses in speed** during up time:

- **Idle time and short stops:** the machine waits, nothing is broken.
- **Reduced speed:** the machine moves, but slower than designed.

### 4.2 Why we report seconds, not an OEE %

The Performance formula needs the **theoretical cycle time**, and the
task does not give it. We could only estimate it from the data, and that
would be an assumption. So we report **lost seconds** per event and per
cycle. When the theoretical cycle time is known, the OEE value follows
directly.

### 4.3 Working definition

> The machine repeats the same pallet cycle. A performance loss is a
> step that takes **clearly longer than the same step normally takes**.

"Clearly longer" is made exact in Section 6.

### 4.4 Constraints of the data

| Constraint | Consequence |
|---|---|
| One recording, about 506 s, 9 pallets | Small sample: only 7 complete cycles |
| No theoretical cycle time | No OEE %, only lost seconds |
| No labels (nobody marked the losses) | Supervised learning is impossible, detection must be unsupervised |
| Siemens: every signal described. Beckhoff: only the station | Beckhoff meanings are readings of names and are marked as such |
| Irregular logging: median 76 ms (Siemens) / 93 ms (Beckhoff), gaps up to 1.84 s | Deviations below 1.84 s cannot be told apart from a logging gap |
| No encoder on the warehouse belt, the cantilever or the small conveyor | For those, "standstill" and "slow" cannot be separated |

---

## 5. Data understanding

### 5.1 Two limits measured from the logging

| Limit | Value | How it is derived | Used for |
|---|---|---|---|
| **Detection limit** | 1.84 s | Largest time between two log rows in either file | Minimum extra time to count as a loss. Also the minimum length of a "standstill" window |
| **Sync tolerance** | 0.169 s | One median step of Siemens (0.076 s) + one of Beckhoff (0.093 s) | Largest difference allowed when the same edge is compared across the PLCs |

Why the sum of both median steps? Each logger notices a switch at its
next sample, so it can be up to one step late. If both are late in
opposite directions, the same edge can differ by up to both steps.

### 5.2 Start of production

The first press of the HMI reset button (`E200.0`, documented: "Reset
button from HMI") happens at **8.93 s**. Before it, the warehouse homes
its cantilever (0.41–10.45 s). Every activation that starts before 8.93 s
is excluded. The previous version used a hand-picked 12 s instead.

### 5.3 H1 – Are the two PLC clocks in sync?

The pre-processing notebook tried to estimate the offset by correlating
overall activity and did not succeed (correlation 0.024). There is a
direct test: the Siemens output `A201.0` is **documented** as "Safetydata
from curtain to beckhoff". The Beckhoff input `VG.safeCurtainShort`
switches at the same moments. The same physical edge is therefore logged
by both PLCs:

| Siemens A201.0 (s) | Beckhoff safeCurtainShort (s) | Difference (s) |
|---|---|---|
| 8.851 | 8.882 | +0.031 |
| 255.396 | 255.487 | +0.091 |
| 264.292 | 264.263 | −0.029 |
| 390.061 | 390.100 | +0.039 |
| 400.438 | 400.396 | −0.042 |
| 453.141 | 453.190 | +0.049 |
| 460.743 | 460.772 | +0.029 |

The largest difference is 0.091 s, below the 0.169 s tolerance, and the
sign changes (the clocks do not drift apart). **H1 accepted:** no clock
correction is needed. This also settles the open point in the
pre-processing notebook.

### 5.4 H2 – Does the machine repeat the same cycle?

The Beckhoff signals `WH.desiredPalletX/Y` (reading of the name:
requested rack position) change 9 times. Each change starts a **pallet
cycle**.

| Cycle | Start (s) | Duration (s) | Belt fwd | Belt back | Robot moves | Complete? |
|---|---|---|---|---|---|---|
| 1 | 0.4 | 45.0 | 1 | 1 | 1 | no (contains start-up) |
| 2–8 | 45.4 … 404.4 | 48.6 – 68.2 | 1 | 1 | 2 | **yes** |
| 9 | 453.0 | 52.8 | 1 | 1 | 1 | no (cut off by end of recording) |

Every complete cycle has the same steps. The drilling station runs each
of its steps (oven in, oven out, drill, small conveyor, suction) 9 times
in total, once per part. **H2 accepted:** comparing a step with "the
same step in other cycles" makes sense.

---

## 6. Data preparation and modelling

### 6.1 The activation table

An **activation** is one ON-period of one actuator output:

```
signal:   0 0 0 1 1 1 1 1 0 0 0
                |<-- duration -->|
              start            end
```

Only actuator outputs can be "too long", so only they are used:

- **Siemens:** signals starting with `A` at the sorting or drilling
  station (documented outputs). 19 outputs.
- **Beckhoff:** signals with `.Q` in the name. `Q` is the IEC 61131-3
  notation for an output. 22 outputs.

Result: **677 activations** of 41 outputs in the production window.
Nothing is resampled. We work with the logged switching times, so short
pulses are not lost.

### 6.2 Normal duration: median and IQR

For each output we describe its normal ON-duration with **robust**
statistics. One extreme value cannot shift them much:

- **Median:** the middle value.
- **Q1 / Q3:** the values below which 25 % / 75 % of the durations lie.
- **IQR = Q3 − Q1:** the width of the middle half of the data.

The mean and standard deviation are **not** used. With only 9 values, a
single 5 s drill run would pull the mean up and hide itself.

### 6.3 The detection rule

An activation is flagged as **too long** if both conditions hold:

1. **Tukey's fence:** duration > Q3 + k × IQR with k = 3. This is
   Tukey's (1977) standard rule for "far out" values. No tuning.
2. **Above the detection limit:** duration − median ≥ 1.84 s.

Worked example, drill `A1.0`: durations 0.94–1.05 s, median 0.99 s,
fence 1.22 s. The run at 385.07 s lasted **5.07 s**: above the fence, and
4.08 s longer than the median, which is more than 1.84 s. **Flagged.**

Outputs with fewer than 5 activations are skipped, because quartiles of
4 or fewer values are not meaningful. This affects only the three sorting
pushers, and none of them is ever more than 1.32 s above its median. So
skipping them changes nothing.

### 6.4 From activations to events

Flagged activations that **overlap in time** are one event. For example,
in E1 the warehouse axis and the drilling-station suction are both too
long at the same time. No "gap" parameter is needed.

The previous version merged detections up to 5 s apart. That joined the
warehouse belt (E4) and the small conveyor (E5), two different machines
on two PLCs, into one event. They are now separate.

### 6.5 H3 – Result and sensitivity

10 flagged activations → **8 events**.

| Fence k | Minimum excess | Flagged activations | Events |
|---|---|---|---|
| 1.5 | 1.00 s | 29 | 16 |
| 1.5 | **1.84 s** | **10** | **8** |
| 1.5 | 3.00 s | 9 | 7 |
| 3.0 | 1.00 s | 14 | 10 |
| 3.0 | **1.84 s** | **10** | **8** |
| 3.0 | 3.00 s | 9 | 7 |

- At the detection limit, the fence factor makes **no** difference.
- Below the limit (1 s), many more small deviations appear. Those are
  too small to separate from logging gaps.
- At 3 s, only E5 (+2.2 s) disappears, so E5 is the least certain event.

**H3 accepted:** 8 events lie within 1–10 and do not depend on k.

---

## 7. Evaluation

### 7.1 H4 – Real, or a logging gap / longer travel?

Two alternative explanations are tested for each flagged activation:

- **Logging gap:** each edge can be off by the logging step before it.
  Subtracting both uncertainties (at most 0.21 s) still leaves an excess
  of at least **+2.06 s** (E5) and up to +10.3 s.
- **Longer travel to a far rack position:** each flagged activation is
  longer than the **longest normal** activation of the same output by at
  least **1.96 s** (E5), which is more than the detection limit. The
  longest normal activation already includes the farthest rack position
  in the recording.

**H4 accepted.**

### 7.2 H5 – Standstill or slow motion?

This decides between the two OEE performance sub-types. Three kinds of
evidence are used.

**(a) Paired output signals (warehouse).** For every warehouse motor
direction the Beckhoff file has two signals, for example `WH.Q3_horRack`
and `WH.Q3_horizontal_towards_rack`. The variable file does not explain
the difference, so we measured it. In normal operation the two signals
switch within 0.26 s of each other (107 windows). There are only 4
windows in which the first one stays ON while its partner is OFF for
longer than the detection limit:

| Signal pair | Partner OFF (s) | Length (s) | Event |
|---|---|---|---|
| Q3 horizontal axis | 252.42–264.26 | 11.84 | E1 |
| Q3 horizontal axis | 453.09–460.68 | 7.58 | E7 |
| Q7 cantilever forward | 390.10–400.40 | 10.30 | E6 |
| Q8 cantilever backward | 468.45–473.34 | 4.89 | E8 |

**(b) Encoder of the horizontal axis.** A moving axis produces encoder
edges. In the 24 normal horizontal moves the encoder is never silent for
more than **0.88 s**. In E1 it is silent for **11.84 s** and in E7 for
**7.58 s**, exactly while the partner signal is OFF. So the partner
behaves like the **motor output** and the first signal like the
**request**. That reading fits the data but is not documented.

*Why only "silent or not", and no speed from the pulse count?* The logger
samples about 11 times per second, close to the pulse rate, so counts
are unreliable. A silence of several seconds, however, is unambiguous.

**(c) Arrival at the light barrier (belt and small conveyor).** These
have no encoder. What we can measure is when the transported part reaches
the light barrier:

- The warehouse belt switches off 0 to 0.45 s after the last barrier
  change, in all 18 runs. In other words, **the belt runs until the
  pallet reaches the barrier**.
- Normally the pallet arrives after 0.4–0.65 s. In E2, E3 and E4 it
  takes **7.5 s, 6.7 s and 9.3 s**.
- The outside barrier `WH.I3` changes while the belt is **off** only 4
  times, all in the cycles of E2/E3 and E4 (308.3 / 310.8 s and
  361.3 / 364.1 s). Something moved in front of the barrier while no belt
  was running. What it was is an **open question**.
- E5: the part reaches the small-conveyor barrier `E1.0` after 2.56 s
  instead of 0.77–1.00 s.
- E6: the drill-time variables `MD28` / `MW32` (documented: "Drilling
  time in second") change value **only** during the one long drill run
  (385.07–390.06 s).

**Result H5:** standstill is proven for **E1, E6, E7, E8**, which makes
them idle time / short stops. For **E2–E5** it cannot be decided whether
the part stood still or moved slowly. Either way, both sub-types belong
to the performance row.

### 7.3 H6 – What caused the events?

| Cause window | Signals and checks | Matches |
|---|---|---|
| Safety curtain stop 255.40–264.29 s (8.9 s) | `A201.0` low (documented: safety data from curtain), red light `M0.3` on (documented: "1 segment red ON"), ends at reset press `E200.0` | E1 |
| Safety curtain stop 390.06–400.44 s (10.4 s) | same three checks | E6 |
| Safety curtain stop 453.14–460.74 s (7.6 s) | same three checks | E7 |
| HMI stop 468.55–473.45 s (4.9 s) | `Visual.start` low, `Visual.Stoplight` high (station "Visualization", reading of names) | E8 |

Two findings that the cause does **not** explain:

- **E1:** the horizontal-axis motor output was already OFF **3.0 s
  before** the curtain signal dropped (252.4 s vs 255.4 s).
- **E6:** the drill was already **4.0 s overdue** when the curtain
  stopped the machine. The drill switched off at the curtain stop.

We do not claim *why* the curtain was interrupted. The data shows only
*that* it was, and that each stop was ended by a reset.

**Result H6:** cause found for 4 of 8 events. E2–E5 have no cause signal.

### 7.4 H7 – Do the events make the cycle longer?

Complete cycles (2–8):

| Cycle | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Duration (s) | 53.1 | 57.6 | 53.9 | **63.1** | **63.1** | **68.2** | 48.6 |
| Events | – | – | – | E1 | E2, E3 | E4, E5, E6 | – |

**Exact permutation test, explained.** If the events had no effect, the
labels "with event" and "without event" would be arbitrary. There are
C(7,3) = 35 ways to choose 3 of the 7 cycles. For each way we compute
"mean of the 3 chosen − mean of the other 4". Only 1 of the 35 is as
large as the observed one (11.5 s), so **p = 1/35 = 0.029**. The test
needs no assumption about the distribution (no normality).

- Every event cycle is longer than every clean cycle.
- p = 0.029 is the **smallest p-value possible** with 7 cycles. The
  separation is complete, but the sample is small.
- **H7 accepted.**
- The 3 event cycles took **33.9 s** longer than 3 median clean cycles
  (53.5 s each). Their events add up to 45.0 s of equipment lost time.
  The difference is real: other equipment keeps working in parallel. For
  example, in E2 the robot started its pick 1.45 s after the belt start,
  as in normal cycles, while the belt was still waiting for the pallet.
  Equipment lost time is therefore **not** simply added up for the
  machine.
- E7 and E8 lie in cycle 9, which is cut off, so they cannot be tested
  this way.

### 7.5 H8 – Did we miss anything?

The duration rule only sees outputs that are **ON**. A stop in which
**nothing** is switched on would be invisible to it. So we measured every
period in which **no actuator output of either PLC switches**:

- Longest such period inside the clean cycles: **3.96 s** (normal).
- Periods longer than that: 6. Five lie inside E1, E4, E6, E7, E8. The
  sixth (4.28 s at 497.9 s) is a warehouse horizontal move of 4.37 s,
  within its normal range (up to 4.42 s).

**H8 accepted:** no hidden stop exists.

### 7.6 All hypotheses

| ID | Phase | Hypothesis | Result |
|---|---|---|---|
| H1 | Data understanding | PLC clocks are in sync | Accepted (max 0.091 s ≤ 0.169 s) |
| H2 | Data understanding | Machine repeats one cycle | Accepted (7 identical cycles) |
| H3 | Modelling | Losses = activations far longer than normal | Accepted (8 events, independent of k) |
| H4 | Evaluation | Detections are real (no gap, no distance) | Accepted |
| H5 | Evaluation | Extra time is a standstill | 4 of 8 proven, 4 not decidable |
| H6 | Evaluation | Cause is visible (curtain or HMI) | 4 of 8 |
| H7 | Evaluation | Events make the cycle longer | Accepted (p = 0.029) |
| H8 | Evaluation | No stop is missed | Accepted |

---

## 8. Limitations (say these openly)

1. **Small sample:** 7 complete cycles. The statistics are as strong as
   7 cycles allow, and no stronger.
2. **No theoretical cycle time:** no OEE % can be calculated.
3. **Beckhoff meanings come from names**, for example "request vs motor
   output" and "Visual.start = HMI start". The data is consistent with
   these readings, but the variable file does not state them.
4. **No motion sensor** on the belt, the cantilever (only the paired
   output), the small conveyor or the drill. So E2–E5 cannot be split
   into "stop" or "slow".
5. **Outputs with widely varying normal durations** (compressor `A1.6`:
   19–55 s; oven door `A4.1`: 22–74 s) cannot be judged by a duration
   rule.
6. **Timer-driven slow-downs:** if a motor ran slower but its PLC timer
   switched it off on time, the ON-duration would look normal. This is
   checked only where a light barrier exists (belt, small conveyor).
7. **Timing resolution:** edges are known to about ±0.1–0.2 s, and to
   ±1.8 s near a logging gap.

---

## 9. Open questions for the professor

1. E2–E4: the warehouse belt runs, but the pallet reaches the barrier
   7–9 s late, and the outside barrier changes while the belt is off.
   Was the pallet blocked or held? Is that one of the reported losses?
2. Do the safety-curtain stops (E1, E6, E7) count as **performance**
   losses (short stops) or as **availability** losses?
3. E1: why was the horizontal axis already waiting 3 s before the
   curtain signal? E6: why did the drill run 5 s instead of 1 s?
4. Can you confirm the meaning of the paired warehouse signals
   (`WH.Q3_horRack` vs `WH.Q3_horizontal_towards_rack`) and of
   `Visual.start` / `Visual.Stoplight`?
5. Is a theoretical cycle time available, so that an OEE performance
   value can be calculated?

---

## 10. What this means for choosing a model (next step)

| Fact from this analysis | Consequence for the model choice |
|---|---|
| 7 complete cycles, 8 events, no labels | Not enough to train or validate a supervised or deep model (for example an LSTM) |
| The informative unit is the **activation** (duration per step), not the raw 50 ms grid | Any model should take the activation table as input (features: signal, duration, position in cycle, rack position) |
| A transparent rule already finds 8 events and H8 shows nothing else is there | The rule is the **baseline**. Its 8 events are the **reference labels** to evaluate any model |
| Several normal durations depend on rack position or are bimodal (robot short vs long move) | A model should learn "normal per step and context", for example per signal and rack position |

Candidate model families to compare next, against the baseline:

- **Unsupervised outlier detection** on the activation table, for
  example Isolation Forest or Local Outlier Factor.
- **Clustering of cycles**, for example k-means on step durations per
  cycle.

A model is only worth choosing if it finds the same 8 events without
adding events that H8 shows do not exist, and if it is still explainable.

---

## 11. What changed compared with the previous version

| Previous version | Now | Why |
|---|---|---|
| Start-up window hand-picked at 12 s | First documented reset press (8.93 s) | No hand-picked number |
| 6 tuned parameters (ratio 1.3, isolation ±25 %, 3 peers, 5 s cluster gap, …) | Tukey k = 3 and the measured detection limit; overlap grouping | Fewer assumptions, same detections |
| 7 events (belt + small conveyor merged) | 8 events | Different equipment, no overlap in time |
| Lost time: sometimes cause-window length, sometimes excess | Always duration − median | One rule for all events |
| Clock sync "not verified" | Verified with a documented cross-PLC signal | Direct test instead of correlation |
| "Curtain caused the stop" | Plus: 3.0 s unexplained wait before the curtain (E1), 4.0 s drill overrun (E6) | The data shows more than the cause alone |
| Belt events "unexplained" | Mechanism measured: belt runs until the barrier, pallet 6.7–9.3 s late, barrier changes with the belt off | More precise, cause still open |
| H6: cycle time "rises by roughly the lost time" | Exact permutation test; cycle impact (33.9 s) smaller than the equipment lost time (45.0 s) | Correct statistics, honest about parallel work |
| No completeness check | H8: no hidden stop | A duration rule cannot see idle-only stops |

---

## 12. Five-minute presentation outline

1. **Task and definition (30 s):** OEE performance row. Our definition: a
   step that takes clearly longer than the same step normally takes.
2. **Data and constraints (45 s):** two PLCs, 506 s, 9 pallets. The
   Beckhoff signals are not documented. Detection limit 1.84 s from the
   logging gaps.
3. **Can we trust the timing? (30 s):** the clock check with the
   documented curtain signal, 0.09 s.
4. **Method (60 s):** activation table → median / IQR → Tukey fence +
   detection limit → overlapping detections = event. Show the drill
   example.
5. **Result (60 s):** Figure 1 (timeline), the event table, 8 events.
6. **Evidence (60 s):** Figure 2 (encoder silent = standstill), Figure 3
   (belt arrival), Figure 4 (cycle durations, p = 0.029).
7. **Limits and open questions (15 s):** E2–E5 without a cause, whether
   safety stops count as performance or availability.

### Likely questions and short answers

- *Why not machine learning?* No labels and 7 cycles. A model could not
  be validated. The rule is the baseline and creates the labels for the
  next step.
- *Why k = 3?* It is Tukey's standard "far out" rule, and k = 1.5 gives
  the same 8 events.
- *Why 1.84 s?* It is the largest logging gap, measured. Anything smaller
  could be a gap in the log.
- *Is 11 % / X % the OEE?* No. Without a theoretical cycle time we report
  seconds only.
- *Why is the cycle loss (33.9 s) smaller than the sum of event losses
  (45.0 s)?* Other equipment works in parallel while one piece of
  equipment waits.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| Activation | One ON-period of one actuator output |
| Median | Middle value of the sorted durations |
| IQR | Interquartile range, Q3 − Q1: width of the middle 50 % |
| Tukey fence | Q3 + k × IQR. Values above it are outliers (k = 3: "far out") |
| Detection limit | Largest logging gap (1.84 s). The smallest excess we can trust |
| Sync tolerance | Largest expected difference of one edge seen by both PLCs (0.169 s) |
| Encoder | Sensor that produces pulses while an axis moves |
| Light barrier | Sensor that changes state when an object passes it |
| Permutation test | Test that compares the observed split with all possible splits of the same data |
| Pallet cycle | Time between two pallet requests (`WH.desiredPalletX/Y` change) |

---

## 14. How to run

```bash
# from the project root
python scripts/performance_loss_analysis.py      # writes results/ and figures
# or open notebooks/02_performance_loss_analysis.ipynb and run all cells
```

Outputs:

- `results/performance_loss_events.csv` – the 8 events
- `results/long_activations.csv` – the 10 flagged activations with all
  checks
- `results/pallet_cycles.csv` – complete cycles with durations and events
- `results/hypothesis_results.csv` – H1–H8
- `results/figures/fig1_event_timeline.png`, `fig2_stall_evidence.png`,
  `fig3_belt_arrival.png`, `fig4_cycle_durations.png`

To regenerate the notebook after editing the script, convert the
`# %%` cells, for example with `jupytext --to notebook`.
