# Designing Machine Learning Systems — design workbook

Use this workbook with the 16-lesson course in Furina ML. It contains a worked
delivery-delay design, local lab instructions, exercises and a submission rubric.
The design targets below are illustrative requirements, not measured service
performance. Replace them with evidence for your own project.

## 1. Product brief

### Worked example

| Question | Proposed answer |
|---|---|
| User | A dispatcher deciding which deliveries need a status check |
| Unit | One delivery at its dispatch timestamp |
| Prediction | Probability of arriving after the original promised window |
| Action | Add the delivery to an inspection workflow |
| Label | Late or on time relative to the promise recorded at dispatch |
| Unknown outcomes | Pending until the documented outcome policy resolves them |
| Baseline | Current workflow or a simple rule; the executable lab uses a constant score |
| Product outcome | Missed promises, with inspection workload and complaints as guardrails |
| Exclusions | No automated driver discipline, route control or real customer notifications |
| Stopping condition | Unreliable labels or no useful gain over the baseline within the agreed budget |

**Your exercise:** define the decision, user, horizon, label policy and stopping
condition for your own case. Name the product owner who can validate them.

Choose the action policy explicitly. A fixed-capacity inspection queue might
rank the top 100 cases per hour. The lab instead minimizes a declared binary
error cost. These are different policies; the lab does not enforce a top-100
constraint or simulate whether inspections prevent delays.

## 2. Requirements and acceptance criteria

Complete the evidence column after measuring or reviewing the assumption.

| Requirement | Worked assumption | Evidence still needed |
|---|---|---|
| Peak load | 1,000 eligible requests per second | Production traffic profile |
| Request deadline | 200 ms end to end | Representative load test |
| Deadline-success target | 99.9% over 28 days | Agreed eligibility and success rules |
| Feature freshness | At most 5 minutes for live congestion | Measured ingestion and materialization delay |
| Model quality | Improves the agreed validation objective over the baseline | Dataset and slice review |
| Product benefit | Fewer missed promises without excess inspection work | Controlled intervention experiment |
| Recovery | A compatible previous release and a dependency-free fallback | Failure drill |
| Cost | Defined budget per period and per eligible decision | Measured resource use and current unit costs |

The **lab** has deliberately different, explicit settings: synthetic event
times in minutes, a 30-minute feature-age limit, a 100 ms simulated response
deadline and an 80 ms simulated candidate-work budget. Its settings are a
mechanics exercise, not a claim that the proposed production requirements pass.

**Your exercise:** define the numerator, denominator, window, exclusions and
owner for every metric. Decide whether a fallback is successful for the
availability indicator, the freshness indicator and the product task.

## 3. Architecture

~~~mermaid
flowchart TD
    Events[Versioned source events] --> Validate[Validate and deduplicate]
    Validate --> History[Historical feature snapshots]
    History --> Train[Train and select on validation]
    Train --> Evaluate[Independent evaluation and release review]
    Evaluate --> Registry[Approved model package]
    Request[Eligible request] --> Serve[Prediction service]
    Validate --> Online[Available online features]
    Online --> Serve
    Registry --> Serve
    Serve --> Action[Response and dispatcher action]
    Serve --> Log[Prediction and exposure log]
    Outcome[Delayed delivery outcomes] --> Join[Join mature labels]
    Log --> Join
    Join --> Monitor[Service, data and quality review]
    Monitor --> Gate[Retraining investigation and gates]
    Gate --> Train
~~~

Label the diagram with deployment boundaries, storage locations and owners.
The registry supplies a loaded release; it is not a per-request model-selection
service. The outcome path must retain prediction identity and label availability.

**Your exercise:** adapt the diagram to batch serving. Include a result snapshot,
atomic publication, expiry and a policy for entities absent from the batch.

## 4. Data and interface contracts

| Artifact | Required contract |
|---|---|
| Source event | Event ID, entity, event time, availability time, schema version, units, revision |
| Feature | Entity, definition version, value, event time, materialization/availability time, maximum age |
| Training row | Prediction ID/time, eligible features, target definition, label availability, split |
| Model package | Model, transforms, feature contract, threshold, release identity, validation decision |
| Prediction log | Prediction/request ID, assigned and served version, score, threshold, action, feature metadata, fallback |
| Outcome | Prediction join key, target value, outcome time, availability time, correction version |

For every field, document null handling, valid ranges, ownership and retention.
Event IDs make retries deduplicable; correction versions need their own ordering.
Protect live output from a partially completed backfill.

### Timeline exercise

| Feature | Event time | First available | Value |
|---|---|---|---|
| A | 09:50 | 09:52 | 0.2 |
| B | 09:58 | 10:05 | 0.9 |
| C | 10:02 | 10:03 | 0.4 |

For a prediction at 10:00 and maximum feature age 15 minutes:

1. Choose the available feature.
2. Explain why B is ineligible despite its earlier event time.
3. Repeat with a maximum age of 5 minutes.

**Worked answer:** A is eligible at 10:00 under the 15-minute policy. B was
unavailable and C had not occurred. Under the 5-minute policy there is no eligible
feature, so the declared missing/stale policy applies.

## 5. Dataset and label design

Document:

- The unit of independence and intended generalization: future deliveries,
  unseen routes, unseen customers, or a combination.
- The label definition and treatment of cancellations, revised promises and
  missing outcomes.
- Source-level split boundaries, any gaps and each historical availability cutoff.
- Training-only fitting of preprocessing and feature selection.
- Sampling policy and representativeness of validation and test populations.
- Label coverage, maturity window and differences between observed and missing cases.

**Worked example:** a day-20 training run cannot use a day-19 delivery whose
label first arrives on day 22. Excluding future dispatches does not solve this
availability problem; the label cutoff must be enforced as well.

**Your exercise:** construct five deliveries with different outcome delays.
At one declared cutoff, list which may train, which may evaluate and which
remain unlabeled. Never turn an unresolved outcome into a negative by default.

## 6. Run the executable lab

Download ml_system_design.py into a folder. It needs Python 3.10 or later and
only the standard library; skip the shared PyTorch requirements for this lab.

    python3 ml_system_design.py --output-dir runs/normal

On Windows, use python in place of python3. Training uses real gradient updates
for a logistic model on generated deliveries. The constant baseline learns
the training class frequency. Both decision thresholds are chosen on validation
data using false-positive cost 1 and false-negative cost 4.

The candidate is eligible only when validation cost improves by at least
0.005 per delivery. The fixed test period is reported separately and never
chooses a threshold or candidate.

The generator produces 2,100 delivery IDs. Prediction time is index × 10
synthetic minutes:

| Partition | Indices, start inclusive / stop exclusive | Purpose |
|---|---|---|
| Training | 0 / 900 | Fit statistics and model |
| Validation | 1000 / 1250 | Choose thresholds and eligibility |
| Test | 1350 / 1600 | Independent fixed-policy report |
| Replay | 1700 / 2100 | 400 simulated requests with delayed outcomes |

Gaps are longer than the generated label delay. The lab asserts that labels in
one development/evaluation partition are available before the next partition
begins. Its fixed generator seeds preserve source identities across scenarios.

### Artifacts

| File | Inspect for |
|---|---|
| model.json | Coefficients, training-only means/scales, thresholds, feature contract, validation gate and content-derived model versions |
| manifest.json | Data/split identity, scenario settings, script and model checksums |
| predictions.jsonl | One JSON object per prediction, with assignment, served version, action, features and fallback reason |
| report.json | Baseline/candidate evaluation, rollout events, label coverage, mature quality and invariant checks |

Rerunning in the same output directory replaces these files. Use a new directory
for each comparison you intend to retain.

### Reuse the same release in a fresh process

    python3 ml_system_design.py --load-model runs/normal/model.json --output-dir runs/replay

This skips training and threshold selection. Verify identical validation and
test reports, model contents and normal prediction logs. In the log, reconstruct
one candidate score from distance, congestion, rain, saved means/scales and
coefficients. Apply the saved threshold and compare the action.

The model package supports inference reproduction. It is not an exact-resume
optimizer checkpoint. Source data are synthetic, no web service is started and
no real notifications or external actions occur.

## 7. Evaluation and release decision

Complete this table from your run:

| Evidence | Baseline | Candidate |
|---|---|---|
| Validation cohort size | | |
| Validation FP / FN | | |
| Validation cost per delivery | | |
| Selected threshold | | |
| Test cohort size | | |
| Test precision / recall | | |
| Test cost per delivery | | |
| Log loss | | |

Compute the declared cost:

    cost_per_delivery = (false_positives + 4 * false_negatives) / cohort_size

A lower error cost is not measured business savings. The constant baseline
may select every delivery because missing a delay has a larger declared cost.
Discuss whether this action policy is acceptable for a real inspection workflow.

**Your exercise:** propose a minimum-recall or top-k policy, specify how it
will be selected, and explain which comparisons must be repeated. Keep the
test period outside that selection process.

For an online product experiment, define the assignment unit, primary outcome,
guardrails, expected interference, label window and decision procedure. A model
that predicts well does not establish that the resulting intervention helps.

## 8. Serving and capacity decision record

| Option | Useful when | Main design burden |
|---|---|---|
| Batch predictions | A known population can wait until publication | Snapshot completeness, expiry and missing entities |
| Online inference | Decisions need request-specific or fresh information | Tail latency, dependencies, overload and fallback |
| Streaming | Continuous events require timely state or decisions | Ordering, lateness, lag, state recovery and replay |
| Edge | Network independence or local processing is important | Device budgets, compatible updates and disconnected behavior |

**Worked capacity arithmetic:** four slots per replica with 100 ms mean
occupancy gives an idealized 40 requests/second. At 70% utilization, plan
28 requests/second. A 1,000 requests/second peak needs ceiling(1000/28) = 36
replicas. One spare gives 37; after one fails, 36 remain under this model.

List the assumptions that can invalidate the estimate: shared dependencies,
nonlinear concurrency, long-tail requests, bursts, cold starts and uneven load.
One spare replica does not establish zone-level resilience. Measure end-to-end
latency; adding component p99s does not generally give the request p99.

**Your exercise:** use the in-course calculator, record two alternative
configurations, and write the load test that would validate the preferred one.
Include payload distribution, arrival pattern, warmup, duration and failure cases.

## 9. Failure and rollout experiments

Reuse the normal package so the model and thresholds remain fixed.

    python3 ml_system_design.py --load-model runs/normal/model.json --candidate-latency-ms 300 --output-dir runs/slow
    python3 ml_system_design.py --load-model runs/normal/model.json --feature-outage --output-dir runs/outage
    python3 ml_system_design.py --load-model runs/normal/model.json --distance-shift-km 20 --output-dir runs/shift
    python3 ml_system_design.py --load-model runs/normal/model.json --canary-percent 0 --output-dir runs/baseline-only

Canary assignment uses a stable hash of request identity. A 20% setting means
20 of 100 hash buckets, so a finite sample need not contain exactly 20%
candidate assignments. Real systems must choose an assignment unit appropriate
for repeated users, dispatchers and interference between actions.

| Scenario | Inspect | Expected interpretation |
|---|---|---|
| Normal | Assignment, served versions, fallback reasons | Missing/stale inputs may still use the baseline |
| Slow candidate | Candidate timeouts and rollback | Stop candidate exposure after the declared technical gate trips |
| Feature outage | Dependency fallback count during the middle third | The baseline needs no feature lookup |
| Distance shift | Input-shift signal, label coverage and quality | Changed inputs alone do not prove quality loss |
| Zero canary | All assigned versions | No candidate receives traffic |

The technical rollback gate requires at least ten candidate attempts and a
timeout fraction above 10%. This is a teaching rule, not a statistically powered
quality test. Candidate work is capped at a simulated 80 ms, leaving 5 ms for
fallback inside a 100 ms response deadline. A rollout can therefore be unhealthy
while aggregate deadline-success remains perfect.

The distance shift changes live inputs and their generated outcomes together;
training, validation and test periods remain unchanged. It illustrates covariate
shift under the same synthetic conditional outcome mechanism.

**Your exercise:** locate the exact request that causes rollback. Verify that
later requests stop being assigned to the candidate. Explain why a previous
model would not rescue an incompatible feature-schema migration on its own.

## 10. Monitoring and feedback

| Layer | Example indicators | First response |
|---|---|---|
| Service | Deadline failures, per-version latency, load, queue age | Inspect the affected request path and dependencies |
| Data | Freshness, missing fields, invalid units, distribution shifts | Check ingestion, materialization and source contracts |
| Prediction | Score distribution, decision rate, fallback reason | Inspect policy, model version and feature availability |
| Outcome | Coverage, label delay, mature quality, slice counts | Validate the join and compare comparable cohorts |

The report provides quality by **assigned** and **served** version. A candidate
request that falls back remains in the assigned-candidate group but is served
by the baseline. Served-version comparisons select on that behavior; do not
mistake them for an unbiased comparison of treatment groups.

Generated positive outcomes take longer to become available. The recent observed
subset can therefore be biased. Compare it with the mature cohort, whose
prediction times precede the evaluation cutoff by the maximum label delay.
Keep their counts visible.

**Your exercise:** specify an owner, threshold, window and diagnostic action
for five alerts. Include the failure of the monitoring pipeline itself.

Retraining is a separate decision from promotion. Describe a data-quality and
label-maturity gate, a training trigger, validation criteria, release review
and staged exposure. Record interventions and exposure in a real system:
dispatcher actions could change the very outcomes later used as labels.

## 11. Incident and recovery runbook

Fill one copy for a dependency outage and one for an incompatible release:

1. **Trigger:** the indicator and population that crossed its threshold.
2. **Impact:** affected users, decisions and time window.
3. **First checks:** version, recent changes, dependencies, freshness and coverage.
4. **Mitigation:** stop exposure, invoke fallback or pause the affected action.
5. **Owner:** who can make the change and who needs the result.
6. **Evidence:** prediction IDs, release, bounded logs and scenario configuration.
7. **Recovery criteria:** restored contracts, freshness, successful probes and monitored reintroduction.
8. **Follow-up:** contributing causes, corrective work and a test or drill.

**Worked outage response:** feature lookups become unavailable. Keep the
dependency-free baseline accessible, measure fallback exposure, restore the
feature source, verify materialization freshness, then resume candidate
exposure under the original gates. Do not train on an outage-produced dataset
without reviewing whether its inputs still represent the task.

## 12. Access, retention and lifecycle ownership

List who can read raw events, publish feature data, train, approve releases and
change active serving versions. Validate input shape and authorized scope at
interfaces. Document retention and correction/deletion propagation for each
artifact. Keep secrets and unnecessary raw payloads out of model packages.

Checksums identify content changes; they are not publisher authentication.
Pseudonymous IDs can remain linkable. Set ownership and access according to the
actual product and data context.

**Your exercise:** identify two hidden consumers of a feature and describe how
to retire that feature without breaking rollback or downstream reporting.

## 13. Transfer the design

Choose one extension and revise the contracts:

- **Recommendations:** candidate generation, ranking, index/encoder versions,
  exposure logging, cold starts and retrieval quality.
- **Forecasting:** creation time, horizon, rolling evaluation, revised records,
  seasonal baselines and uncertainty.
- **LLM applications:** model/prompt/index versions, document permissions,
  retrieval evaluation, tool authorization and side-effect idempotency.

Write a decision record with the requirement, alternatives, selected option,
evidence, trade-offs and a condition for reconsidering it.

## 14. Capstone submission and self-review

Submit:

1. A completed product brief and measurable requirements.
2. A three-path architecture with contracts and owners.
3. Dataset, feature-availability and label-maturity examples.
4. A release package and validation/test report.
5. Capacity estimates and a proposed real load test.
6. A normal replay and at least two failure scenarios.
7. Monitoring, retraining and incident runbooks.
8. A decision record and a list of unresolved assumptions.

Score each item **0 = missing**, **1 = described**, **2 = supported by an
artifact, calculation or executable check**:

| Review criterion | Score | Evidence or next step |
|---|---|---|
| Output connects to a defined action and product outcome | | |
| Historical features and labels respect availability | | |
| Model, transforms and threshold reproduce together | | |
| Quality selection is independent of the test period | | |
| Serving choices follow freshness and latency requirements | | |
| Capacity arithmetic declares assumptions and failure limits | | |
| A candidate regression and a dependency outage have observable responses | | |
| Monitoring distinguishes drift, missing labels and quality loss | | |
| Retraining and promotion have separate gates | | |
| Ownership, data access, retention and recovery are defined | | |
| Conclusions distinguish prototype evidence from unmeasured production claims | | |

Passing knowledge checks does not complete these practical requirements.
The prototype does not measure real throughput, cloud cost or intervention
benefit. Finish with the specific load test, data review and product experiment
that would resolve those gaps.

## Primary references

- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Feast: point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)
- [Apache Beam model](https://beam.apache.org/documentation/basics/)
- [Implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [Canarying releases](https://sre.google/workbook/canarying-releases/)
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
