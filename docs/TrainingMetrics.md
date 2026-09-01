# Training Metrics

During training SPANet writes [TensorBoard](https://www.tensorflow.org/tensorboard)
logs so you can monitor optimization and reconstruction performance in real time.
As described in the [`ttbar` guide](./TTBar.md), the logs are written under the
output directory (e.g. `spanet_output/version_N`) and can be viewed with:

```bash
tensorboard --logdir spanet_output
```

then navigating to `localhost:6006` in a browser.

This page documents **every scalar that is logged**, how it is computed, and — where
the computation is not obvious from the name — the exact formula. It is meant as a
reference for interpreting the TensorBoard curves.

## How the metrics are produced

The metrics come from three places in the code:

- **Losses** are logged from
  [`jet_reconstruction_training.py`](../spanet/network/jet_reconstruction/jet_reconstruction_training.py)
  during the training step (tags prefixed with `loss/`) and from the mirrored
  validation routine in
  [`jet_reconstruction_validation.py`](../spanet/network/jet_reconstruction/jet_reconstruction_validation.py)
  (tags prefixed with `validation_loss/`). In the combined plots the **solid** line
  is the training value and the **dashed** line is the validation value.
- **Accuracy, purity, regression and classification metrics** are computed once per
  validation epoch in
  [`jet_reconstruction_validation.py`](../spanet/network/jet_reconstruction/jet_reconstruction_validation.py)
  (`validation_step` → `compute_metrics`) and the purity breakdown in the
  [`SymmetricEvaluator`](../spanet/dataset/evaluator.py). They are logged with
  `on_epoch=True`, so the plotted value is the epoch average over all validation
  batches.
- **The learning rate** is logged automatically by PyTorch-Lightning's
  `LearningRateMonitor` callback, which is registered in
  [`train.py`](../spanet/train.py).

A few notes that apply throughout:

- Many terms are only logged when their corresponding task is enabled. This is
  controlled by the `*_loss_scale` options (`assignment_loss_scale`,
  `detection_loss_scale`, `kl_loss_scale`, `regression_loss_scale`,
  `classification_loss_scale`, `mdmm_loss_scale`) in the
  [options file](./Options.md). A scale of `0` disables both the loss term and its
  logging.
- Every logged loss term is already multiplied by its `*_loss_scale` factor.
- Tags that contain `{particle}`, `{key}` or `{eventCategory}` produce **one curve
  per** target particle, per regression/classification key, or per event-multiplicity
  category, respectively.
- TensorBoard groups tags into sections by the text before the first `/`, so e.g.
  all `loss/…` tags appear together, all `jet/…` tags appear together, and so on.
  The **File name** column below is that grouping; the **TensorBoard tag(s)** column
  gives the exact scalar names.

## Losses

All loss scalars are logged from `training_step` (prefix `loss/`) and its validation
twin `compute_validation_losses` (prefix `validation_loss/`).

| File name | TensorBoard tag(s) | Detailed explanation & formula |
|---|---|---|
| `total_loss` | `loss/total_loss`, `validation_loss/total_loss` | Sum of all **enabled** loss terms per optimizer step. The training step builds a list of the enabled terms (detection, symmetric/KL, regression, classification, assignment — each already multiplied by its `*_loss_scale`), concatenates them and logs the **sum**: `total_loss = Σ_k λ_k L_k`. (The value returned for back-propagation is the *mean* of the concatenated vector, but the logged scalar is the sum.) Validation logs the same sum over the enabled terms. |
| `assignment_loss` | `loss/{particle}/assignment_loss`, `validation_loss/{particle}/assignment_loss` | Per-particle jet-assignment loss — a **focal categorical cross-entropy** on the assignment log-probability of the *true* jet tuple. For each particle: `L_assign = -w · (1 - p_t)^γ · log p_t`, where `p_t = exp(log P[true jets])` is the probability the model assigns to the correct jet combination, `γ = focal_gamma`, and `w` is the balancing weight. Absent particles are masked to 0. It is summed over the particles in an event, divided by the number of present particles (or the event-weight denominator when `balance_events`), then scaled by `assignment_loss_scale`. One tag is logged **per particle/target** name. |
| `detection_loss` | `loss/{particle}/detection_loss`, `validation_loss/{particle}/detection_loss` | Per-particle particle-detection loss — **binary cross-entropy with logits** between the detection logit and the particle-present mask: `L_det = BCE_logits(detection, mask)`, weighted, averaged over present particles, and scaled by `detection_loss_scale`. One tag per particle. |
| `symmetric_loss` | `loss/symmetric_loss`, `validation_loss/symmetric_loss` | Symmetry penalty built from the **Jensen–Shannon divergence** between symmetric assignment distributions. For every event transposition pair `(i, j)`: `div = JSD(P_i, P_j)` and `ℓ = exp(-div²)` (masked to 0 when either particle is absent). These are averaged over pairs, then `L = Σ(w · ℓ) / Σ(mask)`, scaled by `kl_loss_scale`. The JSD itself is `JSD(P, Q) = ½·KL(P‖M) + ½·KL(Q‖M)` with `M = ½(P + Q)`. **Note:** the logged number is the `exp(-JSD²)` penalty (small ⇒ well-separated distributions), not the raw JSD. Only logged when there is ≥1 event transposition and `kl_loss_scale > 0`. |
| `regression_loss` | `loss/regression/{key}`, `validation_loss/regression/{key}` | Auxiliary regression loss, one tag per regression `key`, logged only if `regression_loss_scale > 0`. The functional form depends on the regression type: **gaussian** → `((ŷ - y) / σ)²`; **laplacian** → `\|(ŷ - y) / σ\|`; **log-gaussian** → normal negative-log-likelihood in log-space. Targets are normalized by the per-key `mean`/`std`, `NaN` targets are masked out, the per-key mean is taken, and the result is scaled by `regression_loss_scale`. |
| `classification_loss` | `loss/classification/{key}`, `validation_loss/classification/{key}` | Auxiliary classification loss, one tag per classification `key`, logged only if `classification_loss_scale > 0`. Standard **cross-entropy** `L = -Σ_c y_c log ŷ_c` (with `ignore_index = -1`), optionally class-balanced via `classification_weights`. When `balance_events` is set it becomes an event-weighted mean `Σ_n L_n · w_n / Σ_m w_m`. Scaled by `classification_loss_scale`. |
| `total_loss_no_mdmm` | `loss/total_loss_no_mdmm` | Only emitted when `mdmm_loss_scale > 0`. The MDMM (Modified Differential Method of Multipliers) path logs **two** totals: `total_loss` is the constrained objective returned by the MDMM module, while `total_loss_no_mdmm` is the plain mean of the enabled loss terms *before* the constraint machinery. Comparing them shows how much the constraints are perturbing the raw objective. |
| `weights` | `weights/{i}` | Only when `balance_losses` (GradNorm-style adaptive loss balancing) is active. Logs the softmax loss-balancing weight for each of the `num_losses` task terms, `a_i = softmax(logits)_i`, which are learned so the gradient magnitudes across tasks are equalized. One tag per loss index `i`. |

## Accuracy and purity metrics (validation)

Computed in `validation_step` → `compute_metrics` and logged with `on_epoch=True`, so
each point is an epoch average.

| File name | TensorBoard tag(s) | Detailed explanation & formula |
|---|---|---|
| `validation_accuracy` | `validation_accuracy`, `validation_average_jet_accuracy` | Two whole-event summary scores. **`validation_accuracy`** is the fraction of events in which *all* target particles are fully and correctly reconstructed; it is aliased to `jet/accuracy_{N}_of_{N}` (full event, `N` = number of targets). **`validation_average_jet_accuracy`** is the weight-normalized mean of the per-target reconstruction rate, `mean( weighted correct-jet count / Σ target weights )` over events that have targets. `validation_accuracy` is the quantity used for early stopping, learning-rate scheduling, and hyper-parameter optimization. |
| `jet_accuracy` | `jet/accuracy_{i}_of_{j}` | Event-completeness grid for the **assignment** head. For events containing exactly `j` target particles, the fraction whose number of fully-correctly-assigned particles is `≥ i`: `acc_{i/j} = mean( c ≥ i \| #particles = j )`, where `c` counts the particles for which *every* assigned jet index matches the target (`np.all(prediction == target)`), taken as the best over the event's symmetry permutations. One tag for each valid pair `1 ≤ i ≤ j ≤ N`. |
| `particle_accuracy` | `particle/accuracy_{i}_of_{j}` | The same completeness grid, but for the **detection** head instead of assignment: for events with `j` particles, the fraction where `≥ i` particle-presence predictions match the mask. A prediction is "present" when `particle_score ≥ 0.5`, and correctness is `mask == (score ≥ 0.5)`. One tag per `(i, j)`. |
| `particle_detection_metrics` | `particle/accuracy`, `particle/sensitivity`, `particle/specificity`, `particle/f_score` | Global (flattened over all particles and events) binary-classification quality of the detection head, computed with scikit-learn: **accuracy** = fraction of correct present/absent calls; **sensitivity** = recall on the "present" class `TP / (TP + FN)`; **specificity** = recall on the "absent" class `TN / (TN + FP)`; **f_score** = `F₁ = 2·precision·recall / (precision + recall)`. All use a 0.5 threshold on `particle_scores`. |
| `purity` | `Purity/{eventCategory}/event_purity`, `Purity/{eventCategory}/event_proportion`, `Purity/{eventCategory}/{cluster}_purity` | Symmetry-aware reconstruction purity from `SymmetricEvaluator.full_report`, broken down by event category (particle-multiplicity pattern, e.g. `2b1t`, with `*` denoting a wildcard count). **`event_purity`** = fraction of events in that category where *all* particles are correct, maximized over the event symmetry group. **`{cluster}_purity`** = per-symmetry-cluster jet purity `Σ(best correct assignments in cluster) / Σ(cluster masks)`, maximized over intra-cluster permutations. **`event_proportion`** = fraction of the validation set falling in that category (`event_mask.mean()`). Targets and predictions are pre-sorted within symmetry orbits to remove trivial permutation ambiguity. |

## Regression and classification breakdowns (validation)

| File name | TensorBoard tag(s) | Detailed explanation & formula |
|---|---|---|
| `regression_{key}` | `REGRESSION/{key}_percent_error`, `REGRESSION/{key}_absolute_error` (scalars); `REGRESSION/{key}_percent_deviation`, `REGRESSION/{key}_absolute_deviation` (histograms) | Per-regression-target error, with `δ = ŷ - y`. **percent_error** = `mean \|δ / y\|` (mean absolute *relative* error); **absolute_error** = `mean \|δ\|`. In addition, two **histograms** of the *signed* deviations are logged each step via `add_histogram`: **percent_deviation** = `δ / y` and **absolute_deviation** = `δ`, so you can inspect bias and spread rather than just the mean. |
| `classification_{key}` | `CLASSIFICATION/{key}_accuracy`, `…_accuracy_target0`, `…_accuracy_target1`, `…_accuracy_event_weight`, `…_accuracy_event_weight_target0/1`, `…_accuracy_event_and_class_weight`, `…_accuracy_event_and_class_weight_target0/1` | Per-classification-key accuracy in three weighting schemes × three populations (**9 tags per key**). *Unweighted:* `accuracy` = overall `mean(ĉ = c)`; `accuracy_target0/1` = accuracy restricted to true-class-0 / true-class-1 samples. *Event-weighted:* `accuracy_event_weight[_target0/1]` = `Σ (ĉ=c)·w_evt / Σ w_evt`, using the per-sample physics event weights. *Event × class weighted:* `accuracy_event_and_class_weight[_target0/1]` = the same but with `w_evt · w_class`, where `w_class` is the per-class balancing weight. |

## Optimizer

| File name | TensorBoard tag(s) | Detailed explanation & formula |
|---|---|---|
| `learning_rate` | `lr-Adam` (name set by the callback) | Logged automatically by PyTorch-Lightning's `LearningRateMonitor` callback (registered in `train.py`), **once per step** (`interval: 'step'`). The schedule is either a linear warmup→decay (`get_linear_schedule_with_warmup`) when `learning_rate_cycles < 1`, or cosine-with-hard-restarts (`get_cosine_with_hard_restarts_schedule_with_warmup`) otherwise, with `learning_rate_warmup_epochs` of warmup ramping from 0 up to `learning_rate`. |
