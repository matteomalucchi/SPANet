# Hyperparameter Optimization

SPANet ships with two hyperparameter search scripts:

| Script            | Backend                    | Extra dependencies                            |
|-------------------|----------------------------|-----------------------------------------------|
| `spanet.tune`     | [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) | `ray[tune]`, `ray[train]`, `hyperopt` |
| `spanet.optimize` | Plain Python / PyTorch Lightning | none (`optuna` only for bayesian search) |

`spanet.optimize` exists because Ray is a heavy dependency that is awkward to
install on some clusters and laptops. It offers the same workflow as
`spanet.tune` -- sample configurations, train each of them for a few epochs,
kill the unpromising ones early, report the best -- but runs the trials
sequentially inside a single process and only relies on packages that SPANet
already needs.

## Quick start

```bash
# 20 random configurations, 10 epochs each, on the first 10% of the dataset.
python -m spanet.optimize options_files/full_hadronic_ttbar/example.json \
    -t 20 -e 10 -p 10 -g 1 \
    -l spanet_output -n my_first_study
```

The best configuration is written to `spanet_output/my_first_study/best_options.json`
as a complete options file, so a full training run of the winner is simply:

```bash
python -m spanet.train -of spanet_output/my_first_study/best_options.json --gpus 1
```

Run `python -m spanet.optimize --help` for the full list of arguments.

## Search algorithms

Selected with `-a / --algorithm`:

* `random` (default) -- uniform random search. Reliable, embarrassingly simple,
  no extra dependencies.
* `grid` -- exhaustive search. Every parameter of the search space must be
  discrete (`choice`, `randint`, `quniform` or a constant).
* `optuna` -- bayesian optimization with a Tree-structured Parzen Estimator.
  Requires `pip install optuna`. This is the closest equivalent to the
  `HyperOptSearch` used by `spanet.tune`.

## Early stopping of bad trials

Selected with `-P / --pruner`:

* `median` (default) -- the median stopping rule. After each validation epoch a
  trial is killed if its best value so far is worse than the median of the best
  values of all previously completed trials at the same epoch. This plays the
  same role as the `ASHAScheduler` of the Ray version.
* `hyperband` -- successive halving, only available with `--algorithm optuna`.
* `none` -- every trial runs for the full `--num_epochs`.

`--min_trials` controls how many trials must finish before pruning starts, and
`--warmup_epochs` how many epochs a trial is always allowed to run.

## Defining the search space

Without `-sf / --search_space_file` the built-in search space is used, which is
the same one as in `spanet.tune`. To customize it, pass a JSON file whose keys
are entries of the [options file](Options.md):

```json
{
    "hidden_dim": {"type": "choice", "values": [32, 64, 96, 128]},
    "num_encoder_layers": {"type": "randint", "low": 1, "high": 7},
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
    "focal_gamma": {"type": "uniform", "low": 0.0, "high": 1.0},
    "dropout": {"type": "quniform", "low": 0.0, "high": 0.3, "q": 0.05},
    "linear_block_type": ["GRU", "Gated", "Resnet"],
    "batch_size": 1024
}
```

A complete example is available in
[`options_files/search_spaces/example.json`](../options_files/search_spaces/example.json).

### Distributions

| Type         | Parameters        | Description                                             |
|--------------|-------------------|---------------------------------------------------------|
| `choice`     | `values`          | Uniform choice out of a list.                            |
| `uniform`    | `low`, `high`     | Continuous uniform in `[low, high]`.                     |
| `loguniform` | `low`, `high`     | Log-uniform in `[low, high]`, both bounds positive.      |
| `quniform`   | `low`, `high`, `q`| Uniform in `[low, high]` quantized to multiples of `q`.  |
| `randint`    | `low`, `high`     | Integer in `[low, high)`, as in `ray.tune.randint`.      |
| `constant`   | `value`           | Fixed value, overrides the base options file.            |

Two shorthands are accepted: a bare JSON list is a `choice`, and any other bare
JSON value is a `constant`.

### Ray-style search spaces

Search space files written for `spanet.tune` also work here: strings such as
`"tune.loguniform(1e-5, 1e-1)"` or `"tune.choice([32, 64])"` are parsed into the
equivalent distribution. Unlike `spanet.tune`, they are parsed rather than
`eval`-ed, so only literal arguments are allowed.

## Output

Everything is written to `{log_dir}/{name}`:

| File / directory    | Contents                                                              |
|---------------------|-----------------------------------------------------------------------|
| `study.json`        | Full state of the study: one entry per trial with its metric history.  |
| `results.csv`       | Flat table of every trial, its status, its value and its parameters.   |
| `search_space.json` | The search space that was actually used.                               |
| `best_config.json`  | Only the sampled hyperparameters of the best trial.                    |
| `best_options.json` | Complete options file for the best trial, ready for `spanet.train`.    |
| `optuna.db`         | Optuna study state, only with `--algorithm optuna`.                    |
| `trial_XXXX/`       | Per-trial options, config, metric history and TensorBoard logs.        |

`study.json`, `results.csv` and `best_options.json` are rewritten after every
trial, so a study can be inspected -- or interrupted with `Ctrl-C` -- at any
point without losing results.

## Resuming a study

```bash
python -m spanet.optimize options_files/ttH.json -t 20 -n my_first_study --resume
```

`--resume` appends new trials to the existing study, restores the state of the
pruner, and, for `--algorithm optuna`, reloads the sampler state from
`optuna.db` so the bayesian search keeps learning from the previous trials.
Without `--resume` an existing study directory is overwritten.

## Choosing the objective

By default the metric is taken from the `metric` entry of the base options file
(`validation_accuracy` unless it is set to something else), and it is maximized.
Both can be overridden:

```bash
python -m spanet.optimize options_files/ttH.json -mt "loss/total_loss" -md min
```

Any value logged by the model may be used, for example
`validation_average_jet_accuracy` or `CLASSIFICATION/{key}_accuracy`. See
[TrainingMetrics.md](TrainingMetrics.md). If the metric is missing, the trial
fails with an error listing every metric the model actually logged.

## Practical notes

* **Keep the trials short.** A search only needs to rank configurations, not to
  train them to convergence. `-e 10` epochs together with `-p 10` (10% of the
  dataset) is a good starting point.
* **One device per trial.** Trials run one after another on a single device. A
  multi-device trial would make Lightning re-launch the script in DDP
  subprocesses, and each subprocess would start its own copy of the study. To
  use several GPUs, launch one study per GPU with different
  `CUDA_VISIBLE_DEVICES` and different `--name`.
* **Failing trials do not stop the search.** A configuration that runs out of
  memory or is otherwise invalid is recorded with status `failed` and the study
  moves on to the next trial.
* **Checkpoints are not saved by default** to keep the study directory small.
  Pass `--save_checkpoints` to keep the best checkpoint of every trial.
* **Budget the search** with `--timeout` (total seconds for the study) and
  `--time_limit DD:HH:MM:SS` (wall-clock limit for a single trial).
* **Preview a search** with `--dry_run`, which prints the configurations that
  would be trained without training anything.
