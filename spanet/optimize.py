"""
Hyperparameter optimization for SPANet -- Ray-free alternative to ``spanet/tune.py``.

This script provides the same functionality as the Ray Tune based ``spanet.tune``
(random / bayesian search over the SPANet ``Options`` with early stopping of bad
trials) but only depends on packages that are already required to train SPANet.
Trials are executed sequentially, in-process, one PyTorch-Lightning ``Trainer``
per trial.

Search algorithms
-----------------
``random``  Uniform random search over the search space. No extra dependencies.
``grid``    Exhaustive search over a fully discrete search space. No extra dependencies.
``optuna``  Tree-structured Parzen Estimator (bayesian). Requires ``pip install optuna``.

Early stopping of unpromising trials
------------------------------------
``median``  Median stopping rule: a trial is killed at the end of a validation
            epoch if its best value so far is worse than the median of the best
            values of all previously completed trials at the same epoch.
``none``    Every trial runs for the full number of epochs.

Examples
--------
    # Simple random search, 20 trials of 10 epochs each.
    python -m spanet.optimize options_files/ttH.json -t 20 -e 10

    # Bayesian search with a custom space, resuming a previous study.
    python -m spanet.optimize options_files/ttH.json -sf search_space.json \
        -a optuna -t 50 -e 20 -l spanet_output -n my_study --resume

    # Only print the configurations that would be trained.
    python -m spanet.optimize options_files/ttH.json -t 5 --dry_run

Outputs (all inside ``{log_dir}/{name}``)
-----------------------------------------
    study.json          Full state of the study, one entry per trial.
    results.csv         Flat table of every trial and its metric value.
    search_space.json   The search space that was actually used.
    best_config.json    The best set of sampled hyperparameters.
    best_options.json   A complete SPANet options file using the best parameters,
                        ready to be passed to ``spanet.train --options_file``.
    trial_XXX/          Per-trial directory with options, metric history and
                        TensorBoard logs.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import re
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

# NOTE: torch / pytorch_lightning / spanet are imported lazily inside
# `train_trial` so that the search-space and sampler machinery can be imported
# (and unit tested) without a full deep learning stack.


# =========================================================================================
# Search space definition
# =========================================================================================

class Distribution:
    """Base class for a single hyperparameter distribution."""

    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def grid(self) -> List[Any]:
        raise TypeError(
            f"{self.__class__.__name__} is continuous and cannot be used with --algorithm grid. "
            f"Use a 'choice' / 'randint' / 'quniform' distribution instead."
        )

    def suggest_optuna(self, trial, name: str) -> Any:
        raise NotImplementedError

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError


class Constant(Distribution):
    """A fixed value. Useful to override a base option for every trial."""

    def __init__(self, value: Any):
        self.value = value

    def sample(self, rng: random.Random) -> Any:
        return self.value

    def grid(self) -> List[Any]:
        return [self.value]

    def suggest_optuna(self, trial, name: str) -> Any:
        return self.value

    def to_json(self) -> Dict[str, Any]:
        return {"type": "constant", "value": self.value}


class Choice(Distribution):
    """Uniform choice out of a fixed list of values."""

    def __init__(self, values: Sequence[Any]):
        if len(values) == 0:
            raise ValueError("'choice' distribution requires at least one value.")
        self.values = list(values)

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def grid(self) -> List[Any]:
        return list(self.values)

    def suggest_optuna(self, trial, name: str) -> Any:
        # Optuna categorical parameters must be simple scalars.
        if all(value is None or isinstance(value, (bool, int, float, str)) for value in self.values):
            return trial.suggest_categorical(name, self.values)

        index = trial.suggest_int(f"{name}__index", 0, len(self.values) - 1)
        return self.values[index]

    def to_json(self) -> Dict[str, Any]:
        return {"type": "choice", "values": self.values}


class Uniform(Distribution):
    """Continuous uniform distribution in [low, high]."""

    def __init__(self, low: float, high: float):
        if high < low:
            raise ValueError(f"'uniform' requires low <= high, got low={low}, high={high}.")
        self.low = float(low)
        self.high = float(high)

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def suggest_optuna(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high)

    def to_json(self) -> Dict[str, Any]:
        return {"type": "uniform", "low": self.low, "high": self.high}


class LogUniform(Distribution):
    """Log-uniform distribution in [low, high]. Both bounds must be positive."""

    def __init__(self, low: float, high: float):
        if low <= 0 or high <= 0:
            raise ValueError(f"'loguniform' requires strictly positive bounds, got low={low}, high={high}.")
        if high < low:
            raise ValueError(f"'loguniform' requires low <= high, got low={low}, high={high}.")
        self.low = float(low)
        self.high = float(high)

    def sample(self, rng: random.Random) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))

    def suggest_optuna(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=True)

    def to_json(self) -> Dict[str, Any]:
        return {"type": "loguniform", "low": self.low, "high": self.high}


class QUniform(Distribution):
    """Uniform distribution in [low, high] quantized to multiples of ``q``."""

    def __init__(self, low: float, high: float, q: float):
        if q <= 0:
            raise ValueError(f"'quniform' requires a strictly positive q, got q={q}.")
        if high < low:
            raise ValueError(f"'quniform' requires low <= high, got low={low}, high={high}.")
        self.low = float(low)
        self.high = float(high)
        self.q = float(q)

    def _quantize(self, value: float) -> float:
        quantized = self.low + round((value - self.low) / self.q) * self.q
        # Guard against floating point drift pushing us outside of the bounds.
        quantized = min(max(quantized, self.low), self.high)
        return round(quantized, 12)

    def sample(self, rng: random.Random) -> float:
        return self._quantize(rng.uniform(self.low, self.high))

    def grid(self) -> List[float]:
        num_steps = int(math.floor((self.high - self.low) / self.q + 1e-9))
        return [self._quantize(self.low + i * self.q) for i in range(num_steps + 1)]

    def suggest_optuna(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, step=self.q)

    def to_json(self) -> Dict[str, Any]:
        return {"type": "quniform", "low": self.low, "high": self.high, "q": self.q}


class IntUniform(Distribution):
    """Uniform integer distribution. ``low`` is inclusive, ``high`` is exclusive (as in ray.tune.randint)."""

    def __init__(self, low: int, high: int):
        low, high = int(low), int(high)
        if high <= low:
            raise ValueError(f"'randint' requires low < high, got low={low}, high={high}.")
        self.low = low
        self.high = high

    def sample(self, rng: random.Random) -> int:
        return rng.randrange(self.low, self.high)

    def grid(self) -> List[int]:
        return list(range(self.low, self.high))

    def suggest_optuna(self, trial, name: str) -> int:
        return trial.suggest_int(name, self.low, self.high - 1)

    def to_json(self) -> Dict[str, Any]:
        return {"type": "randint", "low": self.low, "high": self.high}


# Default search space, equivalent to the one in spanet/tune.py.
DEFAULT_SEARCH_SPACE: Dict[str, Distribution] = {
    "hidden_dim": Choice([32, 64, 96, 128]),

    "num_encoder_layers": Choice([1, 2, 3, 4, 5, 6]),
    "num_branch_embedding_layers": Choice([1, 2, 4, 6]),
    "num_branch_encoder_layers": Choice([1, 2, 4, 6]),

    "num_regression_layers": Choice([1, 2, 4, 6]),
    "num_classification_layers": Choice([1, 2, 4, 6]),

    "learning_rate": LogUniform(1e-5, 1e-1),
    "focal_gamma": Uniform(0.0, 1.0),
    "l2_penalty": LogUniform(1e-6, 1e-2),
}


_DISTRIBUTION_BUILDERS = {
    "constant": lambda spec: Constant(spec["value"]),
    "choice": lambda spec: Choice(spec.get("values", spec.get("value"))),
    "grid_search": lambda spec: Choice(spec.get("values", spec.get("value"))),
    "uniform": lambda spec: Uniform(spec["low"], spec["high"]),
    "loguniform": lambda spec: LogUniform(spec["low"], spec["high"]),
    "log_uniform": lambda spec: LogUniform(spec["low"], spec["high"]),
    "quniform": lambda spec: QUniform(spec["low"], spec["high"], spec["q"]),
    "randint": lambda spec: IntUniform(spec["low"], spec["high"]),
    "int": lambda spec: IntUniform(spec["low"], spec["high"]),
}

# ray.tune style strings, e.g. "tune.loguniform(1e-5, 1e-1)". Supported so that
# search space files written for spanet/tune.py keep working here.
_RAY_STYLE_PATTERN = re.compile(r"^\s*(?:tune\.)?(\w+)\s*\((.*)\)\s*$", re.DOTALL)

_RAY_STYLE_ALIASES = {
    "choice": lambda args, kwargs: Choice(args[0] if args else kwargs["categories"]),
    "grid_search": lambda args, kwargs: Choice(args[0] if args else kwargs["values"]),
    "uniform": lambda args, kwargs: Uniform(*args, **kwargs),
    "loguniform": lambda args, kwargs: LogUniform(*args, **kwargs),
    "quniform": lambda args, kwargs: QUniform(*args, **kwargs),
    "randint": lambda args, kwargs: IntUniform(*args, **kwargs),
}


def _parse_ray_style(expression: str) -> Optional[Distribution]:
    """Parse a ray.tune style search space string without executing arbitrary code."""
    match = _RAY_STYLE_PATTERN.match(expression)
    if match is None:
        return None

    name = match.group(1)
    if name not in _RAY_STYLE_ALIASES:
        return None

    try:
        call = ast.parse(f"_({match.group(2)})", mode="eval").body
        args = [ast.literal_eval(argument) for argument in call.args]
        kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Unable to parse search space entry '{expression}': {error}") from error

    return _RAY_STYLE_ALIASES[name](args, kwargs)


def parse_distribution(name: str, spec: Any) -> Distribution:
    """Convert a single JSON search space entry into a Distribution."""
    if isinstance(spec, Distribution):
        return spec

    if isinstance(spec, dict):
        if "type" not in spec:
            raise ValueError(f"Search space entry '{name}' is a dict but has no 'type' key.")

        distribution_type = str(spec["type"]).lower()
        if distribution_type not in _DISTRIBUTION_BUILDERS:
            raise ValueError(
                f"Unknown distribution type '{spec['type']}' for '{name}'. "
                f"Valid types: {sorted(set(_DISTRIBUTION_BUILDERS))}."
            )

        try:
            return _DISTRIBUTION_BUILDERS[distribution_type](spec)
        except KeyError as error:
            raise ValueError(f"Search space entry '{name}' is missing the {error} key.") from error

    if isinstance(spec, list):
        return Choice(spec)

    if isinstance(spec, str):
        distribution = _parse_ray_style(spec)
        if distribution is not None:
            return distribution
        return Constant(spec)

    return Constant(spec)


def load_search_space(search_space_file: Optional[str]) -> Dict[str, Distribution]:
    """Load the search space from a JSON file, falling back to DEFAULT_SEARCH_SPACE."""
    if search_space_file is None:
        return dict(DEFAULT_SEARCH_SPACE)

    with open(search_space_file, 'r') as json_file:
        raw_space = json.load(json_file)

    if not isinstance(raw_space, dict):
        raise ValueError(f"Search space file '{search_space_file}' must contain a JSON object.")

    return {name: parse_distribution(name, spec) for name, spec in raw_space.items()}


def search_space_to_json(space: Dict[str, Distribution]) -> Dict[str, Any]:
    return {name: distribution.to_json() for name, distribution in space.items()}


def search_space_size(space: Dict[str, Distribution]) -> Optional[int]:
    """Total number of grid points, or None if the space is continuous."""
    size = 1
    for distribution in space.values():
        try:
            size *= len(distribution.grid())
        except TypeError:
            return None
    return size


# =========================================================================================
# Samplers
# =========================================================================================

class Sampler:
    """Proposes configurations and (optionally) learns from finished trials."""

    supports_pruning_feedback = False

    def ask(self, trial_number: int) -> Tuple[Dict[str, Any], Any]:
        """Return the configuration for the next trial and an opaque sampler handle."""
        raise NotImplementedError

    def tell(self, handle: Any, value: Optional[float], status: str) -> None:
        """Report the outcome of a trial back to the sampler."""

    def report(self, handle: Any, step: int, value: float) -> bool:
        """Report an intermediate value. Returns True if the trial should be pruned."""
        return False

    @property
    def num_previous_trials(self) -> int:
        return 0


class RandomSampler(Sampler):
    def __init__(self, space: Dict[str, Distribution], seed: Optional[int] = None):
        self.space = space
        self.rng = random.Random(seed)

    def ask(self, trial_number: int) -> Tuple[Dict[str, Any], Any]:
        return {name: distribution.sample(self.rng) for name, distribution in self.space.items()}, None


class GridSampler(Sampler):
    def __init__(self, space: Dict[str, Distribution], shuffle: bool = False, seed: Optional[int] = None):
        self.space = space
        self.names = list(space.keys())
        self.points = [dict(zip(self.names, values)) for values in product(*(space[name].grid() for name in self.names))]

        if shuffle:
            random.Random(seed).shuffle(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def ask(self, trial_number: int) -> Tuple[Dict[str, Any], Any]:
        if trial_number >= len(self.points):
            raise IndexError("Grid search space exhausted.")
        return dict(self.points[trial_number]), None


class OptunaSampler(Sampler):
    """Bayesian optimization (TPE) backed by optuna. Optional dependency."""

    supports_pruning_feedback = True

    def __init__(
        self,
        space: Dict[str, Distribution],
        metric: str,
        mode: str,
        pruner: str,
        num_epochs: int,
        min_trials: int,
        warmup_epochs: int,
        storage: Optional[str] = None,
        study_name: str = "spanet_optimize",
        seed: Optional[int] = None,
        resume: bool = False,
    ):
        try:
            import optuna
        except ImportError as error:
            raise ImportError(
                "The 'optuna' search algorithm requires an extra dependency. "
                "Please run: pip install optuna    (or use --algorithm random)"
            ) from error

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.optuna = optuna
        self.space = space
        self.metric = metric

        if pruner == "median":
            optuna_pruner = optuna.pruners.MedianPruner(
                n_startup_trials=min_trials,
                n_warmup_steps=warmup_epochs
            )
        elif pruner == "hyperband":
            optuna_pruner = optuna.pruners.HyperbandPruner(
                min_resource=max(1, num_epochs // 4),
                max_resource=num_epochs
            )
        else:
            optuna_pruner = optuna.pruners.NopPruner()

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=resume,
            direction="maximize" if mode == "max" else "minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna_pruner
        )

    @property
    def num_previous_trials(self) -> int:
        return len(self.study.trials)

    def ask(self, trial_number: int) -> Tuple[Dict[str, Any], Any]:
        trial = self.study.ask()
        config = {name: distribution.suggest_optuna(trial, name) for name, distribution in self.space.items()}
        return config, trial

    def report(self, handle: Any, step: int, value: float) -> bool:
        handle.report(value, step)
        return handle.should_prune()

    def tell(self, handle: Any, value: Optional[float], status: str) -> None:
        if status == "complete" and value is not None:
            self.study.tell(handle, value, state=self.optuna.trial.TrialState.COMPLETE)
        elif status == "pruned":
            self.study.tell(handle, state=self.optuna.trial.TrialState.PRUNED)
        else:
            self.study.tell(handle, state=self.optuna.trial.TrialState.FAIL)


# =========================================================================================
# Median stopping rule
# =========================================================================================

class MedianPruner:
    """
    Median stopping rule, used for the random and grid search algorithms.

    A running trial is stopped at step ``s`` if its best value up to ``s`` is
    worse than the median of the best-up-to-``s`` values of all trials that have
    already completed at least ``s`` steps.
    """

    def __init__(self, mode: str = "max", min_trials: int = 3, warmup_steps: int = 1):
        self.mode = mode
        self.min_trials = max(1, min_trials)
        self.warmup_steps = max(0, warmup_steps)
        self.history: Dict[int, List[float]] = {}

    def add_trial(self, values: Sequence[float]) -> None:
        """Register the full (per-step) metric history of a finished trial."""
        best_so_far = None
        for step, value in enumerate(values, start=1):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            best_so_far = value if best_so_far is None else self._best(best_so_far, value)
            self.history.setdefault(step, []).append(best_so_far)

    def should_prune(self, step: int, best_so_far: float) -> bool:
        if step <= self.warmup_steps:
            return False

        previous = self.history.get(step, [])
        if len(previous) < self.min_trials:
            return False

        median = _median(previous)
        return best_so_far < median if self.mode == "max" else best_so_far > median

    def _best(self, left: float, right: float) -> float:
        return max(left, right) if self.mode == "max" else min(left, right)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def is_better(value: float, reference: Optional[float], mode: str) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if reference is None:
        return True
    return value > reference if mode == "max" else value < reference


def infer_mode(metric: str) -> str:
    """Losses should be minimized, everything else maximized."""
    return "min" if "loss" in metric.lower() or "error" in metric.lower() else "max"


# =========================================================================================
# Single trial
# =========================================================================================

def build_trial_options(base_options_dict: Dict[str, Any], config: Dict[str, Any], num_epochs: int, num_workers: int):
    """Create a fresh Options object for one trial. Never mutates the base options."""
    from spanet.options import Options

    options = Options()
    options.update_options(base_options_dict)
    options.update_options(config)

    # The learning rate schedule is built from options.epochs, so it has to match
    # the number of epochs the trial is actually allowed to run for.
    options.epochs = num_epochs
    options.num_dataloader_workers = num_workers
    options.verbose_output = False

    return options


def train_trial(
    options,
    trial_directory: str,
    metric: str,
    mode: str,
    num_epochs: int,
    use_gpu: bool,
    fp16: bool,
    time_limit: Optional[str],
    save_checkpoints: bool,
    on_epoch_end,
) -> List[float]:
    """
    Train a single configuration and return the metric value after every validation epoch.

    ``on_epoch_end(step, value, best_so_far)`` is called after each validation epoch
    and should return True to stop the trial early.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    from spanet import JetReconstructionModel

    class TrialReportCallback(pl.Callback):
        def __init__(self):
            self.history: List[float] = []
            self.pruned = False

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return

            value = trainer.callback_metrics.get(metric)
            if value is None:
                available = ", ".join(sorted(str(key) for key in trainer.callback_metrics))
                raise KeyError(
                    f"Metric '{metric}' was not logged by the model. Available metrics: {available}"
                )

            value = float(value)
            self.history.append(value)

            best_so_far = max(self.history) if mode == "max" else min(self.history)
            if on_epoch_end(len(self.history), value, best_so_far):
                self.pruned = True
                trainer.should_stop = True

    report_callback = TrialReportCallback()
    callbacks: List[pl.Callback] = [report_callback]

    if save_checkpoints:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(trial_directory, "checkpoints"),
            monitor=metric,
            mode=mode,
            save_top_k=1,
            save_last=False,
            # The metric name is not part of the filename since SPANet metrics may
            # contain a '/', which is not usable inside of a checkpoint name.
            filename="best-{epoch:03d}",
            auto_insert_metric_name=False
        ))

    # A single device per trial on purpose: with more than one device Lightning
    # would re-launch this script in DDP subprocesses, and every subprocess would
    # start its own copy of the whole study. To use several GPUs, run several
    # studies with different CUDA_VISIBLE_DEVICES and different --name.
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "auto",
        devices=1,
        strategy="auto",
        precision="16-mixed" if fp16 else "32-true",

        gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
        max_epochs=num_epochs,
        max_time=time_limit,

        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=save_checkpoints,
        logger=TensorBoardLogger(save_dir=trial_directory, name="", version="."),
        callbacks=callbacks
    )

    trainer.fit(JetReconstructionModel(options))

    return report_callback.history


# =========================================================================================
# Study bookkeeping
# =========================================================================================

def load_previous_study(study_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(study_file):
        return []

    with open(study_file, 'r') as json_file:
        study = json.load(json_file)

    return study.get("trials", [])


def save_study(study_file: str, header: Dict[str, Any], trials: List[Dict[str, Any]]) -> None:
    with open(study_file, 'w') as json_file:
        json.dump({**header, "trials": trials}, json_file, indent=4, default=str)


def save_results_table(results_file: str, trials: List[Dict[str, Any]]) -> None:
    if len(trials) == 0:
        return

    parameters = sorted({key for trial in trials for key in trial["config"]})
    columns = ["trial", "status", "value", "epochs", "runtime"] + parameters

    with open(results_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for trial in trials:
            writer.writerow([
                trial["trial"],
                trial["status"],
                trial.get("value"),
                len(trial.get("history", [])),
                round(trial.get("runtime", 0.0), 1),
                *(trial["config"].get(parameter) for parameter in parameters)
            ])


def format_config(config: Dict[str, Any]) -> str:
    parts = []
    for key, value in config.items():
        parts.append(f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts)


def print_summary(trials: List[Dict[str, Any]], metric: str, mode: str) -> None:
    completed = [trial for trial in trials if trial["status"] == "complete"]
    pruned = [trial for trial in trials if trial["status"] == "pruned"]
    failed = [trial for trial in trials if trial["status"] == "failed"]

    print()
    print("=" * 100)
    print(f"Finished {len(trials)} trials: {len(completed)} complete, {len(pruned)} pruned, {len(failed)} failed.")
    print("=" * 100)

    ranked = sorted(
        (trial for trial in trials if trial.get("value") is not None),
        key=lambda trial: trial["value"],
        reverse=(mode == "max")
    )

    for rank, trial in enumerate(ranked[:10], start=1):
        print(f"{rank:>3}. trial {trial['trial']:<4} {metric} = {trial['value']:.5f}  [{trial['status']}]")
        print(f"     {format_config(trial['config'])}")

    if len(ranked) == 0:
        print("No trial produced a usable metric value.")


# =========================================================================================
# Main optimization loop
# =========================================================================================

def optimize_spanet(
    base_options_file: str,
    search_space_file: Optional[str] = None,
    algorithm: str = "random",
    pruner: str = "median",
    metric: Optional[str] = None,
    mode: Optional[str] = None,
    num_trials: int = 10,
    num_epochs: int = 10,
    num_workers: int = 4,
    gpus: int = 0,
    fp16: bool = False,
    save_checkpoints: bool = False,
    batch_size: Optional[int] = None,
    limit_dataset: Optional[float] = None,
    time_limit: Optional[str] = None,
    timeout: Optional[float] = None,
    min_trials: int = 3,
    warmup_epochs: int = 1,
    random_seed: Optional[int] = None,
    event_file: Optional[str] = None,
    training_file: Optional[str] = None,
    validation_file: Optional[str] = None,
    log_dir: str = "spanet_output",
    name: str = "spanet_optimize",
    resume: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    # -------------------------------------------------------------------------------------------------------
    # Load the base options and the search space.
    # -------------------------------------------------------------------------------------------------------
    with open(base_options_file, 'r') as json_file:
        base_options_dict = json.load(json_file)

    for key, value in (
        ("event_info_file", event_file),
        ("training_file", training_file),
        ("validation_file", validation_file),
        ("batch_size", batch_size),
    ):
        if value is not None:
            base_options_dict[key] = value

    if limit_dataset is not None:
        base_options_dict["dataset_limit"] = limit_dataset / 100

    space = load_search_space(search_space_file)

    if gpus > 1:
        print(f"Warning: only one device per trial is supported, ignoring --gpus {gpus}. "
              f"Run several studies with different CUDA_VISIBLE_DEVICES to use multiple GPUs.")
        gpus = 1

    if pruner == "hyperband" and algorithm != "optuna":
        print("Warning: the 'hyperband' pruner is only available with --algorithm optuna, using 'median' instead.")
        pruner = "median"

    metric = metric if metric is not None else base_options_dict.get("metric", "validation_accuracy")
    mode = mode if mode is not None else infer_mode(metric)

    # -------------------------------------------------------------------------------------------------------
    # Prepare the output directory and restore any previous state.
    # -------------------------------------------------------------------------------------------------------
    study_directory = os.path.join(log_dir, name)
    study_file = os.path.join(study_directory, "study.json")

    trials: List[Dict[str, Any]] = load_previous_study(study_file) if resume else []
    if resume and len(trials) > 0:
        print(f"Resuming study '{name}' with {len(trials)} previous trials.")
    elif os.path.exists(study_file):
        print(f"Warning: overwriting the existing study in '{study_directory}'. Use --resume to continue it instead.")

    if not dry_run:
        os.makedirs(study_directory, exist_ok=True)
        with open(os.path.join(study_directory, "search_space.json"), 'w') as json_file:
            json.dump(search_space_to_json(space), json_file, indent=4)

    # -------------------------------------------------------------------------------------------------------
    # Build the sampler and the pruner.
    # -------------------------------------------------------------------------------------------------------
    if algorithm == "grid":
        sampler: Sampler = GridSampler(space, seed=random_seed)
        grid_size = len(sampler)
        if num_trials > grid_size:
            print(f"Grid search space only has {grid_size} points, reducing the number of trials.")
            num_trials = grid_size
    elif algorithm == "optuna":
        # Optuna keeps its own state in an sqlite database so that --resume can
        # restore the state of the TPE sampler and not just the list of trials.
        storage = None
        if not dry_run:
            database_file = os.path.abspath(os.path.join(study_directory, "optuna.db"))
            if not resume and os.path.exists(database_file):
                os.remove(database_file)
            storage = f"sqlite:///{database_file}"

        sampler = OptunaSampler(
            space,
            metric=metric,
            mode=mode,
            pruner=pruner,
            num_epochs=num_epochs,
            min_trials=min_trials,
            warmup_epochs=warmup_epochs,
            storage=storage,
            study_name=name,
            seed=random_seed,
            resume=True
        )
    elif algorithm == "random":
        sampler = RandomSampler(space, seed=random_seed)
    else:
        raise ValueError(f"Unknown search algorithm '{algorithm}'. Valid options: random, grid, optuna.")

    median_pruner = MedianPruner(mode=mode, min_trials=min_trials, warmup_steps=warmup_epochs)
    use_median_pruner = pruner == "median" and not sampler.supports_pruning_feedback

    if use_median_pruner:
        for trial in trials:
            if trial["status"] == "complete":
                median_pruner.add_trial(trial.get("history", []))

    # -------------------------------------------------------------------------------------------------------
    # Report the configuration of this study.
    # -------------------------------------------------------------------------------------------------------
    space_size = search_space_size(space)
    print("=" * 100)
    print(f"SPANet hyperparameter optimization (Ray-free)")
    print("-" * 100)
    print(f"  Base options    : {base_options_file}")
    print(f"  Output directory: {study_directory}")
    print(f"  Algorithm       : {algorithm}")
    print(f"  Pruner          : {pruner}")
    print(f"  Objective       : {mode}imize '{metric}'")
    print(f"  Trials          : {num_trials} x {num_epochs} epochs")
    print(f"  Search space    : {len(space)} parameters, {space_size if space_size else 'infinite'} configurations")
    print("=" * 100)

    if dry_run:
        for trial_number in range(num_trials):
            try:
                config, handle = sampler.ask(trial_number)
            except IndexError:
                break
            sampler.tell(handle, None, "failed")
            print(f"trial {trial_number:<4} {format_config(config)}")
        return {"trials": [], "best": None}

    _seed_everything(random_seed)

    header = {
        "name": name,
        "base_options_file": os.path.abspath(base_options_file),
        "algorithm": algorithm,
        "pruner": pruner,
        "metric": metric,
        "mode": mode,
        "num_epochs": num_epochs,
        "search_space": search_space_to_json(space),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }

    # Failed trials are never reported as the best configuration, even when they
    # crashed late enough to have produced a usable metric value.
    best_value = None
    best_trial = None
    for trial in trials:
        if trial["status"] != "failed" and is_better(trial.get("value"), best_value, mode):
            best_value, best_trial = trial["value"], trial

    # -------------------------------------------------------------------------------------------------------
    # Run the trials.
    # -------------------------------------------------------------------------------------------------------
    study_start_time = time.time()
    completed_this_run = 0

    try:
        for trial_number in range(len(trials), len(trials) + num_trials):
            if timeout is not None and (time.time() - study_start_time) > timeout:
                print(f"Reached the total time budget of {timeout:.0f}s, stopping the search.")
                break

            try:
                config, handle = sampler.ask(trial_number)
            except IndexError:
                print("Search space exhausted, stopping the search.")
                break

            trial_directory = os.path.join(study_directory, f"trial_{trial_number:04d}")
            os.makedirs(trial_directory, exist_ok=True)

            print()
            print("-" * 100)
            print(f"Trial {trial_number} ({completed_this_run + 1}/{num_trials})")
            print(f"  {format_config(config)}")
            print("-" * 100)

            options = build_trial_options(base_options_dict, config, num_epochs, num_workers)
            options.save(os.path.join(trial_directory, "options.json"))
            with open(os.path.join(trial_directory, "config.json"), 'w') as json_file:
                json.dump(config, json_file, indent=4)

            pruned = {"value": False}
            reported_values: List[float] = []

            def on_epoch_end(step: int, value: float, best_so_far: float) -> bool:
                print(f"  epoch {step:>3}: {metric} = {value:.5f} (best {best_so_far:.5f})")
                reported_values.append(value)

                if sampler.supports_pruning_feedback:
                    should_prune = pruner != "none" and sampler.report(handle, step, value)
                else:
                    should_prune = use_median_pruner and median_pruner.should_prune(step, best_so_far)

                if should_prune:
                    print(f"  pruning trial {trial_number} after epoch {step}.")
                    pruned["value"] = True

                return should_prune

            trial_start_time = time.time()
            status, value, history, error = "complete", None, [], None

            try:
                _seed_everything(random_seed, verbose=False)
                history = train_trial(
                    options=options,
                    trial_directory=trial_directory,
                    metric=metric,
                    mode=mode,
                    num_epochs=num_epochs,
                    use_gpu=gpus > 0,
                    fp16=fp16,
                    time_limit=time_limit,
                    save_checkpoints=save_checkpoints,
                    on_epoch_end=on_epoch_end
                )
            except KeyboardInterrupt:
                raise
            except Exception as exception:  # A single bad configuration must not kill the whole study.
                status, error = "failed", f"{type(exception).__name__}: {exception}"
                print(f"  trial {trial_number} failed: {error}")
                traceback.print_exc()

            # A trial that crashed mid-training still has the epochs it did finish.
            if len(history) == 0:
                history = reported_values

            usable = [entry for entry in history if not math.isnan(entry)]
            if len(usable) > 0:
                value = max(usable) if mode == "max" else min(usable)

            if status != "failed":
                status = "pruned" if pruned["value"] else "complete"
            if value is None:
                status = "failed"
                error = error or "No metric value was reported by the trial."

            record = {
                "trial": trial_number,
                "status": status,
                "value": value,
                "config": config,
                "history": history,
                "runtime": time.time() - trial_start_time,
                "directory": trial_directory,
                "error": error,
            }

            with open(os.path.join(trial_directory, "result.json"), 'w') as json_file:
                json.dump(record, json_file, indent=4, default=str)

            trials.append(record)
            completed_this_run += 1

            sampler.tell(handle, value, status)
            if use_median_pruner and status == "complete":
                median_pruner.add_trial(history)

            if status != "failed" and is_better(value, best_value, mode):
                best_value, best_trial = value, record
                print(f"  new best {metric}: {best_value:.5f}")

            header["updated"] = datetime.now().isoformat(timespec="seconds")
            save_study(study_file, header, trials)
            save_results_table(os.path.join(study_directory, "results.csv"), trials)

            if best_trial is not None:
                _save_best(study_directory, base_options_dict, best_trial, num_epochs, num_workers)

    except KeyboardInterrupt:
        print("\nInterrupted, saving the results collected so far.")
        header["updated"] = datetime.now().isoformat(timespec="seconds")
        save_study(study_file, header, trials)
        save_results_table(os.path.join(study_directory, "results.csv"), trials)

    # -------------------------------------------------------------------------------------------------------
    # Final report.
    # -------------------------------------------------------------------------------------------------------
    print_summary(trials, metric, mode)

    if best_trial is not None:
        print()
        print(f"Best {metric} = {best_value:.5f} (trial {best_trial['trial']})")
        print(f"Best hyperparameters found were: {json.dumps(best_trial['config'], indent=4, default=str)}")
        print(f"Full options file written to: {os.path.join(study_directory, 'best_options.json')}")

    return {"trials": trials, "best": best_trial}


def _save_best(
    study_directory: str,
    base_options_dict: Dict[str, Any],
    best_trial: Dict[str, Any],
    num_epochs: int,
    num_workers: int
) -> None:
    with open(os.path.join(study_directory, "best_config.json"), 'w') as json_file:
        json.dump(best_trial["config"], json_file, indent=4, default=str)

    best_options = build_trial_options(base_options_dict, best_trial["config"], num_epochs, num_workers)
    # The best options file is meant to be used for a full training run, so restore
    # the epoch count and the verbosity from the base options file.
    best_options.epochs = base_options_dict.get("epochs", best_options.epochs)
    best_options.verbose_output = base_options_dict.get("verbose_output", True)
    best_options.num_dataloader_workers = base_options_dict.get("num_dataloader_workers", num_workers)
    best_options.save(os.path.join(study_directory, "best_options.json"))


def _seed_everything(random_seed: Optional[int], verbose: bool = True) -> None:
    seed = random_seed
    if seed is None:
        environment_seed = int(os.environ.get("SEED", -1))
        seed = environment_seed if environment_seed >= 0 else None

    if seed is None:
        return

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if verbose:
        print(f"Using random seed {seed}.")


if __name__ == '__main__':
    parser = ArgumentParser(description="Hyperparameter optimization for SPANet without Ray Tune.")

    parser.add_argument(
        "base_options_file", type=str,
        help="Base options file to load and adjust with the sampled hyperparameters."
    )

    parser.add_argument(
        "-sf", "--search_space_file", type=str, default=None,
        help="JSON file defining the search space. Defaults to the built-in search space."
    )

    parser.add_argument(
        "-a", "--algorithm", type=str, default="random", choices=["random", "grid", "optuna"],
        help="Search algorithm. 'optuna' (bayesian TPE) requires: pip install optuna."
    )

    parser.add_argument(
        "-P", "--pruner", type=str, default="median", choices=["median", "hyperband", "none"],
        help="Early stopping rule for unpromising trials. 'hyperband' is only available with --algorithm optuna."
    )

    parser.add_argument(
        "-mt", "--metric", type=str, default=None,
        help="Metric to optimize. Defaults to the 'metric' entry of the base options file."
    )

    parser.add_argument(
        "-md", "--mode", type=str, default=None, choices=["min", "max"],
        help="Whether the metric should be minimized or maximized. Inferred from the metric name by default."
    )

    parser.add_argument(
        "-t", "--num_trials", type=int, default=10,
        help="Number of trials to run."
    )

    parser.add_argument(
        "-e", "--num_epochs", type=int, default=10,
        help="Number of training epochs per trial."
    )

    parser.add_argument(
        "-w", "--num_workers", type=int, default=4,
        help="Number of dataloader workers to use for each trial."
    )

    parser.add_argument(
        "-g", "--gpus", type=int, default=0,
        help="Set to 1 to force training on a GPU. 0 selects the accelerator automatically. "
             "Only one device per trial is supported."
    )

    parser.add_argument(
        "-fp16", "--fp16", action="store_true",
        help="Use Torch AMP for the trials."
    )

    parser.add_argument(
        "--save_checkpoints", action="store_true",
        help="Save the best checkpoint of every trial. Disabled by default to save disk space."
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=None,
        help="Override the batch size of the base options file."
    )

    parser.add_argument(
        "-p", "--limit_dataset", type=float, default=None,
        help="Limit the dataset to the first L percent of the data (0 - 100). Speeds up the search."
    )

    parser.add_argument(
        "--time_limit", type=str, default=None,
        help="Time limit for a single trial, in the format DD:HH:MM:SS."
    )

    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Total time budget for the whole study, in seconds. No new trial is started past it."
    )

    parser.add_argument(
        "--min_trials", type=int, default=3,
        help="Number of completed trials required before the pruner starts stopping trials."
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=1,
        help="Number of epochs a trial is allowed to run before it may be pruned."
    )

    parser.add_argument(
        "-r", "--random_seed", type=int, default=None,
        help="Random seed for the sampler and for training."
    )

    parser.add_argument("-ef", "--event_file", type=str, default=None,
                        help="Override the event file of the base options file.")

    parser.add_argument("-tf", "--training_file", type=str, default=None,
                        help="Override the training file of the base options file.")

    parser.add_argument("-vf", "--validation_file", type=str, default=None,
                        help="Override the validation file of the base options file.")

    parser.add_argument(
        "-l", "--log_dir", type=str, default="spanet_output",
        help="Output directory for all of the trials."
    )

    parser.add_argument(
        "-n", "--name", type=str, default="spanet_optimize",
        help="The sub-directory to create for this study."
    )

    parser.add_argument(
        "--resume", action="store_true",
        help="Continue a previous study stored in {log_dir}/{name} instead of overwriting it."
    )

    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only print the configurations that would be trained, without training anything."
    )

    optimize_spanet(**parser.parse_args().__dict__)
