"""Sampling (muestreo) strategies for the e-commerce text-mining pipeline (Bitácora 04).

This module is the piece that is *new* to this bitácora. It mirrors the design
of ``feature_extractors.py``: each class-balancing technique required by the
activity is encapsulated behind the common ``SamplingStrategy`` interface
(*Strategy* pattern), and a *Template Method* (``run``) times ``fit_resample``
uniformly and records the resulting class distribution, so the notebook can
build its comparison tables and bar charts without duplicating measurement code.

The three techniques are:

- **Submuestreo aleatorio** (``RandomUnderSampler``): drops examples from the
  majority classes until every class matches the *minority* count. Cheap and
  fast, but discards potentially useful data.
- **Sobremuestreo con réplicas** (``RandomOverSampler``): duplicates examples of
  the minority classes until every class matches the *majority* count. Keeps all
  data but the exact replicas can encourage overfitting.
- **Sobremuestreo SMOTE** (``SMOTE``): synthesises new minority examples by
  interpolating between a sample and its k nearest neighbours, instead of
  copying. Produces a balanced set without literal duplicates.

Keeping the strategies in a standalone, importable module (instead of the
notebook) makes them *picklable*, which is what lets each technique run in its
own worker process with ``concurrent.futures.ProcessPoolExecutor`` -- the second
CPU-parallel stage the activity asks us to demonstrate ("tres técnicas
ejecutadas de manera paralela en CPU").

Design note. Only the **training** matrix is ever resampled; the test set is
left untouched so that every technique (and the no-sampling baseline) is
evaluated on the exact same, unbiased hold-out. The imbalanced-learn samplers
accept the sparse CSR matrix produced by ``TfidfVectorizer`` directly, so no
densification is needed at this stage.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingResult:
    """Outcome of applying a single sampling strategy to the training set."""

    name: str
    #: Resampled feature matrix (scipy sparse CSR or dense ndarray).
    X_resampled: object
    #: Resampled label vector (ndarray).
    y_resampled: object
    elapsed_seconds: float
    #: ``{label: count}`` after resampling (labels are the encoded ints).
    class_distribution: dict

    @property
    def n_samples(self) -> int:
        return int(self.y_resampled.shape[0])


class SamplingStrategy(ABC):
    """Common interface for every sampling strategy."""

    #: Human-readable name of the technique (used in tables, filenames, charts).
    name: str

    @abstractmethod
    def build(self):
        """Return a fresh, unfitted imbalanced-learn sampler."""

    def run(self, X_train, y_train) -> SamplingResult:
        """Resample the training set, time ``fit_resample`` and record the
        resulting per-class counts."""
        sampler = self.build()

        start = time.perf_counter()
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        elapsed = time.perf_counter() - start

        distribution = {int(label): int(count) for label, count in sorted(Counter(y_res).items())}
        return SamplingResult(
            name=self.name,
            X_resampled=X_res,
            y_resampled=y_res,
            elapsed_seconds=elapsed,
            class_distribution=distribution,
        )


class RandomUnderSamplingStrategy(SamplingStrategy):
    """Submuestreo aleatorio: descarta ejemplos de las clases mayoritarias hasta
    igualar el conteo de la clase minoritaria."""

    name = "Submuestreo aleatorio"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def build(self):
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(random_state=self.random_state)


class ReplicationOverSamplingStrategy(SamplingStrategy):
    """Sobremuestreo con réplicas: duplica ejemplos de las clases minoritarias
    hasta igualar el conteo de la clase mayoritaria."""

    name = "Sobremuestreo replicas"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def build(self):
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=self.random_state)


class SMOTEOverSamplingStrategy(SamplingStrategy):
    """Sobremuestreo SMOTE: genera ejemplos sintéticos de las clases minoritarias
    interpolando entre cada muestra y sus ``k_neighbors`` vecinos más cercanos."""

    name = "Sobremuestreo SMOTE"

    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def build(self):
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors)


def run_sampler(strategy: SamplingStrategy, X_train, y_train) -> SamplingResult:
    """Module-level dispatch target for ``ProcessPoolExecutor.submit``.

    Pins BLAS/OpenMP threads to 1 inside the worker so that each technique truly
    occupies a single core, then delegates to ``strategy.run``. Unlike the
    classification stage (where only scalar metrics cross the process boundary),
    here the resampled matrix itself is the payload, so the inter-process cost is
    not negligible -- a nuance worth reporting for the CPU-parallel muestreo.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    return strategy.run(X_train, y_train)


#: Canonical ordering of the three techniques required by the activity.
DEFAULT_SAMPLING_STRATEGIES = [
    RandomUnderSamplingStrategy(),
    ReplicationOverSamplingStrategy(),
    SMOTEOverSamplingStrategy(),
]
