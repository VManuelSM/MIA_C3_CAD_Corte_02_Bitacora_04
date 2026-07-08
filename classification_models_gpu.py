"""GPU classification strategies for the e-commerce text-mining pipeline (Bitácora 04).

Adapted from ``classification_models_gpu.py`` (Bitácora 03). Every algorithm
required by the activity (Random Forest, Logistic Regression, SVM) is
encapsulated behind a common ``GPUClassifierStrategy`` interface (*Strategy*
pattern), with a *Template Method* (``run``) that uniformly times the
``fit``/``predict`` cycle, computes the metrics and saves the confusion matrix.

Two changes with respect to Bitácora 03, both motivated by the *muestreo* focus
of this bitácora:

1. Beyond plain ``accuracy`` (which is misleading on imbalanced data), ``run``
   also reports **balanced accuracy** (mean per-class recall) and **macro-F1**,
   the metrics that actually reveal whether a sampling technique helped the
   minority classes.
2. ``run`` takes a ``tag`` (the name of the sampling technique that produced the
   training set) so that the confusion-matrix filenames of the 4 sampling
   variants x 3 classifiers do not collide.

As in Bitácora 03 there is **no** ``ProcessPoolExecutor``: a single GPU is one
device, so the three strategies are run **sequentially** by
``run_gpu_classifiers``. Estimators come from RAPIDS cuML; metrics are computed
with scikit-learn on the host after predictions are copied back from the GPU.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def _slug(text: str) -> str:
    """Filesystem/Obsidian-friendly slug (letters, digits and underscores)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


@dataclass(frozen=True)
class GPUClassificationResult:
    """Outcome of training and evaluating a single classifier on GPU."""

    name: str
    #: Name of the sampling technique that produced the training set.
    sampling: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    train_seconds: float
    predict_seconds: float
    report: str
    matrix_path: Path

    @property
    def total_seconds(self) -> float:
        return self.train_seconds + self.predict_seconds


class GPUClassifierStrategy(ABC):
    """Common interface for every GPU classification strategy."""

    #: Human-readable name of the algorithm (used in filenames and reports).
    name: str
    #: cuML's RandomForestClassifier does not accept sparse input (unlike
    #: LogisticRegression/SVC); subclasses that need it set this to True so
    #: ``run`` densifies the TF-IDF matrix just for that estimator.
    requires_dense: bool = False

    @abstractmethod
    def build(self):
        """Return a fresh, unfitted cuML estimator."""

    def run(self, X_train, y_train, X_test, y_test, target_names, output_dir: Path, tag: str = "") -> GPUClassificationResult:
        """Fit the estimator, predict on the test set, time both stages and
        persist the confusion matrix as a PNG under ``output_dir``."""
        estimator = self.build()

        # cuML wants float32 features and int32 labels. Densify only for the
        # estimators that cannot consume sparse input (Random Forest).
        y_train_i = np.asarray(y_train).astype(np.int32)
        y_test_i = np.asarray(y_test).astype(np.int32)
        if self.requires_dense:
            X_train = (X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)).astype(np.float32)
            X_test = (X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)).astype(np.float32)
        else:
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)

        start = time.perf_counter()
        estimator.fit(X_train, y_train_i)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = estimator.predict(X_test)
        predict_seconds = time.perf_counter() - start

        # cuML predictions may come back as cuDF/CuPy objects; bring them to
        # host (NumPy) so scikit-learn's metrics can consume them.
        y_pred_host = np.asarray(y_pred.get() if hasattr(y_pred, "get") else y_pred).astype(int)
        y_test_host = y_test_i.astype(int)

        accuracy = float((y_pred_host == y_test_host).mean())
        balanced = float(balanced_accuracy_score(y_test_host, y_pred_host))
        macro_f1 = float(f1_score(y_test_host, y_pred_host, average="macro"))
        report = classification_report(y_test_host, y_pred_host, target_names=target_names)

        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = output_dir / f"cad-actividad4-cm-{_slug(tag)}-{_slug(self.name)}.png"
        matrix = confusion_matrix(y_test_host, y_pred_host)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=target_names)
        fig, ax = plt.subplots(figsize=(6, 6))
        display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        ax.set_title(f"{self.name} GPU — {tag}")
        fig.tight_layout()
        fig.savefig(matrix_path, dpi=150)
        plt.close(fig)

        return GPUClassificationResult(
            name=self.name,
            sampling=tag,
            accuracy=accuracy,
            balanced_accuracy=balanced,
            macro_f1=macro_f1,
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            report=report,
            matrix_path=matrix_path,
        )


class RandomForestGPUStrategy(GPUClassifierStrategy):
    """Random Forest classifier (cuML), GPU-accelerated bagging of decision trees."""

    name = "Random Forest"
    requires_dense = True

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def build(self):
        from cuml.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)


class LogisticRegressionGPUStrategy(GPUClassifierStrategy):
    """Multinomial logistic regression (cuML), GPU-accelerated linear classifier."""

    name = "Regresion Logistica"

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter

    def build(self):
        from cuml.linear_model import LogisticRegression

        return LogisticRegression(max_iter=self.max_iter)


class SVMGPUStrategy(GPUClassifierStrategy):
    """Support Vector Machine with RBF kernel (cuML), GPU-accelerated."""

    name = "SVM"

    def __init__(self, kernel: str = "rbf"):
        self.kernel = kernel

    def build(self):
        from cuml.svm import SVC

        return SVC(kernel=self.kernel)


def run_gpu_classifiers(strategies, X_train, y_train, X_test, y_test, target_names, output_dir: Path, tag: str = ""):
    """Run every strategy **sequentially** (one GPU, no process pool) over a
    single training set and return the list of :class:`GPUClassificationResult`,
    printing a professor-style report block for each algorithm as it finishes."""
    results = []
    for strategy in strategies:
        print("*" * 40)
        print(f"{strategy.name} GPU  [{tag}]")
        print("*" * 40)

        result = strategy.run(X_train, y_train, X_test, y_test, target_names, output_dir, tag=tag)
        results.append(result)

        print(f"Accuracy: {result.accuracy:.4f} | Balanced acc: {result.balanced_accuracy:.4f} | Macro-F1: {result.macro_f1:.4f}")
        print(result.report)
        print(f"Matriz guardada: {result.matrix_path}")
        print(f"Tiempo: {result.total_seconds:.2f}s\n")

    return results


#: Canonical ordering of the three strategies, matching the activity diagram.
DEFAULT_GPU_STRATEGIES = [
    RandomForestGPUStrategy(),
    LogisticRegressionGPUStrategy(),
    SVMGPUStrategy(),
]
