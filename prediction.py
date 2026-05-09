"""
prediction.py — Universal async prediction engine
Teen Mental Health Depression Detection

Replică logica MATLAB:
  cvpartition(Y, 'HoldOut', 0.3)          →  train_test_split(test_size=0.3, stratify=y)
  fitcknn(XTrain, YTrain, 'NumNeighbors', 5) →  KNeighborsClassifier(n_neighbors=5)

Extinde cu:
  - Benchmark multi-algoritm (7 clasificatori)
  - Selecție automată a celui mai bun model prin CV stratificat
  - Interfață async non-blocantă (asyncio + ThreadPoolExecutor)
  - Metrici complete: accuracy, precision, recall, F1, ROC-AUC
"""

from __future__ import annotations

import asyncio
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── Constante (identice cu MATLAB) ────────────────────────────────────────────

FEATURES: list[str] = [
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level",
]

TARGET: str = "depression_label"
TEST_SIZE: float = 0.30     # HoldOut 0.3 — identic MATLAB
RANDOM_STATE: int = 42
CV_FOLDS: int = 5           # StratifiedKFold → echivalent cvpartition('KFold', 5)

# ThreadPoolExecutor global — reutilizat de toate predicțiile async
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pred_worker")


# ── Clasificatori candidați ───────────────────────────────────────────────────

def _build_candidates() -> dict[str, Any]:
    """
    Returnează dicționar {nume: estimator}.
    KNN-5 este baseline-ul MATLAB; celelalte sunt extinse pentru comparare.
    """
    return {
        "KNN-5 (MATLAB baseline)": KNeighborsClassifier(
            n_neighbors=5, metric="minkowski", weights="uniform"
        ),
        "KNN-3": KNeighborsClassifier(
            n_neighbors=3, metric="minkowski", weights="distance"
        ),
        "KNN-7": KNeighborsClassifier(
            n_neighbors=7, metric="minkowski", weights="distance"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
        ),
        "SVC-RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
        ),
    }


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    """Rezultatele complete ale unui clasificator."""
    name: str
    accuracy_cv: float          # media CV pe train
    accuracy_cv_std: float      # deviație standard CV
    accuracy_holdout: float     # acuratețe pe set de test (30%)
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fit_time_s: float
    predict_time_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name:<30} "
            f"CV={self.accuracy_cv:.4f}±{self.accuracy_cv_std:.4f}  "
            f"HO={self.accuracy_holdout:.4f}  "
            f"F1={self.f1:.4f}  "
            f"AUC={self.roc_auc:.4f}  "
            f"fit={self.fit_time_s:.3f}s  "
            f"pred={self.predict_time_ms:.3f}ms"
        )


@dataclass
class BenchmarkReport:
    """Raport complet al benchmark-ului multi-model."""
    results: list[ModelMetrics] = field(default_factory=list)
    best: ModelMetrics | None = None

    @property
    def ranked(self) -> list[ModelMetrics]:
        """Rezultate sortate descendent după CV accuracy, apoi F1."""
        return sorted(
            self.results, key=lambda m: (m.accuracy_cv, m.f1), reverse=True
        )

    def summary(self) -> str:
        header = (
            f"\n{'─'*96}\n"
            f"{'Model':<30} {'CV-Acc':>12} {'HO-Acc':>8} "
            f"{'F1':>8} {'AUC':>8} {'Fit(s)':>8} {'Pred(ms)':>10}\n"
            f"{'─'*96}"
        )
        rows = "\n".join(
            f"{'★ ' if m.name == self.best.name else '  '}"
            f"{m.name:<28} "
            f"{m.accuracy_cv:.4f}±{m.accuracy_cv_std:.4f}  "
            f"{m.accuracy_holdout:>8.4f}  "
            f"{m.f1:>8.4f}  "
            f"{m.roc_auc:>8.4f}  "
            f"{m.fit_time_s:>8.3f}  "
            f"{m.predict_time_ms:>10.3f}"
            for m in self.ranked
        )
        best_line = (
            f"\n{'─'*96}\n"
            f"★  Cel mai bun model: {self.best.name}\n"
            f"   CV Accuracy : {self.best.accuracy_cv:.4f} ± {self.best.accuracy_cv_std:.4f}\n"
            f"   HO Accuracy : {self.best.accuracy_holdout:.4f}  "
            f"({self.best.accuracy_holdout*100:.2f}%)  "
            f"← echivalent MATLAB {self.best.accuracy_holdout*100:.2f}%\n"
            f"   F1 Score    : {self.best.f1:.4f}\n"
            f"   ROC-AUC     : {self.best.roc_auc:.4f}\n"
            f"{'─'*96}"
        )
        return header + "\n" + rows + best_line


# ── Motor principal ───────────────────────────────────────────────────────────

class PredictionEngine:
    """
    Motor de predicție thread-safe cu selecție automată a celui mai bun model.

    Flux:
        engine = PredictionEngine("Teen_Mental_Health_Dataset.csv")
        report = engine.initialize()          # sync
        # sau
        report = await engine.initialize_async()  # async

        result = engine.predict([7, 5, 2.5, 2.3, 1, 8, 9, 8])
        # sau
        result = await engine.predict_async([7, 5, 2.5, 2.3, 1, 8, 9, 8])
    """

    def __init__(self, data_path: str | Path) -> None:
        self._data_path = Path(data_path)
        self._scaler: StandardScaler | None = None
        self._model: Any | None = None
        self._report: BenchmarkReport | None = None
        self._ready: bool = False

    # ── API public ────────────────────────────────────────────────────────────

    def initialize(self) -> BenchmarkReport:
        """
        Inițializare sincronă:
          1. Încarcă și pregătește datele (mirrors MATLAB HoldOut 0.3)
          2. Rulează benchmark pe 8 clasificatori cu 5-fold StratifiedKFold
          3. Alege cel mai bun model și îl antrenează pe întregul set de train
        """
        print("\n[PredictionEngine] Pregătire date...")
        X_train, X_test, y_train, y_test, scaler = self._prepare_data()

        print("[PredictionEngine] Rulare benchmark...\n")
        report = self._run_benchmark(X_train, X_test, y_train, y_test)

        print(report.summary())

        print(f"\n[PredictionEngine] Reantrenare model final: {report.best.name}")
        self._scaler = scaler
        self._model = self._refit_winner(report.best.name, X_train, y_train)
        self._report = report
        self._ready = True

        print("[PredictionEngine] Gata.\n")
        return report

    async def initialize_async(self) -> BenchmarkReport:
        """
        Inițializare asincronă — rulează benchmark-ul CPU-bound
        într-un thread executor fără a bloca event loop-ul.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_EXECUTOR, self.initialize)

    def predict(self, input_values: list[float] | np.ndarray) -> int:
        """
        Predicție sincronă pentru un singur caz.

        Parametri (ordinea identică cu MATLAB nouCopil):
            [daily_social_media_hours, sleep_hours, screen_time_before_sleep,
             academic_performance, physical_activity, stress_level,
             anxiety_level, addiction_level]

        Returnează:
            1 — depresie prezentă
            0 — fără depresie
        """
        self._assert_ready()
        arr = np.array(input_values, dtype=float).reshape(1, -1)
        scaled = self._scaler.transform(arr)
        return int(self._model.predict(scaled)[0])

    async def predict_async(self, input_values: list[float] | np.ndarray) -> int:
        """
        Predicție asincronă — non-blocantă.
        Rulează predict() în thread pool pentru a nu bloca event loop-ul.

        Exemplu:
            result = await engine.predict_async([7, 5, 2.5, 2.3, 1, 8, 9, 8])
        """
        self._assert_ready()
        loop = asyncio.get_event_loop()
        values = list(input_values) if not isinstance(input_values, list) else input_values
        return await loop.run_in_executor(_EXECUTOR, self.predict, values)

    def predict_proba(self, input_values: list[float] | np.ndarray) -> float:
        """
        Returnează probabilitatea de depresie (0.0 – 1.0).
        Disponibil doar dacă modelul suportă predict_proba.
        """
        self._assert_ready()
        arr = np.array(input_values, dtype=float).reshape(1, -1)
        scaled = self._scaler.transform(arr)
        if hasattr(self._model, "predict_proba"):
            return float(self._model.predict_proba(scaled)[0][1])
        raise NotImplementedError(
            f"{self.best_model_name} nu suportă predict_proba."
        )

    @property
    def report(self) -> BenchmarkReport | None:
        return self._report

    @property
    def best_model_name(self) -> str | None:
        return self._report.best.name if self._report else None

    @property
    def holdout_accuracy(self) -> float | None:
        return self._report.best.accuracy_holdout if self._report else None

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Internals ─────────────────────────────────────────────────────────────

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "PredictionEngine nu este inițializat. "
                "Apelați .initialize() sau await .initialize_async() mai întâi."
            )

    def _prepare_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        Citire CSV + split 70/30 (identic MATLAB cvpartition HoldOut 0.3).
        Normalizare StandardScaler (scaler se fit-uiește DOAR pe train).
        """
        df = pd.read_csv(self._data_path)
        X = df[FEATURES].values.astype(float)
        y = df[TARGET].values.astype(int)

        # stratify=y → echivalent cu cvpartition(Y, 'HoldOut', 0.3) stratificat
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        n_total = len(y)
        n_train = len(y_train)
        n_test = len(y_test)
        dep_rate = y.mean() * 100
        print(
            f"  Date: {n_total} rânduri  |  "
            f"Train: {n_train} ({(1-TEST_SIZE)*100:.0f}%)  |  "
            f"Test: {n_test} ({TEST_SIZE*100:.0f}%)  |  "
            f"Rată depresie: {dep_rate:.1f}%"
        )
        return X_train, X_test, y_train, y_test, scaler

    def _run_benchmark(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> BenchmarkReport:
        """
        Evaluează fiecare clasificator cu:
          - 5-fold StratifiedKFold CV pe setul de train
          - Holdout test pe setul de test 30%
        Selectează câștigătorul după (CV accuracy ↓, F1 ↓).
        """
        cv_strategy = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        candidates = _build_candidates()
        results: list[ModelMetrics] = []

        for name, clf in candidates.items():
            # ── 5-fold CV pe setul de antrenament ──
            t0 = time.perf_counter()
            cv_result = cross_validate(
                clf, X_train, y_train,
                cv=cv_strategy,
                scoring="accuracy",
                return_train_score=False,
                n_jobs=1,           # n_jobs=-1 e deja pe estimator unde e util
            )
            cv_time = time.perf_counter() - t0

            # ── Antrenare finală + evaluare holdout ──
            clf.fit(X_train, y_train)

            t_pred = time.perf_counter()
            y_pred = clf.predict(X_test)
            pred_ms = (time.perf_counter() - t_pred) * 1000

            # ROC-AUC (necesită predict_proba)
            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = float("nan")

            m = ModelMetrics(
                name=name,
                accuracy_cv=float(cv_result["test_score"].mean()),
                accuracy_cv_std=float(cv_result["test_score"].std()),
                accuracy_holdout=float(accuracy_score(y_test, y_pred)),
                precision=float(precision_score(y_test, y_pred, zero_division=0)),
                recall=float(recall_score(y_test, y_pred, zero_division=0)),
                f1=float(f1_score(y_test, y_pred, zero_division=0)),
                roc_auc=auc,
                fit_time_s=cv_time,
                predict_time_ms=pred_ms,
            )
            results.append(m)
            print(f"  {m}")

        best = max(results, key=lambda m: (m.accuracy_cv, m.f1))
        return BenchmarkReport(results=results, best=best)

    def _refit_winner(
        self, best_name: str, X_train: np.ndarray, y_train: np.ndarray
    ) -> Any:
        """Antrenează modelul câștigător pe întregul set de train."""
        model = _build_candidates()[best_name]
        model.fit(X_train, y_train)
        return model


# ── Singleton module-level ────────────────────────────────────────────────────

_engine: PredictionEngine | None = None


def get_engine(data_path: str | Path | None = None) -> PredictionEngine:
    """
    Returnează singleton-ul PredictionEngine.
    La primul apel, data_path este obligatoriu.
    Apelurile ulterioare returnează instanța deja inițializată.
    """
    global _engine
    if _engine is None:
        if data_path is None:
            raise ValueError("data_path este obligatoriu la primul apel get_engine().")
        _engine = PredictionEngine(data_path)
        _engine.initialize()
    return _engine


def predict_sync(
    input_values: list[float],
    data_path: str | Path | None = None,
) -> int:
    """
    Shortcut sincron — pentru Streamlit și alte contexte non-async.

    Exemplu:
        result = predict_sync([7, 5, 2.5, 2.3, 1, 8, 9, 8], "Teen_Mental_Health_Dataset.csv")
    """
    return get_engine(data_path).predict(input_values)


async def predict_async(
    input_values: list[float],
    data_path: str | Path | None = None,
) -> int:
    """
    Shortcut async — pentru FastAPI, asyncio scripts, etc.

    Exemplu:
        result = await predict_async([7, 5, 2.5, 2.3, 1, 8, 9, 8])
    """
    global _engine
    if _engine is None:
        if data_path is None:
            raise ValueError("data_path este obligatoriu la primul apel predict_async().")
        _engine = PredictionEngine(data_path)
        await _engine.initialize_async()
    return await _engine.predict_async(input_values)


# ── Entry point standalone ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    csv_path = Path(__file__).parent / "Teen_Mental_Health_Dataset.csv"

    print("=" * 96)
    print(" BENCHMARK PERFORMANȚĂ — Teen Mental Health Depression Predictor")
    print("=" * 96)

    engine = PredictionEngine(csv_path)
    report = engine.initialize()

    # Test pe cazul din MATLAB: nouCopil = [7, 5, 2.5, 2.3, 1, 8, 9, 8]
    matlab_case = [7, 5, 2.5, 2.3, 1, 8, 9, 8]
    result = engine.predict(matlab_case)
    prob = engine.predict_proba(matlab_case) if hasattr(engine._model, "predict_proba") else None

    print("\n── Test caz MATLAB (nouCopil) ─────────────────────────────────────────────")
    print(f"  Input : {dict(zip(FEATURES, matlab_case))}")
    print(f"  Output: {'Depresie prezentă (1)' if result == 1 else 'Fără depresie (0)'}")
    if prob is not None:
        print(f"  Probabilitate depresie: {prob*100:.1f}%")
    print(f"\n  Model utilizat: {engine.best_model_name}")
    print(f"  Acuratețe HoldOut: {engine.holdout_accuracy*100:.2f}%")
    print("=" * 96)

    # Test async
    async def _async_test() -> None:
        print("\n── Test predicție async ────────────────────────────────────────────────────")
        cases = [
            ([7, 5, 2.5, 2.3, 1, 8, 9, 8], "Caz MATLAB (risc înalt)"),
            ([2, 8, 0.5, 3.8, 1.8, 2, 1, 1], "Profil sănătos"),
            ([6, 5, 2.8, 2.1, 0.5, 9, 9, 9], "Risc maxim"),
        ]
        tasks = [engine.predict_async(vals) for vals, _ in cases]
        results = await asyncio.gather(*tasks)
        for (vals, label), pred in zip(cases, results):
            status = "DEPRESIE (1)" if pred == 1 else "SĂNĂTOS (0)"
            print(f"  {label:<30} → {status}")

    asyncio.run(_async_test())
    print("=" * 96)
