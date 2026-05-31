"""
MLOps Pipeline - Training NLU Intent Classifier.

Pipeline pelatihan model klasifikasi intent untuk router query pengguna.
Menggunakan TF-IDF + LinearSVC dengan hyperparameter tuning via GridSearchCV.

Alur:
1. Muat dataset dari file JSON (Data Decoupling).
2. Train/Test Split SEBELUM augmentasi (cegah Data Leakage).
3. Hyperparameter Tuning dengan GridSearchCV (3-fold CV).
4. Evaluasi pada Test Set (Precision, Recall, F1-Score).
5. Ekspor model .pkl dan metrik .json untuk CI/CD tracking.

Usage:
    python train_intent_model.py
"""

import json
import os
import logging
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ═══════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s",
    handlers=[
        logging.FileHandler("training_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset intent dari file JSON.

    File JSON harus berformat array of objects:
    [{"text": "...", "intent": "..."}, ...]

    Args:
        filepath: Path absolut atau relatif ke file JSON dataset.

    Returns:
        DataFrame dengan kolom 'text' dan 'intent'.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        ValueError: Jika format data tidak valid.
    """
    logger.info(f"Mencoba memuat dataset dari {filepath}...")

    if not os.path.exists(filepath):
        logger.error(f"File dataset tidak ditemukan di {filepath}")
        raise FileNotFoundError(f"File {filepath} tidak ditemukan.")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Validasi kolom yang dibutuhkan
    required_columns = {"text", "intent"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset tidak memiliki kolom yang diperlukan: {missing}. "
            f"Kolom yang ada: {list(df.columns)}"
        )

    logger.info(f"Berhasil memuat {len(df)} baris data.")
    logger.info(f"Distribusi intent:\n{df['intent'].value_counts().to_string()}")

    return df


def train_and_evaluate() -> None:
    """
    Pipeline utama: training, evaluasi, dan ekspor model + metrik.

    Steps:
    1. Load dataset dari file JSON.
    2. Train/Test split (80/20, stratified).
    3. GridSearchCV untuk hyperparameter tuning.
    4. Evaluasi pada test set.
    5. Ekspor model ke .pkl dan metrik ke .json.
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PELATIHAN MODEL INTENT")
    logger.info("=" * 60)

    # ── STEP 1: Load Dataset (Data Decoupling) ──
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "intent_dataset.json")
    df = load_dataset(dataset_path)

    X = df["text"]
    y = df["intent"]

    # ── STEP 2: Train/Test Split SEBELUM Augmentasi ──
    # PENTING: Split dilakukan sebelum augmentasi data apapun
    # untuk mencegah data leakage (brief requirement).
    logger.info("Melakukan pemisahan Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    logger.info(
        f"Data Training: {len(X_train)} sampel | "
        f"Data Testing: {len(X_test)} sampel"
    )

    # ── STEP 3: Pipeline + GridSearchCV ──
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("clf", LinearSVC(random_state=42, dual=False)),
    ])

    param_grid: dict[str, list[Any]] = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    logger.info("Memulai Hyperparameter Tuning dengan GridSearchCV (3-fold CV)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    logger.info(f"Pelatihan Selesai. Parameter optimal: {grid_search.best_params_}")
    logger.info(f"Best CV F1-Score (macro): {grid_search.best_score_:.4f}")

    # ── STEP 4: Evaluasi pada Test Set ──
    logger.info("Mengevaluasi model pada Test Set (unseen data)...")
    y_pred = best_model.predict(X_test)

    # Classification report sebagai string (untuk logging)
    report_str = classification_report(y_test, y_pred)
    logger.info(f"\nLaporan Evaluasi Metrik:\n{report_str}")

    # Classification report sebagai dict (untuk JSON export)
    report_dict: dict[str, Any] = classification_report(
        y_test, y_pred, output_dict=True
    )

    # ── STEP 5: Export Model (.pkl) ──
    output_dir = os.path.join(os.path.dirname(__file__), "app", "agent")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "intent_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Model disimpan di: {model_path}")

    # ── STEP 6: Export Metrics (.json) untuk CI/CD Tracking ──
    metrics_path = os.path.join(output_dir, "intent_metrics.json")
    metrics_payload: dict[str, Any] = {
        "model_type": "TfidfVectorizer + LinearSVC",
        "best_params": grid_search.best_params_,
        "best_cv_score_f1_macro": round(grid_search.best_score_, 4),
        "test_set_size": len(X_test),
        "train_set_size": len(X_train),
        "classification_report": report_dict,
        "intent_labels": sorted(y.unique().tolist()),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Metrik evaluasi diekspor ke: {metrics_path}")
    logger.info("=" * 60)
    logger.info("PIPELINE SELESAI")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        logger.error(f"Terjadi kesalahan fatal pada pipeline: {e}", exc_info=True)
        raise
