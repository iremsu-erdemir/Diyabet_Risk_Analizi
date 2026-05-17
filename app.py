"""
Şeker Hastalığı Tahmin Uygulaması
Streamlit tabanlı interaktif web arayüzü
"""

import re

import streamlit as st
import pandas as pd
import joblib
import json

from src.paths import get_paths, snap_display_scalar

# ======================== SAYFA KONFIGÜRASYONU ========================
st.set_page_config(
    page_title="Diyabet Risk Analizi",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.25rem; padding-bottom: 2rem; max-width: 1100px; }
        .hero-title {
            text-align: center;
            margin: 0.75rem 0 1.35rem 0;
            padding-top: 0.35rem;
            line-height: 1.15;
            user-select: none;
        }
        .hero-title .hero-letter {
            display: inline-block;
            font-size: clamp(2rem, 6vw, 3.1rem);
            font-weight: 800;
            letter-spacing: 0.02em;
            color: #FF4B4B;
            text-shadow: 2px 3px 0 rgba(255, 75, 75, 0.18);
            animation: hero-pop 0.55s ease backwards;
        }
        .hero-title .hero-space { display: inline-block; width: 0.35em; }
        @keyframes hero-pop {
            from { opacity: 0; transform: translateY(12px) scale(0.85); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .info-card, .result-card, .deploy-card, .recall-note {
            padding: 1rem 1.1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            margin-bottom: 1rem;
        }
        .deploy-card {
            background: #eef4fb;
            border-left: 4px solid #7ba3d4;
        }
        .deploy-card p { margin: 0.2rem 0; font-size: 0.9rem; line-height: 1.5; color: #334155; }
        .recall-note {
            background: #f5f0fa;
            border-left: 4px solid #b8a4d4;
            font-size: 0.9rem;
            color: #475569;
            line-height: 1.55;
        }
        .result-card--low {
            border-left: 4px solid #8fc9a8;
            background: #f0f9f4;
        }
        .result-card--mid {
            border-left: 4px solid #e8c878;
            background: #fdf8ee;
        }
        .result-card--high {
            border-left: 4px solid #e0a0a0;
            background: #fdf3f3;
        }
        .result-card h3 { margin: 0 0 0.35rem 0; font-size: 1.05rem; font-weight: 600; color: #1e293b; }
        .result-card p { margin: 0.2rem 0; font-size: 0.9rem; color: #475569; }
        .result-card .card-tag {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .metrics-table-wrap {
            overflow-x: auto;
            margin: 0.5rem 0 0.85rem 0;
            -webkit-overflow-scrolling: touch;
        }
        table.metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }
        table.metrics-table th {
            text-align: left;
            padding: 0.55rem 0.7rem;
            font-weight: 600;
            background: #eef2f7;
            border-bottom: 2px solid #dde4ec;
            white-space: nowrap;
            color: #334155;
        }
        table.metrics-table td {
            padding: 0.5rem 0.7rem;
            border-bottom: 1px solid #e8edf2;
            color: #475569;
        }
        table.metrics-table tr.best-recall-row td {
            background: #e8f0fa;
            font-weight: 600;
            color: #1e293b;
        }
        table.metrics-table tr.best-recall-row td:first-child::after {
            content: " · en iyi recall";
            font-size: 0.72rem;
            font-weight: 500;
            color: #5b7fa6;
        }
        .prod-metrics-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem 1rem;
            margin: 0.75rem 0 0.35rem 0;
        }
        .prod-metric-item { min-width: 0; }
        .prod-metric-label {
            font-size: 0.82rem;
            color: #64748b;
            margin-bottom: 0.2rem;
        }
        .prod-metric-value {
            font-size: 1.45rem;
            font-weight: 600;
            color: #1e293b;
            line-height: 1.25;
        }
        .prod-metric-value.prod-metric-model {
            font-size: 1.05rem;
            font-weight: 700;
            word-break: break-word;
        }
        .prod-metric-caption {
            font-size: 0.8rem;
            color: #64748b;
            margin: 0.15rem 0 0 0;
        }
        .combo-info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem 1rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        .combo-info-item { min-width: 0; }
        .combo-info-label {
            font-size: 0.82rem;
            color: #64748b;
            margin-bottom: 0.2rem;
        }
        .combo-info-value {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1e293b;
            line-height: 1.35;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .of-section { margin-bottom: 1rem; }
        .of-section h4 {
            font-size: 0.95rem;
            font-weight: 600;
            color: #334155;
            margin: 0 0 0.5rem 0;
        }
        .of-approach-card {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 1px solid #dbeafe;
            background: #eff6ff;
            font-size: 0.88rem;
            color: #334155;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            font-weight: 600;
        }
        @media (max-width: 900px) {
            .prod-metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            .hero-title .hero-letter { font-size: 1.65rem; }
            .prod-metrics-grid { grid-template-columns: 1fr; }
            .combo-info-grid { grid-template-columns: 1fr; }
            table.metrics-table { font-size: 0.78rem; }
            table.metrics-table th, table.metrics-table td { padding: 0.4rem 0.45rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================== SABIT DEĞERLERLER ========================
PATHS = get_paths()
MODEL_PATH = PATHS.model_path
METRICS_PATH = PATHS.metrics_dir / "metrics_summary.json"
EDA_SUMMARY_PATH = PATHS.metrics_dir / "eda_summary.json"
FIGURES_DIR = PATHS.figures_dir
ALL_MODELS_DIR = PATHS.project_root / "models" / "all_models"

DEPLOYMENT_DISPLAY_NAME = "HistGradientBoosting + SMOTE + threshold"
DEPLOYMENT_TABLE_NAME = "HistGradientBoosting"
DEFAULT_TEST_RATIO = "0.20"
DEFAULT_PREPROCESS_MODE = "smote"

DEPLOY_METRIC_SOURCE_LABELS = {
    "test": "Test hold-out",
    "validation": "Doğrulama kümesi",
}

FEATURE_INFO = {
    "Pregnancies": {
        "label": "Hamilelik",
        "description": "Toplam hamilelik sayısı",
        "min": 0,
        "max": 17,
        "unit": "",
    },
    "Glucose": {
        "label": "Glikoz",
        "description": "Açlık / OGTT glikozu (mg/dL)",
        "min": 0,
        "max": None,
        "unit": "mg/dL",
    },
    "BloodPressure": {
        "label": "Kan basıncı",
        "description": "Diyastolik kan basıncı (mmHg)",
        "min": 0,
        "max": 122,
        "unit": "mmHg",
    },
    "SkinThickness": {
        "label": "Cilt",
        "description": "Triceps cilt kalınlığı (mm)",
        "min": 0,
        "max": 99,
        "unit": "mm",
    },
    "Insulin": {
        "label": "İnsülin",
        "description": "2 saat sonrası serum insülini (mu U/ml)",
        "min": 0,
        "max": 846,
        "unit": "mu U/ml",
    },
    "BMI": {
        "label": "VKİ",
        "description": "Vücut kitle indeksi (kg/m²)",
        "min": 0,
        "max": 67,
        "unit": "kg/m²",
    },
    "DiabetesPedigreeFunction": {
        "label": "Pedigri",
        "description": "Ailede diyabet geçmişi skoru",
        "min": 0.08,
        "max": 2.42,
        "unit": "",
    },
    "Age": {
        "label": "Yaş",
        "description": "Kişinin yaşı",
        "min": 21,
        "max": 81,
        "unit": "yıl",
    },
}

MODEL_DISPLAY_NAMES = {
    "svm": "SVM",
    "logreg": "Logistic Regression",
    "knn": "K-NN",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient Boosting",
}

MODEL_ORDER = [
    "svm",
    "logreg",
    "knn",
    "random_forest",
    "adaboost",
    "gradient_boosting",
]

COMPARISON_MODEL_KEYS = list(MODEL_ORDER)

ANALYSIS_MODEL_OPTIONS = {
    **{k: MODEL_DISPLAY_NAMES[k] for k in MODEL_ORDER},
    "deployment": f"{DEPLOYMENT_TABLE_NAME} (üretim)",
}
ANALYSIS_MODEL_SELECT_KEYS = COMPARISON_MODEL_KEYS + ["deployment"]

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")
METRIC_COLUMNS = ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")

_HYPERPARAM_LABELS = {
    "model__l2_regularization": "L2 düzenlileştirme",
    "model__learning_rate": "Öğrenme oranı",
    "model__max_depth": "Maksimum derinlik",
    "model__max_iter": "Maksimum iterasyon",
    "model__min_samples_leaf": "Yaprak minimum örnek",
    "model__n_iter_no_change": "Erken durdurma sabır",
}

MENU_OPTIONS = ("Tahmin", "Model Analizi", "Proje Hakkında")

FEATURES_LIST = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# ======================== YARDIMCI FONKSİYONLAR ========================


@st.cache_resource
def load_model():
    """Eğitilmiş modeli yükle"""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Model metriklerini yükle"""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_per_model_test_metrics():
    p = PATHS.metrics_dir / "per_model_test_metrics.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_eda_summary():
    """EDA özet bilgilerini yükle"""
    if not EDA_SUMMARY_PATH.exists():
        return None
    with open(EDA_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(x: float, decimals: int = 4) -> str:
    """Metrik gösteriminde sıfıra yakın artefaktları temizle."""
    return f"{snap_display_scalar(float(x)):.{decimals}f}"


def predict_diabetes(model, input_data):
    """
    Model ile tahmin yap

    Args:
        model: Eğitilmiş model
        input_data: Input dataframe

    Returns:
        prediction: 0 veya 1 (sağlıklı veya diyabet)
        probability: Diyabet olma olasılığı (0-1)
    """
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1]
    return int(prediction), float(probability)


def safe_predict(model, input_data: pd.DataFrame) -> tuple[int | None, float | None]:
    try:
        return predict_diabetes(model, input_data)
    except Exception:
        return None, None


def get_risk_level(probability: float) -> tuple[str, str]:
    """Olasılığa göre risk seviyesi (Düşük / Orta / Yüksek)."""
    if probability < 0.35:
        return "Düşük risk", "low"
    if probability < 0.65:
        return "Orta risk", "mid"
    return "Yüksek risk", "high"


def get_short_message(prediction: int, probability: float) -> str:
    """Kısa uyarı / bilgilendirme metni."""
    if prediction == 1:
        return (
            "Model, diyabet riski taşıdığınızı düşünüyor. "
            "Bu sonuç tanı değildir; bir sağlık kuruluşuna başvurmanız önerilir."
        )
    if probability >= 0.35:
        return (
            "Model şu an diyabet sinyali göstermiyor; ancak olasılık sınırda. "
            "Düzenli kontrolleri sürdürün."
        )
    return (
        "Model, mevcut girdilere göre düşük risk görüyor. "
        "Sağlıklı yaşam alışkanlıklarını sürdürün."
    )


def prediction_label(prediction: int | None) -> str:
    if prediction is None:
        return "—"
    return "Diyabet riski var" if prediction == 1 else "Diyabet riski düşük"


def resolve_metrics_slice(pm: dict | None) -> tuple[str, str]:
    if not pm:
        return DEFAULT_TEST_RATIO, DEFAULT_PREPROCESS_MODE
    ratio_options = sorted(pm.keys())
    ratio = DEFAULT_TEST_RATIO if DEFAULT_TEST_RATIO in ratio_options else ratio_options[-1]
    mode = DEFAULT_PREPROCESS_MODE if DEFAULT_PREPROCESS_MODE in pm.get(ratio, {}) else "plain"
    return ratio, mode


def get_recall_lookup(pm: dict | None, metrics: dict | None) -> dict[str, float]:
    ratio, mode = resolve_metrics_slice(pm)
    lookup: dict[str, float] = {}
    if pm and ratio in pm and mode in pm[ratio]:
        for key in COMPARISON_MODEL_KEYS:
            sc = pm[ratio][mode].get(key) or {}
            if sc.get("recall") is not None:
                lookup[key] = float(sc["recall"])
    deploy_scores = (metrics or {}).get("best_cv_scores") or {}
    if deploy_scores.get("recall") is not None:
        lookup["deployment"] = float(deploy_scores["recall"])
    return lookup


def _comparison_models_missing() -> list[str]:
    return [k for k in COMPARISON_MODEL_KEYS if not (ALL_MODELS_DIR / f"{k}.joblib").exists()]


@st.cache_resource(show_spinner="Karşılaştırma modelleri hazırlanıyor…")
def _train_and_save_comparison_models() -> bool:
    """Eksik karşılaştırma modellerini bir kez eğitip kaydeder."""
    try:
        from sklearn.base import clone
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        from src.pipeline import (
            SEED,
            _build_classifiers,
            _make_preprocessor,
            _tune_adaboost_n_estimators_val_early_stop,
            infer_target_column,
            load_data,
        )

        df = load_data()
        target_col = infer_target_column(df)
        drop_cols = [c for c in df.columns if c.lower() in {"id", "index", "record_id", "patient_id"}]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        classifiers = _build_classifiers()
        X_tv_tune, _, y_tv_tune, _ = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        classifiers, _ = _tune_adaboost_n_estimators_val_early_stop(
            classifiers, X_tv_tune, y_tv_tune, SEED
        )

        missing = _comparison_models_missing()
        for clf_name in missing:
            local_pre = _make_preprocessor(X)
            pipe = ImbPipeline(
                [
                    ("preprocessor", local_pre),
                    ("sampler", SMOTE(random_state=SEED, k_neighbors=3)),
                    ("model", clone(classifiers[clf_name])),
                ]
            )
            pipe.fit(X, y)
            joblib.dump(pipe, ALL_MODELS_DIR / f"{clf_name}.joblib")
        return True
    except Exception:
        return False


def ensure_comparison_models_saved() -> bool:
    ALL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not _comparison_models_missing():
        return True
    return _train_and_save_comparison_models()


def load_all_comparison_models() -> dict:
    models = {}
    for key in COMPARISON_MODEL_KEYS:
        path = ALL_MODELS_DIR / f"{key}.joblib"
        if not path.exists():
            continue
        try:
            models[key] = joblib.load(path)
        except Exception:
            continue
    return models


def build_all_model_predictions(
    input_df: pd.DataFrame,
    deploy_model,
    comparison_models: dict,
) -> list[dict]:
    rows = []
    pred, prob = safe_predict(deploy_model, input_df)
    rows.append(
        {
            "model_key": "deployment",
            "Model": DEPLOYMENT_DISPLAY_NAME,
            "Tahmin": prediction_label(pred),
            "Diyabet Olasılığı": prob,
            "Sağlıklı Olasılığı": (1.0 - prob) if prob is not None else None,
            "is_deploy": True,
        }
    )
    for key in COMPARISON_MODEL_KEYS:
        model = comparison_models.get(key)
        if model is None:
            rows.append(
                {
                    "model_key": key,
                    "Model": MODEL_DISPLAY_NAMES[key],
                    "Tahmin": "—",
                    "Diyabet Olasılığı": None,
                    "Sağlıklı Olasılığı": None,
                    "is_deploy": False,
                }
            )
            continue
        pred, prob = safe_predict(model, input_df)
        rows.append(
            {
                "model_key": key,
                "Model": MODEL_DISPLAY_NAMES[key],
                "Tahmin": prediction_label(pred),
                "Diyabet Olasılığı": prob,
                "Sağlıklı Olasılığı": (1.0 - prob) if prob is not None else None,
                "is_deploy": False,
            }
        )
    return rows


def enrich_prediction_rows(rows: list[dict], recall_lookup: dict[str, float]) -> pd.DataFrame:
    for row in rows:
        key = row["model_key"]
        recall_val = recall_lookup.get(key)
        row["Recall"] = recall_val
        row["Recall_sort"] = recall_val if recall_val is not None else -1.0

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    best_recall = df["Recall_sort"].max()
    best_name = None
    if best_recall >= 0:
        best_rows = df[df["Recall_sort"] == best_recall]
        best_name = str(best_rows.iloc[0]["Model"])

    explanations = []
    for _, row in df.iterrows():
        if row.get("is_deploy"):
            explanations.append("Tahminlerde kullanılan ana model")
        elif best_name and row["Model"] == best_name and row.get("Recall") is not None:
            explanations.append("Test kümesinde en yüksek recall")
        else:
            explanations.append("Karşılaştırma amaçlı")
    df["Açıklama"] = explanations
    df["is_best_recall"] = df["Model"] == best_name if best_name else False
    return df.sort_values("Recall_sort", ascending=False).reset_index(drop=True)


def render_html_metrics_table(display_df: pd.DataFrame, best_model: str, best_suffix: str) -> None:
    headers = "".join(f"<th>{col}</th>" for col in display_df.columns)
    body_rows = []
    for _, row in display_df.iterrows():
        cls = ' class="best-recall-row"' if row["Model"] == best_model else ""
        cells = "".join(f"<td>{row[col]}</td>" for col in display_df.columns)
        body_rows.append(f"<tr{cls}>{cells}</tr>")
    st.markdown(
        f"""
        <div class="metrics-table-wrap">
        <table class="metrics-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{"".join(body_rows)}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if best_model:
        st.caption(f"En iyi satır: {best_model} ({best_suffix})")


def render_prediction_comparison_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("Karşılaştırma tablosu oluşturulamadı.")
        return

    best_name = None
    if df["is_best_recall"].any():
        best_name = str(df.loc[df["is_best_recall"], "Model"].iloc[0])

    display = pd.DataFrame(
        {
            "Model": df["Model"],
            "Tahmin": df["Tahmin"],
            "Diyabet Olasılığı": df["Diyabet Olasılığı"].apply(
                lambda x: f"{x:.1%}" if x is not None else "—"
            ),
            "Sağlıklı Olasılığı": df["Sağlıklı Olasılığı"].apply(
                lambda x: f"{x:.1%}" if x is not None else "—"
            ),
            "Recall": df["Recall"].apply(lambda x: f"{x:.1%}" if x is not None else "—"),
            "Açıklama": df["Açıklama"],
        }
    )
    render_html_metrics_table(display, best_name, "Recall")

    if best_name:
        st.markdown(
            f'<p style="font-size:0.9rem;color:#475569;margin-top:0.5rem;">'
            f"Recall metriğine göre en güvenilir model: <strong>{best_name}</strong>. "
            f"Bu model diyabetli bireyleri yakalamada en başarılı sonucu vermiştir.</p>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Uygulamanın ana karar modeli HistGradientBoosting + SMOTE + recall odaklı eşik modelidir."
    )


def _combo_cv_scores(metrics: dict, combo_key: str, legacy_scores_key: str) -> dict:
    combo = metrics.get(combo_key) or metrics.get("best_overall") or {}
    if combo.get("cv_scores"):
        return combo["cv_scores"]
    return metrics.get(legacy_scores_key) or {}


def _format_combo_classifier(classifier: str | None) -> str:
    if not classifier or classifier == "—":
        return "—"
    return MODEL_DISPLAY_NAMES.get(classifier, classifier.replace("_", " ").title())


def _format_sampling_category(category: str | None) -> str:
    if not category or category == "—":
        return "—"
    labels = {
        "undersampling": "Alt örnekleme",
        "oversampling": "Üst örnekleme",
        "none": "Yok",
    }
    return labels.get(category, category.replace("_", " ").title())


def _format_sampling_method(method: str | None) -> str:
    if not method or method == "—":
        return "—"
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", method)


def _render_combo_info(combo: dict) -> None:
    algo = _format_combo_classifier(combo.get("classifier"))
    sampling = _format_sampling_category(combo.get("sampling_category"))
    method = _format_sampling_method(combo.get("sampling_method"))
    st.markdown(
        f"""
        <div class="combo-info-grid">
            <div class="combo-info-item">
                <div class="combo-info-label">Algoritma</div>
                <div class="combo-info-value">{algo}</div>
            </div>
            <div class="combo-info-item">
                <div class="combo-info-label">Örnekleme</div>
                <div class="combo-info-value">{sampling}</div>
            </div>
            <div class="combo-info-item">
                <div class="combo-info-label">Yöntemi</div>
                <div class="combo-info-value">{method}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_score_metrics(scores: dict, recall_first: bool = False) -> None:
    if not scores:
        st.warning("Metrik bulunamadı.")
        return
    keys = (
        ["recall", "precision", "f1", "accuracy", "roc_auc"]
        if recall_first
        else ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "roc_auc": "ROC-AUC",
    }
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        if key in scores:
            with col:
                st.metric(labels[key], fmt_metric(scores[key]))


def _humanize_key(key: str) -> str:
    if key in _HYPERPARAM_LABELS:
        return _HYPERPARAM_LABELS[key]
    return key.replace("model__", "").replace("_", " ").strip().title()


def render_hyperparams_table(params: dict) -> None:
    rows = []
    for k, v in params.items():
        if isinstance(v, float):
            val = fmt_metric(v, 6) if abs(v) < 1 else fmt_metric(v, 4)
        else:
            val = str(v)
        rows.append({"Parametre": _humanize_key(k), "Değer": val})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def classification_report_to_df(report: dict) -> pd.DataFrame:
    class_names = {"0": "Sağlıklı (0)", "1": "Diyabet (1)"}
    rows = []
    for cls in ("0", "1"):
        if cls not in report:
            continue
        r = report[cls]
        rows.append(
            {
                "Sınıf": class_names.get(cls, cls),
                "Precision": fmt_metric(r["precision"]),
                "Recall": fmt_metric(r["recall"]),
                "F1": fmt_metric(r["f1-score"]),
                "Destek (n)": str(
                    int(r["support"]) if float(r["support"]).is_integer() else r["support"]
                ),
            }
        )
    if "accuracy" in report:
        rows.append(
            {
                "Sınıf": "Genel doğruluk",
                "Precision": "—",
                "Recall": "—",
                "F1": fmt_metric(report["accuracy"]),
                "Destek (n)": "—",
            }
        )
    for avg_key, label in (("macro avg", "Makro ortalama"), ("weighted avg", "Ağırlıklı ortalama")):
        if avg_key in report:
            r = report[avg_key]
            rows.append(
                {
                    "Sınıf": label,
                    "Precision": fmt_metric(r["precision"]),
                    "Recall": fmt_metric(r["recall"]),
                    "F1": fmt_metric(r["f1-score"]),
                    "Destek (n)": str(
                    int(r["support"]) if float(r["support"]).is_integer() else r["support"]
                ),
                }
            )
    return pd.DataFrame(rows)


def render_classification_report(report: dict, *, title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")
    st.dataframe(classification_report_to_df(report), use_container_width=True, hide_index=True)


def render_key_value_metrics(
    data: dict,
    *,
    labels: dict[str, str] | None = None,
    formatters: dict[str, callable] | None = None,
    n_cols: int = 3,
) -> None:
    labels = labels or {}
    formatters = formatters or {}
    items = [(k, v) for k, v in data.items() if not isinstance(v, (list, dict))]
    if not items:
        return
    cols = st.columns(min(n_cols, len(items)))
    for col, (key, val) in zip(cols, items):
        label = labels.get(key, _humanize_key(key))
        if key in formatters:
            display = formatters[key](val)
        elif isinstance(val, float):
            display = f"{val:.1%}" if 0 <= val <= 1 and key.startswith("ratio") else fmt_metric(val, 4)
        else:
            display = str(val)
        with col:
            st.metric(label, display)


def render_class_imbalance(imb: dict) -> None:
    notes = imb.get("notes")
    if notes:
        st.info(notes)
    counts = imb.get("counts") or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sağlıklı (0)", counts.get(0, counts.get("0", "—")))
    with c2:
        st.metric("Diyabet (1)", counts.get(1, counts.get("1", "—")))
    with c3:
        total = sum(int(v) for v in counts.values()) if counts else 0
        st.metric("Toplam", total if total else "—")


def render_overfitting_mitigation_section(metrics: dict) -> None:
    """Overfitting azaltma özetini kart ve tablo olarak gösterir."""
    st.markdown("**Overfitting azaltma**")

    gb_info = metrics.get("gradient_boosting_hyperparameter_search") or {}
    gap = gb_info.get("cv_fold_train_val_gap") or {}
    iter_summary = gb_info.get("hist_gb_iteration_curve_summary") or {}

    st.markdown(
        '<div class="of-approach-card">'
        "<strong>Yaklaşım:</strong> Derin öğrenme (dropout, batch norm) yerine "
        "<strong>sklearn HistGradientBoosting</strong> kullanıldı. Amaç: düşük train–validation "
        "farkı ile daha iyi genelleme."
        "</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="of-section"><h4>Üretim modeli (HistGradientBoosting)</h4></div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Teknik": "Erken durdurma", "Açıklama": "Doğrulama platosunda durdurma"},
                    {"Teknik": "L2 düzenlileştirme", "Açıklama": "l2_regularization"},
                    {"Teknik": "Ağaç derinliği", "Açıklama": "Düşük max_depth"},
                    {"Teknik": "Yaprak eşiği", "Açıklama": "Yüksek min_samples_leaf"},
                    {"Teknik": "Örnekleme", "Açıklama": "Satır alt örnekleme"},
                    {"Teknik": "SMOTE", "Açıklama": "Sınıf dengesizliği için sentetik örnek"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with col_r:
        st.markdown('<div class="of-section"><h4>Karşılaştırma modelleri</h4></div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Model": "Random Forest", "Varyans azaltma": "max_depth = 5"},
                    {"Model": "K-NN", "Varyans azaltma": "n_neighbors = 15, distance ağırlıklı"},
                    {"Model": "SVM", "Varyans azaltma": "Düşürülmüş C"},
                    {"Model": "Gradient Boosting", "Varyans azaltma": "subsample + sqrt max_features"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Karşılaştırma modelinde early_stopping kapalı; daha yüksek kapasite.")

    st.markdown('<div class="of-section"><h4>Ön işleme</h4></div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            [
                {"Adım": "Aykırı değer", "Yöntem": "IQR (1.5×) kıskaç"},
                {"Adım": "Ölçekleme", "Yöntem": "StandardScaler"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="of-section"><h4>Genelleme metrikleri</h4></div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    cv_gap = gap.get("mean_train_minus_val_accuracy")
    iter_gap = iter_summary.get("mean_train_minus_val_score_curve")
    n_splits = gap.get("n_splits")
    with m1:
        st.metric(
            "Train − val accuracy",
            fmt_metric(float(cv_gap), 4) if cv_gap is not None else "—",
            help="10-fold stratified CV ortalaması; küçük olması hedeflenir.",
        )
    with m2:
        st.metric(
            "Train − val (iterasyon)",
            fmt_metric(float(iter_gap), 4) if iter_gap is not None else "—",
            help="Boosting iterasyon skorları ortalama farkı.",
        )
    with m3:
        st.metric(
            "Ort. eğitim doğruluğu",
            f"{gap['mean_train_accuracy_cv']:.1%}" if gap.get("mean_train_accuracy_cv") is not None else "—",
        )
    with m4:
        st.metric(
            "CV kat sayısı",
            int(n_splits) if n_splits is not None else "—",
        )
    st.caption("Düşük train–validation farkı, modelin eğitim verisine aşırı uyum riskinin azaldığını gösterir.")


def render_cv_fold_gap(gap: dict) -> None:
    render_key_value_metrics(
        {k: v for k, v in gap.items() if k not in ("per_fold_train_accuracy", "per_fold_val_accuracy")},
        labels={
            "n_splits": "CV kat sayısı",
            "mean_train_accuracy_cv": "Ort. eğitim doğruluğu",
            "mean_val_accuracy_cv": "Ort. doğrulama doğruluğu",
            "mean_train_minus_val_accuracy": "Eğitim − doğrulama farkı",
        },
        n_cols=4,
    )
    train_folds = gap.get("per_fold_train_accuracy")
    val_folds = gap.get("per_fold_val_accuracy")
    if train_folds and val_folds:
        fold_df = pd.DataFrame(
            {
                "Fold": list(range(1, len(train_folds) + 1)),
                "Eğitim doğruluğu": train_folds,
                "Doğrulama doğruluğu": val_folds,
            }
        ).set_index("Fold")
        st.line_chart(fold_df, use_container_width=True)


def _model_sort_key(item: tuple[str, dict]) -> int:
    try:
        return MODEL_ORDER.index(item[0])
    except ValueError:
        return len(MODEL_ORDER)


def get_deployment_scores(metrics: dict, deploy_source: str) -> dict:
    if deploy_source == "validation":
        return metrics.get("validation_holdout_scores") or {}
    return metrics.get("best_cv_scores") or {}


def get_deployment_threshold(metrics: dict, deploy_source: str) -> float | None:
    dep = metrics.get("deployment_model") or {}
    key = (
        "optimal_threshold_validation_split"
        if deploy_source == "validation"
        else "optimal_threshold_full_data_oof"
    )
    val = dep.get(key)
    return float(val) if val is not None else None


def build_deployment_metrics_row(metrics: dict, deploy_source: str) -> dict:
    scores = get_deployment_scores(metrics, deploy_source)
    row = {
        "model_key": "deployment",
        "Model": DEPLOYMENT_TABLE_NAME,
    }
    for key in METRIC_KEYS:
        row[key] = float(scores[key]) if scores.get(key) is not None else float("nan")
    return row


def build_model_metrics_frames(
    pm: dict,
    ratio: str,
    mode: str,
    metrics: dict | None = None,
    selected_models: list[str] | None = None,
    deploy_source: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected_models or list(COMPARISON_MODEL_KEYS)
    rows = []
    for clf_name, sc in sorted(pm[ratio][mode].items(), key=_model_sort_key):
        if clf_name not in selected:
            continue
        row = {
            "model_key": clf_name,
            "Model": MODEL_DISPLAY_NAMES.get(clf_name, clf_name.replace("_", " ").title()),
        }
        for key in METRIC_KEYS:
            row[key] = float(sc[key])
        rows.append(row)

    if metrics and "deployment" in selected:
        rows.append(build_deployment_metrics_row(metrics, deploy_source))

    if not rows:
        empty = pd.DataFrame(columns=["model_key", "Model", *METRIC_KEYS])
        return empty, empty

    numeric_df = pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)
    display_df = numeric_df.copy()
    for col, key in zip(METRIC_COLUMNS, METRIC_KEYS):
        display_df[col] = display_df[key].apply(lambda x: fmt_metric(x) if pd.notna(x) else "—")
    display_df = display_df[["Model", *METRIC_COLUMNS]]
    return display_df, numeric_df


def pick_best_recall_model_name(numeric_df: pd.DataFrame) -> str:
    if numeric_df.empty:
        return ""
    ranked = numeric_df.sort_values(["recall", "f1"], ascending=False)
    return str(ranked.iloc[0]["Model"])


def render_recall_comparison_chart(numeric_df: pd.DataFrame) -> None:
    chart_df = numeric_df[["Model", "recall"]].copy()
    chart_df = chart_df.rename(columns={"recall": "Recall"})
    chart_df = chart_df.set_index("Model").sort_values("Recall", ascending=True)
    st.bar_chart(chart_df, use_container_width=True)


def render_production_model_card(metrics: dict, deploy_source: str = "test") -> None:
    deploy_scores = get_deployment_scores(metrics, deploy_source)
    th = get_deployment_threshold(metrics, deploy_source)
    source_label = DEPLOY_METRIC_SOURCE_LABELS.get(deploy_source, deploy_source)

    recall_txt = f"{deploy_scores['recall']:.1%}" if deploy_scores.get("recall") is not None else "—"
    prec_txt = f"{deploy_scores['precision']:.1%}" if deploy_scores.get("precision") is not None else "—"
    auc_txt = f"{deploy_scores['roc_auc']:.1%}" if deploy_scores.get("roc_auc") is not None else "—"
    th_txt = fmt_metric(float(th), 4) if th is not None else "—"

    st.markdown(
        """
        <div class="deploy-card">
            <p><strong>Tahminlerde kullanılan model:</strong>
            HistGradientBoosting + SMOTE + recall odaklı eşik</p>
            <p>Bu model gerçek tahminleri üretir. Diğer modeller karşılaştırma amaçlıdır.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="prod-metrics-grid">
            <div class="prod-metric-item">
                <div class="prod-metric-label">Model</div>
                <div class="prod-metric-value prod-metric-model">HistGradientBoosting</div>
            </div>
            <div class="prod-metric-item">
                <div class="prod-metric-label">Recall</div>
                <div class="prod-metric-value">{recall_txt}</div>
            </div>
            <div class="prod-metric-item">
                <div class="prod-metric-label">Precision</div>
                <div class="prod-metric-value">{prec_txt}</div>
            </div>
            <div class="prod-metric-item">
                <div class="prod-metric-label">ROC-AUC</div>
                <div class="prod-metric-value">{auc_txt}</div>
            </div>
            <div class="prod-metric-item">
                <div class="prod-metric-label">Eşik</div>
                <div class="prod-metric-value">{th_txt}</div>
            </div>
        </div>
        <p class="prod-metric-caption">Metrik kaynağı: {source_label}</p>
        """,
        unsafe_allow_html=True,
    )


def render_technical_details_expander(metrics: dict) -> None:
    with st.expander("Teknik detayları göster"):
        dep = metrics.get("deployment_model") or {}
        deploy_test = metrics.get("best_cv_scores") or {}
        deploy_val = metrics.get("validation_holdout_scores") or {}

        st.markdown("**Eşik ve üretim modeli**")
        band = dep.get("recall_target_band")
        band_txt = f"{band[0]:.0%}–{band[1]:.0%}" if band and len(band) == 2 else "%90–%95"
        st.caption(f"Recall hedef bandı: {band_txt}")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Val. eşik",
                fmt_metric(float(dep.get("optimal_threshold_validation_split", 0)), 4)
                if dep.get("optimal_threshold_validation_split") is not None
                else "—",
            )
        with c2:
            st.metric(
                "OOF tam veri eşiği",
                fmt_metric(float(dep.get("optimal_threshold_full_data_oof", 0)), 4)
                if dep.get("optimal_threshold_full_data_oof") is not None
                else "—",
            )
        st.markdown("**Test kümesi metrikleri (üretim modeli)**")
        _render_score_metrics(deploy_test, recall_first=True)

        if deploy_val:
            st.markdown("**Doğrulama kümesi metrikleri**")
            _render_score_metrics(deploy_val, recall_first=True)

        st.markdown("**Karşılaştırma: recall şampiyonu (10-fold CV)**")
        best_recall = metrics.get("best_recall_overall") or metrics.get("best_overall")
        if best_recall:
            _render_combo_info(best_recall)
            _render_score_metrics(
                _combo_cv_scores(metrics, "best_recall_overall", "reference_recall_best_full_data_cv"),
                recall_first=True,
            )

        st.markdown("**Karşılaştırma: F1 referansı (10-fold CV)**")
        best_f1 = metrics.get("best_f1_overall") or metrics.get("best_overall")
        if best_f1:
            _render_combo_info(best_f1)
            _render_score_metrics(
                _combo_cv_scores(metrics, "best_f1_overall", "reference_f1_best_full_data_cv"),
            )

        if metrics.get("model_selection_notes"):
            st.markdown("**Model seçim notları**")
            st.info(metrics["model_selection_notes"])

        rep_val = metrics.get("classification_report_validation_holdout")
        if rep_val:
            st.markdown("**Doğrulama kümesi — sınıflandırma raporu**")
            render_classification_report(rep_val)

        rep = metrics.get("classification_report_test_holdout")
        if rep:
            st.markdown("**Test kümesi — sınıflandırma raporu**")
            render_classification_report(rep)

        for cm_name, cm_title in (
            ("confusion_matrix_test.png", "Confusion matrix (test)"),
            ("confusion_matrix_best_model.png", "Confusion matrix (üretim)"),
            ("confusion_matrix_validation_holdout.png", "Confusion matrix (doğrulama)"),
        ):
            cm_path = FIGURES_DIR / cm_name
            if cm_path.exists():
                st.markdown(f"**{cm_title}**")
                st.image(str(cm_path), use_container_width=True)

        gb_info = metrics.get("gradient_boosting_hyperparameter_search")
        if gb_info:
            st.markdown("**Hiperparametre araması (HistGradientBoosting)**")
            st.write(
                f"{gb_info.get('classifier', 'HistGradientBoosting')} · {gb_info.get('method')} · "
                f"{gb_info.get('n_iter')} iterasyon · {gb_info.get('inner_cv_folds')}-fold iç CV"
            )
            if gb_info.get("best_params"):
                render_hyperparams_table(gb_info["best_params"])
            if gb_info.get("cv_fold_train_val_gap"):
                st.markdown("**CV fold eğitim / doğrulama farkı**")
                render_cv_fold_gap(gb_info["cv_fold_train_val_gap"])

        if metrics.get("overfitting_mitigation_summary") or metrics.get(
            "gradient_boosting_hyperparameter_search"
        ):
            render_overfitting_mitigation_section(metrics)

        imb = metrics.get("class_imbalance")
        if imb:
            st.markdown("**Sınıf dengesizliği**")
            render_class_imbalance(imb)

        eda = load_eda_summary()
        if eda:
            st.markdown("**Veri seti özeti**")
            st.write(f"Örnek: {eda['rows']} · Özellik: {eda['cols'] - 1}")
            target_ratio = eda.get("target_ratio")
            if target_ratio:
                st.write(
                    f"Sağlıklı: {target_ratio.get(0, 0) * 100:.1f}% · "
                    f"Diyabet: {target_ratio.get(1, 0) * 100:.1f}%"
                )

        st.markdown("**Sistem bilgileri**")
        st.write(f"Python: {metrics.get('python_version', '—').split()[0]}")
        libs = metrics.get("library_versions") or {}
        st.write(f"scikit-learn: {libs.get('sklearn', '—')}")


def render_model_analysis_page(metrics: dict | None) -> None:
    if not metrics:
        st.warning("Model metrikleri bulunamadı. `python main.py train` ile eğitimi çalıştırın.")
        return

    st.markdown("## Model Analizi")
    st.markdown(
        '<div class="recall-note">'
        "Bu projede diyabetli bireyleri kaçırmamak kritik olduğu için model karşılaştırmasında "
        "Recall metriği ön plana çıkarılmıştır. Tabloda tüm modeller gösterilmiştir."
        "</div>",
        unsafe_allow_html=True,
    )

    pm = load_per_model_test_metrics()
    if pm:
        ratio_options = sorted(pm.keys())
        default_ratio = DEFAULT_TEST_RATIO if DEFAULT_TEST_RATIO in ratio_options else ratio_options[-1]
        default_mode = (
            DEFAULT_PREPROCESS_MODE
            if DEFAULT_PREPROCESS_MODE in pm.get(default_ratio, {})
            else "plain"
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            ratio = st.selectbox(
                "Test oranı",
                options=ratio_options,
                index=ratio_options.index(default_ratio),
                format_func=lambda x: f"{float(x):.0%}",
            )
        with col_b:
            mode = st.selectbox(
                "Ön işleme",
                options=["plain", "smote"],
                index=1 if default_mode == "smote" else 0,
                format_func=lambda x: "Örneklemesiz" if x == "plain" else "SMOTE",
            )
        with col_c:
            deploy_source = st.selectbox(
                "Üretim modeli metrikleri",
                options=["test", "validation"],
                index=0,
                format_func=lambda x: DEPLOY_METRIC_SOURCE_LABELS[x],
                help=(
                    "HistGradientBoosting için Recall 85.2%, Precision 53.5%, "
                    "ROC-AUC 83.1%, Eşik 0.2521 değerleri Test hold-out seçeneğindedir."
                ),
            )

        selected_models = st.multiselect(
            "Gösterilecek modeller",
            options=ANALYSIS_MODEL_SELECT_KEYS,
            default=ANALYSIS_MODEL_SELECT_KEYS,
            format_func=lambda k: ANALYSIS_MODEL_OPTIONS[k],
            help="Üretim modelini seçerek tabloya HistGradientBoosting satırını ekleyebilirsiniz.",
        )

        if not selected_models:
            st.warning("En az bir model seçin.")
        else:
            display_df, numeric_df = build_model_metrics_frames(
                pm,
                ratio,
                mode,
                metrics=metrics,
                selected_models=selected_models,
                deploy_source=deploy_source,
            )
            if numeric_df.empty:
                st.warning("Seçilen modeller için metrik bulunamadı.")
            else:
                best_model = pick_best_recall_model_name(numeric_df)

                st.markdown("##### Model karşılaştırması")
                render_html_metrics_table(display_df, best_model, "Recall, eşitlikte F1-Score")
                if "deployment" in selected_models:
                    st.caption(
                        "HistGradientBoosting satırı, üretim modeli metriklerinden gelir; "
                        "test oranı ve ön işleme seçiminden etkilenmez."
                    )

                st.markdown("##### Modellere Göre Recall Karşılaştırması")
                render_recall_comparison_chart(numeric_df)
    else:
        st.warning("Karşılaştırma tablosu için `per_model_test_metrics.json` gerekli.")
        deploy_source = "test"

    st.markdown("##### Üretim modeli")
    render_production_model_card(metrics, deploy_source if pm else "test")

    render_technical_details_expander(metrics)


def render_number_input(feature_key: str, column) -> float:
    info = FEATURE_INFO[feature_key]
    is_integer_field = feature_key in {"Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "Age"}

    if is_integer_field:
        kwargs = dict(
            label=info["label"],
            min_value=int(info["min"]),
            value=int(info["min"]),
            step=1,
            help=info["description"],
            key=feature_key,
        )
        if info.get("max") is not None:
            kwargs["max_value"] = int(info["max"])
        return column.number_input(**kwargs)

    kwargs = dict(
        label=info["label"],
        min_value=float(info["min"]),
        value=float(info["min"]),
        step=0.1 if feature_key == "Glucose" else 0.01,
        help=info["description"],
        key=feature_key,
    )
    if info.get("max") is not None:
        kwargs["max_value"] = float(info["max"])
    return column.number_input(**kwargs)


def render_prediction_page() -> None:
    st.markdown("## Tahmin")
    st.caption("Klinik ölçümleri girin; ana model ve karşılaştırma modelleri tahmin üretir.")

    deploy_model = load_model()
    if deploy_model is None:
        st.warning("Kayıtlı model bulunamadı. Önce `python main.py train` komutunu çalıştırın.")
        return

    col1, col2 = st.columns(2)
    input_dict = {}
    for idx, feature in enumerate(FEATURES_LIST):
        col = col1 if idx % 2 == 0 else col2
        input_dict[feature] = render_number_input(feature, col)

    _, btn_col, _ = st.columns([1, 1.2, 1])
    with btn_col:
        predict_button = st.button("Tahmin Et", type="primary", use_container_width=True)

    if not predict_button:
        return

    input_df = pd.DataFrame([input_dict])
    metrics = load_metrics()
    pm = load_per_model_test_metrics()
    recall_lookup = get_recall_lookup(pm, metrics)

    with st.spinner("Tahmin hesaplanıyor..."):
        prediction, probability = predict_diabetes(deploy_model, input_df)
        if not ensure_comparison_models_saved():
            st.warning(
                "Karşılaştırma modelleri yüklenemedi. "
                "`data/raw/diabetes.csv` mevcut olmalı; ilk tahminde modeller otomatik kaydedilir."
            )
        comparison_models = load_all_comparison_models()
        pred_rows = build_all_model_predictions(input_df, deploy_model, comparison_models)
        comparison_df = enrich_prediction_rows(pred_rows, recall_lookup)

    risk_text, risk_class = get_risk_level(probability)
    healthy_prob = 1.0 - probability
    card_class = f"result-card result-card--{risk_class}"

    st.markdown("### Tahminlerde kullanılan ana model")
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="card-tag">HistGradientBoosting + SMOTE + recall threshold</div>
            <h3>{risk_text}</h3>
            <p>Diyabet olasılığı: <strong>{probability:.1%}</strong></p>
            <p>Sağlıklı olma olasılığı: <strong>{healthy_prob:.1%}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(get_short_message(prediction, probability))

    st.markdown("### Tüm Modellerin Tahmin Karşılaştırması")
    render_prediction_comparison_table(comparison_df)

    with st.expander("Girilen değerleri göster"):
        unit_suffix = lambda k: f" {FEATURE_INFO[k]['unit']}" if FEATURE_INFO[k]["unit"] else ""
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Özellik": FEATURE_INFO[k]["label"],
                        "Değer": f"{v:.2f}{unit_suffix(k)}".strip(),
                    }
                    for k, v in input_dict.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_about_page() -> None:
    st.markdown("## Proje Hakkında")

    st.markdown(
        """
        <div class="info-card">
            <p><strong>Amaç:</strong> Pima Indians veri seti üzerinde makine öğrenmesi ile diyabet riski tahmini.</p>
            <p><strong>Veri:</strong> 768 örnek, 8 özellik · ikili sınıflandırma</p>
            <p><strong>Odak:</strong> Recall — diyabetli bireyleri kaçırmamak</p>
            <p><strong>Ders:</strong> Örüntü Tanıma — Ege Üniversitesi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eda = load_eda_summary()
    if eda:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Örnek sayısı", eda["rows"])
        with c2:
            st.metric("Özellik", eda["cols"] - 1)
        with c3:
            if eda.get("target_ratio"):
                st.metric("Sağlıklı oranı", f"{eda['target_ratio'].get(0, 0) * 100:.1f}%")

    with st.expander("Özellik açıklamaları"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Özellik": info["label"],
                        "Açıklama": info["description"],
                        "Birim": info["unit"] or "—",
                    }
                    for info in FEATURE_INFO.values()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Teknik yığın ve uyarı"):
        st.markdown(
            """
            **Teknolojiler:** Python, scikit-learn, pandas, Streamlit, imbalanced-learn (SMOTE)

            **Tahmin modeli:** HistGradientBoosting + SMOTE + recall odaklı eşik

            **Uyarı:** Bu uygulama tıbbi tanı aracı değildir. Sonuçlar yalnızca eğitim amaçlıdır;
            sağlık kararları için mutlaka bir hekime başvurun.
            """
        )

    with st.expander("Kaynaklar"):
        st.markdown(
            "- [UCI — Pima Indians Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)\n"
            "- [scikit-learn](https://scikit-learn.org/)\n"
            "- [Streamlit](https://docs.streamlit.io/)"
        )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### Menü")
        page = st.radio("Bölüm", MENU_OPTIONS, label_visibility="collapsed")
        st.markdown("---")
        st.caption(
            "Ana tahmin: HistGradientBoosting + SMOTE + recall eşiği. "
            "Karşılaştırma tablolarında Recall ön plandadır."
        )
    return page


def render_header() -> None:
    letters = list("Diyabet Risk Analizi")
    spans = []
    for i, ch in enumerate(letters):
        if ch == " ":
            spans.append('<span class="hero-space"></span>')
            continue
        delay = i * 0.04
        rotate = (-3 + (i % 7)) if i % 2 == 0 else (2 - (i % 5))
        spans.append(
            f'<span class="hero-letter" style="animation-delay:{delay:.2f}s;'
            f'transform:rotate({rotate}deg)">{ch}</span>'
        )
    st.markdown(
        f'<div class="hero-title">{"".join(spans)}</div>',
        unsafe_allow_html=True,
    )


# ======================== UYGULAMANIN ANA BÖLÜMÜ ========================


def main():
    render_header()
    metrics_global = load_metrics()
    page = render_sidebar()

    if page == "Tahmin":
        render_prediction_page()
    elif page == "Model Analizi":
        render_model_analysis_page(metrics_global)
    elif page == "Proje Hakkında":
        render_about_page()


if __name__ == "__main__":
    main()
