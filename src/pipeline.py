from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import (
    ADASYN,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
    SMOTE,
    SVMSMOTE,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import (
    AllKNN,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.paths import DISPLAY_ZERO_EPS, Paths, get_paths, snap_display_scalar


SEED = 42


class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """
    Eğitim kümesinde sütun bazlı Q1/Q3 ve IQR ile alt/üst sınırlar hesaplar;
    dönüşümde değerleri bu aralığa kırpar (KNN/SVM/ölçekleme öncesi gürültü azaltma).
    IQR≈0 olan sütunlarda kırpma yapılmaz.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = float(factor)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25.0, axis=0)
        q3 = np.nanpercentile(X, 75.0, axis=0)
        iqr = q3 - q1
        pad = self.factor * iqr
        self.lo_ = q1 - pad
        self.hi_ = q3 + pad
        flat = (iqr <= 1e-12) | ~np.isfinite(iqr)
        self.lo_ = self.lo_.copy()
        self.hi_ = self.hi_.copy()
        self.lo_[flat] = -np.inf
        self.hi_[flat] = np.inf
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)


def set_seed(seed: int = SEED) -> None:
    # Tekrarlanabilir sonuçlar için rastgelelik sabitlenir.
    random.seed(seed)
    np.random.seed(seed)


def format_display_float(value: float, decimals: int = 3) -> str:
    """Türkçe virgül; sıfıra yakın negatif/pozitifleri sıfırla."""
    v = snap_display_scalar(float(value))
    return f"{v:.{decimals}f}".replace(".", ",")


def _format_grouped_bin_axis_labels(index: pd.Index, snap_eps: float = 0.002) -> list[str]:
    """pd.Interval x ekseni etiketlerinde (-0.001, ... gibi sınırları 0 olarak göster."""
    labels: list[str] = []
    for ix in index:
        if isinstance(ix, pd.Interval):
            lo = float(ix.left)
            hi = float(ix.right)
            if abs(lo) < snap_eps:
                lo = 0.0
            if abs(hi) < snap_eps:
                hi = 0.0
            lc = "[" if ix.closed_left else "("
            rc = "]" if ix.closed_right else ")"
            lo_t = f"{lo:.2f}".replace(".", ",")
            hi_t = f"{hi:.2f}".replace(".", ",")
            labels.append(f"{lc}{lo_t}, {hi_t}{rc}")
        else:
            labels.append(str(ix))
    return labels


class ThresholdedBinaryClassifier:
    """İkili sınıflandırıcı: predict_proba aynı kalır, predict eşik ile üretilir (recall optimizasyonu)."""

    def __init__(self, pipeline: Any, threshold: float = 0.5) -> None:
        self.pipeline = pipeline
        self.threshold = float(threshold)

    def predict(self, X: pd.DataFrame | Any) -> np.ndarray:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        proba_pos = self.pipeline.predict_proba(X_df)[:, 1]
        return (proba_pos >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame | Any) -> np.ndarray:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict_proba(X_df)


def download_kaggle_dataset(dataset_ref: str = "amineipad/diabetes-dataset") -> Path:
    import kagglehub

    # Veri indirildikten sonra standart bir konuma kopyalanır.
    paths = get_paths()
    paths.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_folder = Path(kagglehub.dataset_download(dataset_ref))
    csv_candidates = list(downloaded_folder.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("Kaggle veri kümesi klasöründe CSV bulunamadı.")
    shutil.copy2(csv_candidates[0], paths.raw_data_path)
    return paths.raw_data_path


def load_data(csv_path: Path | None = None) -> pd.DataFrame:
    # Öncelik: parametre ile verilen yol -> varsayılan yol -> data/raw içindeki ilk CSV.
    paths = get_paths()
    target_path = csv_path or paths.raw_data_path
    if not target_path.exists():
        csv_candidates = sorted(paths.raw_data_path.parent.glob("*.csv"))
        if csv_candidates:
            target_path = csv_candidates[0]
        else:
            raise FileNotFoundError(
                "Veri dosyası bulunamadı. `python main.py download` komutunu çalıştırın."
            )
    return pd.read_csv(target_path)


def infer_target_column(df: pd.DataFrame) -> str:
    # Yaygın hedef sütun isimlerinden birini yakalamaya çalışır.
    # Hiçbiri yoksa son sütunu hedef kabul eder.
    candidate_names = {"outcome", "class", "target", "diabetes", "label"}
    for col in df.columns:
        if col.lower() in candidate_names:
            return col
    return df.columns[-1]


def _compact_legend_bar_charts(ax) -> None:
    # context="talk" ile büyük varsayılan yazı; lejant çubukların üstüne binmesin diye sağ dışa alınır.
    # Seaborn hue lejantında başlık (ör. "setting") kaldırılır; handles açıkça verilir.
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        title=None,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=False,
        borderpad=0.25,
        labelspacing=0.35,
        handlelength=1.0,
        handletextpad=0.35,
    )


def run_eda(df: pd.DataFrame, target_col: str) -> Dict:
    # EDA çıktıları için klasörleri garanti altına al.
    paths = get_paths()
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    # Önceki EDA PNG'lerini sil; aynı isimle temiz yeniden üretim (eski lejant/ölçek kalıntısı kalmaz).
    for stale_fig in paths.figures_dir.glob("*.png"):
        stale_fig.unlink(missing_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.25)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Model eğitiminde veri sızıntısı oluşturmaması için ID benzeri sütunları tespit et.
    id_like_columns = [
        c for c in df.columns if c.lower() in {"id", "index", "record_id", "patient_id"}
    ]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Sayısal sütunlarda IQR ile aykırı değer adedini raporla.
    outlier_iqr_counts = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_iqr_counts[col] = 0
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_iqr_counts[col] = int(((df[col] < low) | (df[col] > high)).sum())

    # EDA özetini JSON'a yazmak için sözlükte topla.
    eda = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "id_like_columns": id_like_columns,
        "missing_values": df.isna().sum().to_dict(),
        "target_distribution": df[target_col].value_counts().to_dict(),
        "target_ratio": (df[target_col].value_counts(normalize=True).round(4)).to_dict(),
        "outlier_iqr_counts": outlier_iqr_counts,
    }

    # Grafiklerde sınıf etiketlerini daha okunabilir Türkçe göster.
    outcome_name_map = {0: "Sağlıklı", 1: "Diyabet Hastası"}
    display_name_map = {
        "Pregnancies": "Hamilelik",
        "Glucose": "Glikoz",
        "BloodPressure": "Kan Basıncı",
        "SkinThickness": "Cilt Kalınlığı",
        "Insulin": "İnsülin",
        "BMI": "VKİ",
        "DiabetesPedigreeFunction": "Pedigri",
        "Age": "Yaş",
    }
    df_plot = df.copy()
    df_plot["OutcomeLabel"] = df_plot[target_col].map(outcome_name_map).fillna(df_plot[target_col].astype(str))
    generated_group_plots = []

    def _plot_grouped_outcome_distribution(
        feature_col: str,
        feature_label: str,
        custom_bins: list[float] | None = None,
        custom_labels: list[str] | None = None,
        right: bool = True,
        n_quantiles: int = 5,
    ) -> str | None:
        # Sayısal kolonları aralıklara bölüp hedef sınıf dağılımını gruplu sütun grafik olarak üretir.
        if feature_col not in df_plot.columns:
            return None
        if not pd.api.types.is_numeric_dtype(df_plot[feature_col]):
            return None

        feature_series = df_plot[feature_col]
        if custom_bins is not None:
            grouped_series = pd.cut(
                feature_series,
                bins=custom_bins,
                labels=custom_labels,
                include_lowest=True,
                right=right,
            )
        else:
            non_null_nunique = int(feature_series.dropna().nunique())
            if non_null_nunique < 2:
                return None
            q_count = min(n_quantiles, non_null_nunique)
            grouped_series = pd.qcut(feature_series, q=q_count, duplicates="drop")

        group_counts = pd.crosstab(grouped_series, df_plot["OutcomeLabel"]).fillna(0).astype(int)
        if group_counts.shape[0] < 2:
            return None
        if custom_labels is not None:
            group_counts = group_counts.reindex(custom_labels).fillna(0).astype(int)
        group_counts.index = _format_grouped_bin_axis_labels(group_counts.index)

        for col in ["Sağlıklı", "Diyabet Hastası"]:
            if col not in group_counts.columns:
                group_counts[col] = 0
        group_counts = group_counts[["Sağlıklı", "Diyabet Hastası"]]

        plt.figure(figsize=(11, 6.2))
        ax = group_counts.plot(kind="bar", width=0.78, color=["#5DA5DA", "#F28E2B"])
        plt.title(f"{feature_label} Gruplarına Göre Hastalık Dağılımı", fontsize=16)
        plt.xlabel(f"{feature_label} Grupları", fontsize=14)
        plt.ylabel("Kayıt Sayısı", fontsize=14)
        plt.xticks(rotation=20, ha="right")
        _compact_legend_bar_charts(ax)
        ax.set_ylim(bottom=0)
        for container in ax.containers:
            labels = []
            for rect in container:
                h = float(rect.get_height())
                if abs(h) < 0.5:
                    h = 0.0
                labels.append(str(int(round(h))))
            ax.bar_label(container, labels=labels, padding=4, fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        safe_feature = "".join(ch.lower() if ch.isalnum() else "_" for ch in feature_col).strip("_")
        file_name = f"{safe_feature}_grouped_outcome_distribution.png"
        plt.savefig(paths.figures_dir / file_name, dpi=140)
        plt.close()
        return file_name

    # A) Hedef sınıf dağılımı: sayı grafiği
    plt.figure(figsize=(9.5, 5.8))
    ax = sns.countplot(
        x="OutcomeLabel",
        data=df_plot,
        hue="OutcomeLabel",
        order=["Sağlıklı", "Diyabet Hastası"],
        palette="pastel",
        legend=False,
    )
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=13,
        )
    plt.title("Hedef Sınıf Dağılımı", fontsize=17)
    plt.xlabel("Sınıf", fontsize=14)
    plt.ylabel("Kayıt Sayısı", fontsize=14)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "class_distribution.png", dpi=140)
    plt.close()

    # Makaledeki şekle benzer pasta grafiği (yüzde + adet bilgisi).
    class_counts = df[target_col].value_counts().sort_index()
    label_map = {0: "Sağlıklı", 1: "Diyabet Hastası"}
    pie_labels = [label_map.get(cls, str(cls)) for cls in class_counts.index]
    total_count = int(class_counts.sum())

    def _autopct_with_count(pct: float) -> str:
        count = int(round(pct * total_count / 100.0))
        pct_snapped = snap_display_scalar(pct)
        pct_text = f"{pct_snapped:.1f}".replace(".", ",")
        return f"%{pct_text}\n(n={count})"

    plt.figure(figsize=(9.5, 7.2))
    wedges, texts, autotexts = plt.pie(
        class_counts.values,
        labels=pie_labels,
        autopct=_autopct_with_count,
        startangle=90,
        colors=["#5DA5DA", "#F28E2B"],
        wedgeprops={"edgecolor": "white", "linewidth": 1.4},
        textprops={"fontsize": 15, "weight": "semibold"},
        pctdistance=0.72,
        labeldistance=1.08,
    )
    plt.setp(texts, fontsize=16, weight="semibold")
    plt.setp(autotexts, fontsize=15, weight="bold")
    plt.title("Veri Setindeki Diyabet ve Sağlıklı Kayıt Dağılımı", fontsize=18, weight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "class_distribution_pie.png", dpi=140)
    plt.close()

    # C) Yaş gruplarına göre sınıf dağılımı: gruplanmış sütun grafiği
    if "Age" in df.columns:
        age_labels = ["21-30", "31-40", "41-50", "51-60", "61+"]
        age_bins = [20, 30, 40, 50, 60, np.inf]
        age_plot_file = _plot_grouped_outcome_distribution(
            feature_col="Age",
            feature_label="Yaş",
            custom_bins=age_bins,
            custom_labels=age_labels,
            right=True,
        )
        if age_plot_file:
            # Mevcut rapor akışını bozmamak için eski dosya adı da korunur.
            shutil.copy2(
                paths.figures_dir / age_plot_file,
                paths.figures_dir / "age_group_outcome_distribution.png",
            )
            generated_group_plots.append(age_plot_file)

    # D) BMI kategorilerine göre yığılmış sütun grafiği
    if "BMI" in df.columns:
        bmi_labels = ["Zayif", "Normal", "Fazla Kilolu", "Obez"]
        bmi_bins = [-np.inf, 18.5, 25, 30, np.inf]
        df_plot["BMICategory"] = pd.cut(
            df_plot["BMI"], bins=bmi_bins, labels=bmi_labels, include_lowest=True, right=False
        )
        # Her BMI kategorisinde sınıf dağılımını toplu görmek için çapraz tablo.
        bmi_counts = (
            pd.crosstab(df_plot["BMICategory"], df_plot["OutcomeLabel"])
            .reindex(bmi_labels)
            .fillna(0)
            .astype(int)
        )
        for col in ["Sağlıklı", "Diyabet Hastası"]:
            if col not in bmi_counts.columns:
                bmi_counts[col] = 0
        bmi_counts = bmi_counts[["Sağlıklı", "Diyabet Hastası"]]

        ax = bmi_counts.plot(
            kind="bar",
            stacked=True,
            figsize=(10.5, 6.2),
            color=["#8ECFC9", "#FFBE7A"],
            width=0.8,
        )
        plt.title("BMI Kategorilerine Göre Hedef Sınıf Dağılımı", fontsize=16)
        plt.xlabel("BMI Kategorisi", fontsize=14)
        plt.ylabel("Kayıt Sayısı", fontsize=14)
        plt.xticks(rotation=0)
        _compact_legend_bar_charts(ax)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(paths.figures_dir / "bmi_category_outcome_stacked.png", dpi=140)
        plt.close()

    # E) Glikoz dağılımı: sınıflara göre kutu grafiği
    if "Glucose" in df.columns:
        plt.figure(figsize=(9, 6))
        sns.boxplot(
            data=df_plot,
            x="OutcomeLabel",
            y="Glucose",
            hue="OutcomeLabel",
            palette="pastel",
            legend=False,
        )
        plt.title("Hedef Sınıfa Göre Glikoz Dağılımı (Kutu Grafiği)", fontsize=16)
        plt.xlabel("Sınıf", fontsize=14)
        plt.ylabel("Glucose", fontsize=14)
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "glucose_outcome_boxplot.png", dpi=140)
        plt.close()

    # F) BMI dağılımı: sınıflara göre keman grafiği
    if "BMI" in df.columns:
        plt.figure(figsize=(9, 6))
        sns.violinplot(
            data=df_plot,
            x="OutcomeLabel",
            y="BMI",
            hue="OutcomeLabel",
            palette="coolwarm",
            inner="quartile",
            legend=False,
        )
        plt.title("Hedef Sınıfa Göre BMI Dağılımı (Keman Grafiği)", fontsize=16)
        plt.xlabel("Sınıf", fontsize=14)
        plt.ylabel("BMI", fontsize=14)
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "bmi_outcome_violinplot.png", dpi=140)
        plt.close()

    if numeric_cols:
        plt.figure(figsize=(10, 7))
        corr = df[numeric_cols + [target_col]].corr(numeric_only=True)
        name_map = {**display_name_map, target_col: "Hedef Sınıf"}
        corr_display = corr.copy()
        corr_display.columns = [name_map.get(c, c) for c in corr_display.columns]
        corr_display.index = [name_map.get(c, c) for c in corr_display.index]
        mask = np.triu(np.ones_like(corr_display, dtype=bool))
        # Türkçe sayı gösterimi için ondalık ayırıcı virgüle çevrilir.
        corr_snapped = corr_display.map(lambda v: snap_display_scalar(float(v)) if pd.notna(v) else v)
        corr_annot = corr_snapped.round(2).astype(str).apply(lambda c: c.str.replace(".", ",", regex=False))
        sns.heatmap(
            corr_display,
            mask=mask,
            cmap="coolwarm",
            annot=corr_annot,
            fmt="",
            annot_kws={"size": 11},
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.85, "label": "Korelasyon"},
        )
        plt.title("Korelasyon Isı Haritası", fontsize=17)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "correlation_heatmap.png", dpi=140)
        plt.close()

    # G) Özellik önemi: Random Forest ile hangi değişkenlerin daha etkili olduğunu göster.
    feature_cols = [c for c in df.columns if c != target_col]
    if feature_cols:
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED)
        rf.fit(df[feature_cols], df[target_col])
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values(
            by="importance", ascending=False
        )
        plt.figure(figsize=(10.5, 6.5))
        ax = sns.barplot(data=fi_df, x="importance", y="feature", color="#7AA6DC")
        plt.title("Özellik Önemi (Random Forest)", fontsize=16)
        plt.xlabel("Önem Düzeyi", fontsize=14)
        plt.ylabel("Özellik", fontsize=14)
        for p in ax.patches:
            width = float(p.get_width())
            width_text = format_display_float(width, 3)
            ax.annotate(
                width_text,
                (width, p.get_y() + p.get_height() / 2),
                va="center",
                ha="left",
                fontsize=12,
            )
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "feature_importance_rf.png", dpi=140)
        plt.close()

        # En önemli sayısal değişkenler için (yaş dışında) otomatik grup bazlı dağılım grafikleri üret.
        top_important_numeric = [f for f in fi_df["feature"].tolist() if f in numeric_cols and f != "Age"][:4]
        for feature in top_important_numeric:
            plot_file = _plot_grouped_outcome_distribution(
                feature_col=feature,
                feature_label=display_name_map.get(feature, feature),
                n_quantiles=5,
            )
            if plot_file:
                generated_group_plots.append(plot_file)

        eda["important_numeric_features_for_group_plots"] = top_important_numeric
        eda["important_feature_group_plot_files"] = generated_group_plots

    with open(paths.metrics_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda, f, ensure_ascii=False, indent=2)
    return eda


def split_data(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # Veri train/val/test olarak 70/15/15 oranında ve stratified biçimde bölünür.
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Sayısal ve kategorik sütunları ayırarak farklı ön işleme adımları uygular.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    # Sayısal: median doldur → IQR ile aykırı kıskaç → StandardScaler (SVM/KNN için kritik).
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("iqr_clip", IQROutlierClipper(factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )
    # Kategorik alanlar: en sık değerle doldur + one-hot encode et.
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)]
    )


def _scores(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict:
    # Temel sınıflandırma metriklerini tek fonksiyonda toplar.
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # Olasılık bilgisi varsa ikili problem için ROC-AUC de hesaplanır.
    if y_prob is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return result


def _combo_dict_from_row(row: pd.Series) -> Dict[str, str]:
    return {
        "classifier": str(row["classifier"]),
        "sampling_category": str(row["sampling_category"]),
        "sampling_method": str(row["sampling_method"]),
    }


def _pipeline_for_result_row(
    row: pd.Series,
    pre: ColumnTransformer,
    samplers: Dict[str, object],
    classifiers: Dict[str, object],
) -> ImbPipeline:
    steps: list = [("preprocessor", pre)]
    if str(row["sampling_category"]) != "none":
        steps.append(("sampler", clone(samplers[str(row["sampling_method"])])))
    steps.append(("model", clone(classifiers[str(row["classifier"])])))
    return ImbPipeline(steps)


def _oof_eval_and_confusion_plot(
    pipe: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    title: str,
    figure_path: Path,
) -> Tuple[Dict, Dict]:
    y_cv_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict", n_jobs=-1)
    y_cv_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    cv_scores = _scores(y, y_cv_pred, y_cv_prob)
    class_rep = classification_report(y, y_cv_pred, output_dict=True, zero_division=0)
    conf = confusion_matrix(y, y_cv_pred)
    row_sum = np.maximum(conf.sum(axis=1, keepdims=True), 1e-9)
    conf_pct_row = conf.astype(float) / row_sum
    conf_annot = np.array(
        [
            [
                f"{conf[i, j]}\n({format_display_float(conf_pct_row[i, j] * 100, 1)}%)"
                for j in range(conf.shape[1])
            ]
            for i in range(conf.shape[0])
        ]
    )
    plt.figure(figsize=(7, 5.8))
    sns.heatmap(
        conf,
        annot=conf_annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Sağlıklı (0)", "Diyabet (1)"],
        yticklabels=["Sağlıklı (0)", "Diyabet (1)"],
        annot_kws={"size": 12},
    )
    plt.title(title, fontsize=15)
    plt.xlabel("Tahmin Edilen Sınıf", fontsize=13)
    plt.ylabel("Gerçek Sınıf", fontsize=13)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=140)
    plt.close()
    return cv_scores, class_rep


def _build_classifiers() -> Dict[str, object]:
    # Overfitting / yüksek varyans: düşük kapasite, güçlü düzenleme, KNN’de daha çok komşu.
    return {
        # SVM: daha düşük C → daha güçlü margin regularizasyonu (train–val makasını daraltma)
        "svm": SVC(kernel="rbf", C=0.45, gamma="scale", probability=True, random_state=SEED),
        # LogReg: hafif gevşek C (aşırı cezalandırma bazen val’da “şanslı” düşük loss üretir)
        "logreg": LogisticRegression(
            max_iter=1200, C=0.7, penalty="elasticnet", l1_ratio=0.32, solver="saga", random_state=SEED
        ),
        # KNN: daha fazla komşu + mesafe ağırlığı → sınır daha pürüzsüz, fold bazlı uç spike riski azalır
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance", metric="minkowski", p=2),
        # Random Forest: sığ ağaç (max_depth 5), yaprak/ bölünme eşikleri → ezber azalır
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_split=8,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=SEED,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=60,
            learning_rate=0.55,
            random_state=SEED,
        ),  # n_estimators train_and_compare başında doğrulama log-loss min. ile yeniden ayarlanır
        # sklearn GBC: satır alt örnekleme + yaprak eşiği ile varyans kontrolü
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=8,
            min_samples_leaf=6,
            subsample=0.85,
            max_features="sqrt",
            random_state=SEED,
        ),
    }


def _build_samplers() -> Dict[str, object]:
    # Fazla ve az örnekleme yöntemlerini tek noktadan yönet. SMOTE parametreleri optimize edildi.
    return {
        # SMOTE: k_neighbors optimize edildi (daha stabil sentinel üretiyor)
        "SMOTE": SMOTE(random_state=SEED, k_neighbors=5),
        "KMeansSMOTE": KMeansSMOTE(random_state=SEED),
        "RandomOverSampler": RandomOverSampler(random_state=SEED),
        "ADASYN": ADASYN(random_state=SEED),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=SEED),
        "SVMSMOTE": SVMSMOTE(random_state=SEED),
        "EditedNearestNeighbours": EditedNearestNeighbours(),
        "AllKNN": AllKNN(),
        "InstanceHardnessThreshold": InstanceHardnessThreshold(random_state=SEED),
        "NearMiss": NearMiss(),
        "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule(),
        "OneSidedSelection": OneSidedSelection(random_state=SEED),
        "RandomUnderSampler": RandomUnderSampler(random_state=SEED),
        "TomekLinks": TomekLinks(),
    }


def optimize_threshold_recall_band(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    low: float = 0.90,
    high: float = 0.95,
) -> float:
    """Validasyon olasılıklarında recall'u [low, high] bandına yaklaştıran eşik seçer; mümkün değilse recall'u maksimize eder."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(0.02, 0.995, 400)
    best_t = 0.5
    best_prec = -1.0
    found = False
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        if low <= r <= high:
            found = True
            p = precision_score(y_true, pred, zero_division=0)
            if p > best_prec:
                best_prec = p
                best_t = float(t)
    if found:
        return best_t
    # Bant mümkün değilse: recall'u hedef band merkezine en yakın eşik
    target = (low + high) / 2.0
    best_dist = 1e9
    best_t2 = 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        dist = abs(r - target)
        if dist < best_dist:
            best_dist = dist
            best_t2 = float(t)
    return best_t2


def evaluate_per_model_test_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    classifiers: Dict[str, object],
    test_sizes: Tuple[float, ...] = (0.15, 0.20, 0.25),
    seed: int = SEED,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Her test oranı için tüm modelleri train/test ile değerlendirir (düz ve SMOTE)."""
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for ts in test_sizes:
        key = f"{ts:.2f}"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, random_state=seed, stratify=y
        )
        out[key] = {"plain": {}, "smote": {}}
        for mode, use_smote in (("plain", False), ("smote", True)):
            for clf_name, clf in classifiers.items():
                local_pre = _make_preprocessor(X_train)
                if use_smote:
                    pipe = ImbPipeline(
                        [
                            ("preprocessor", local_pre),
                            ("sampler", SMOTE(random_state=seed, k_neighbors=3)),
                            ("model", clone(clf)),
                        ]
                    )
                else:
                    pipe = ImbPipeline([("preprocessor", local_pre), ("model", clone(clf))])
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                proba = pipe.predict_proba(X_test)[:, 1]
                out[key][mode][clf_name] = _scores(y_test, pred, proba)
    return out


def _hist_gb_deployment_estimator() -> HistGradientBoostingClassifier:
    """HistGradientBoosting: erken durdurma (val plato), L2, düşük derinlik — overfitting azaltma."""
    return HistGradientBoostingClassifier(
        random_state=SEED,
        loss="log_loss",
        early_stopping=True,
        validation_fraction=0.18,
        n_iter_no_change=14,
        tol=1e-4,
        max_iter=400,
        learning_rate=0.055,
        max_depth=3,
        min_samples_leaf=22,
        l2_regularization=4.0,
        max_bins=64,
    )


def plot_hist_gb_learning_curves(fitted_pipe: ImbPipeline, paths: Paths) -> Dict[str, float] | None:
    """Early stopping ile kayıtlı train / iç-doğrulama skor eğrileri (yüksek skor daha iyi)."""
    mod = fitted_pipe.named_steps.get("model")
    if mod is None:
        return None
    vscores = getattr(mod, "validation_score_", None)
    tscores = getattr(mod, "train_score_", None)
    if vscores is None or tscores is None or len(vscores) < 2:
        return None
    vscores = np.asarray(vscores, dtype=float)
    tscores = np.asarray(tscores, dtype=float)
    n = int(min(len(vscores), len(tscores)))
    vscores = vscores[:n]
    tscores = tscores[:n]
    iters = np.arange(1, n + 1)
    gap = float(np.mean(tscores - vscores))
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.plot(
        iters,
        tscores,
        lw=2.9,
        marker="o",
        markersize=5,
        markevery=max(1, n // 18),
        color="#2E7AD6",
        label="Eğitim skoru",
    )
    ax.plot(
        iters,
        vscores,
        lw=2.9,
        marker="s",
        markersize=5,
        markevery=max(1, n // 18),
        color="#E68619",
        label="Doğrulama skoru (early stopping)",
    )
    ax.set_xlabel("Artırma iterasyonu", fontsize=17, fontweight="semibold")
    ax.set_ylabel("Skor (yüksek daha iyi)", fontsize=17, fontweight="semibold")
    ax.set_title(
        "HistGradientBoosting — eğitim vs doğrulama (overfitting izleme)",
        fontsize=19,
        fontweight="bold",
        pad=14,
    )
    ax.legend(fontsize=15, loc="lower right")
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis="both", labelsize=14)
    note = f"Ort. train−val skor farkı: {format_display_float(gap, 4)}"
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        fontsize=14,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.4},
    )
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "training_boosting_iteration_scores.png", dpi=150)
    plt.close()
    return {"mean_train_minus_val_score_curve": gap, "n_iterations_recorded": float(n)}


def compute_all_models_cv_train_val_log_loss(
    X: pd.DataFrame,
    y: pd.Series,
    classifiers: Dict[str, object],
    n_splits: int = 10,
    seed: int = SEED,
) -> Dict[str, Dict[str, Any]]:
    """
    Her sınıflandırıcı için stratified K-fold CV: fold başına eğitim ve doğrulama log-loss özetleri.
    Sınıf oranı fold’lar arasında korunur (StratifiedKFold). Grafik üretilmez; yalnızca JSON/metrik için.
    """
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    summary: Dict[str, Dict[str, Any]] = {}

    for name, clf in classifiers.items():
        train_losses: list[float] = []
        val_losses: list[float] = []

        for tr_idx, va_idx in cv_inner.split(X, y):
            y_tr_fold = y.iloc[tr_idx]
            min_class = int(y_tr_fold.value_counts().min())
            k_sm = 3 if min_class >= 8 else max(1, min(2, min_class - 1))

            pre_fold = _make_preprocessor(X)
            pipe = ImbPipeline(
                [
                    ("preprocessor", pre_fold),
                    ("sampler", SMOTE(random_state=seed, k_neighbors=k_sm)),
                    ("model", clone(clf)),
                ]
            )
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            pipe.fit(X_tr, y_tr)

            if hasattr(pipe, "predict_proba"):
                p_tr = pipe.predict_proba(X_tr)
                p_va = pipe.predict_proba(X_va)
                train_losses.append(float(log_loss(y_tr, p_tr, labels=[0, 1])))
                val_losses.append(float(log_loss(y_va, p_va, labels=[0, 1])))
            else:
                train_losses.append(float(1.0 - accuracy_score(y_tr, pipe.predict(X_tr))))
                val_losses.append(float(1.0 - accuracy_score(y_va, pipe.predict(X_va))))

        summary[name] = {
            "mean_train_log_loss": float(np.mean(train_losses)),
            "mean_val_log_loss": float(np.mean(val_losses)),
            "per_fold_train_log_loss": train_losses,
            "per_fold_val_log_loss": val_losses,
        }

    return summary


def _holdout_smote_preprocessed_arrays(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Epoch / erken-durdurma grafikleri ve AdaBoost ayarı için ortak bölme:
    %75/%25 stratified train–val, median+IQR+scaler, SMOTE (k_neighbors veri kümesine göre).
    """
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    pre_fold = _make_preprocessor(X_tr)
    min_class = int(y_tr.value_counts().min())
    k_sm = 3 if min_class >= 8 else max(1, min(2, min_class - 1))
    X_tr_imp = pre_fold.fit_transform(X_tr, y_tr)
    X_va_imp = pre_fold.transform(X_va)
    sm = SMOTE(random_state=seed, k_neighbors=k_sm)
    X_sm, y_sm = sm.fit_resample(X_tr_imp, y_tr)
    y_sm_arr = np.asarray(y_sm).astype(int)
    y_va_arr = np.asarray(y_va).astype(int)
    return X_tr, y_tr, X_va, y_va, X_sm, y_sm_arr, X_va_imp, y_va_arr, k_sm


def _tune_adaboost_n_estimators_val_early_stop(
    classifiers: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    probe_cap: int = 200,
) -> tuple[Dict[str, object], Dict[str, Any]]:
    """
    AdaBoost: doğrulama log-loss'un minimum olduğu aşamada durdurma (sklearn'de erken durdurma yok;
    uygun n_estimators ile denk). Tüm grid CV bu ayarlanmış modeli kullanır.
    """
    ada = classifiers.get("adaboost")
    if not isinstance(ada, AdaBoostClassifier):
        return classifiers, {}
    lr = float(ada.get_params().get("learning_rate", 0.55))
    _X_tr, _y_tr, _X_va, _y_va, X_sm, y_sm_arr, X_va_imp, y_va_arr, _k_sm = _holdout_smote_preprocessed_arrays(X, y, seed)
    n_probe = int(min(probe_cap, max(100, ada.n_estimators * 3)))
    probe = AdaBoostClassifier(n_estimators=n_probe, learning_rate=lr, random_state=seed)
    probe.fit(X_sm, y_sm_arr)
    val_losses = [
        float(log_loss(y_va_arr, p, labels=[0, 1])) for p in probe.staged_predict_proba(X_va_imp)
    ]
    best_n = int(np.argmin(val_losses)) + 1
    best_n = max(6, min(best_n, n_probe))
    classifiers["adaboost"] = AdaBoostClassifier(n_estimators=best_n, learning_rate=lr, random_state=seed)
    meta = {
        "adaboost_n_estimators_after_val_early_stop": best_n,
        "adaboost_probe_n_estimators": n_probe,
        "adaboost_min_val_log_loss_at_best_epoch": float(val_losses[best_n - 1]),
    }
    return classifiers, meta


def plot_keras_style_loss_curves(
    per_model_losses: Dict[str, Dict[str, list]],
    paths: Paths,
) -> None:
    """
    Her model için Keras benzeri sade Loss grafiği: x ekseni 0–50 (eğitim aşamaları lineer ölçeklenir),
    y ekseni log-loss (düşük daha iyi). Çıktı: outputs/figures/loss_curve_<model>.png
    """
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    for name, d in per_model_losses.items():
        train_ll = d.get("train_loss") or []
        val_ll = d.get("val_loss") or []
        if len(train_ll) < 2 or len(val_ll) < 2:
            continue
        n = min(len(train_ll), len(val_ll))
        t = np.asarray(train_ll[:n], dtype=float)
        v = np.asarray(val_ll[:n], dtype=float)
        x_plot = np.linspace(0.0, 50.0, n)
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        ax.plot(x_plot, t, color="#1f77b4", linewidth=2.0, linestyle="-", label="train")
        ax.plot(x_plot, v, color="#ff7f0e", linewidth=2.0, linestyle="-", label="validation")
        ax.set_title("Loss", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_xlim(0, 50)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("white")
        plt.tight_layout()
        plt.savefig(paths.figures_dir / f"loss_curve_{name}.png", dpi=130, facecolor="white", bbox_inches="tight")
        plt.close()


def plot_hist_gb_keras_style_loss(fitted_pipe: ImbPipeline, paths: Paths) -> None:
    """HistGradientBoosting için train_score_/validation_score_ → log-loss; Keras tarzı Loss grafiği."""
    mod = fitted_pipe.named_steps.get("model")
    if mod is None:
        return
    vscores = getattr(mod, "validation_score_", None)
    tscores = getattr(mod, "train_score_", None)
    if vscores is None or tscores is None or len(vscores) < 2:
        return
    tscores = np.asarray(tscores, dtype=float)
    vscores = np.asarray(vscores, dtype=float)
    n = int(min(len(vscores), len(tscores)))
    # sklearn: erken durdurma skorları genelde negatif log-loss (yüksek daha iyi)
    train_loss = (-tscores[:n]).tolist()
    val_loss = (-vscores[:n]).tolist()
    plot_keras_style_loss_curves(
        {"hist_gradient_boosting": {"train_loss": train_loss, "val_loss": val_loss}},
        paths,
    )


def plot_all_models_epoch_train_val_log_loss(
    X: pd.DataFrame,
    y: pd.Series,
    classifiers: Dict[str, object],
    paths: Paths,
    seed: int = SEED,
    progressive_epochs: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Mevcut CV fold grafiklerine dokunmadan ek çıktı: x = epoch (iterasyon / aşama), y = log-loss.
    Boosting ve diğer eğrilerde doğrulama log-loss minimumu dikey çizgi ile işaretlenir (erken durdurma önerisi).

    - Boosting (GBC, AdaBoost): sklearn `staged_predict_proba` aşamaları = epoch.
    - Random Forest: artan `n_estimators` ile tam yeniden eğitim (epoch = ağaç sayısı adımı).
    - SVM / KNN / LogReg: stratified artan eğitim alt kümesi oranı (epoch = veri aşaması).
    Tek stratified train/val bölmesi (X_tv üzerinde); önişlemci+SMOTE ile uyumlu.
    """
    X_tr, y_tr, X_va, y_va, X_sm, y_sm_arr, X_va_imp, y_va_arr, k_sm = _holdout_smote_preprocessed_arrays(
        X, y, seed
    )

    names = list(classifiers.keys())
    keras_curves: Dict[str, Dict[str, list]] = {}
    n_models = len(names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.0 * n_rows), squeeze=False)
    axes_flat = axes_grid.flatten()
    summary: Dict[str, Dict[str, Any]] = {}

    for idx, (name, clf) in enumerate(classifiers.items()):
        train_ll: list[float] = []
        val_ll: list[float] = []
        curve_mode = "unknown"
        epochs: list[int] = []

        if isinstance(clf, (GradientBoostingClassifier, AdaBoostClassifier)):
            curve_mode = "boosting_staged_predict_proba"
            m = clone(clf).fit(X_sm, y_sm_arr)
            gen_tr = m.staged_predict_proba(X_sm)
            gen_va = m.staged_predict_proba(X_va_imp)
            for ep, (p_tr, p_va) in enumerate(zip(gen_tr, gen_va), start=1):
                train_ll.append(float(log_loss(y_sm_arr, p_tr, labels=[0, 1])))
                val_ll.append(float(log_loss(y_va_arr, p_va, labels=[0, 1])))
                epochs.append(ep)

        elif isinstance(clf, RandomForestClassifier):
            curve_mode = "random_forest_n_estimators"
            n_max = int(clf.get_params().get("n_estimators", 100))
            n_max = max(n_max, 5)
            schedule = np.unique(np.linspace(4, n_max, num=min(24, max(2, n_max - 3)), dtype=int))
            for ep, n_est in enumerate(schedule, start=1):
                m = clone(clf).set_params(n_estimators=int(n_est))
                m.fit(X_sm, y_sm_arr)
                p_tr = m.predict_proba(X_sm)
                p_va = m.predict_proba(X_va_imp)
                train_ll.append(float(log_loss(y_sm_arr, p_tr, labels=[0, 1])))
                val_ll.append(float(log_loss(y_va_arr, p_va, labels=[0, 1])))
                epochs.append(ep)

        else:
            curve_mode = "progressive_stratified_train_fraction"
            fracs = list(np.linspace(0.18, 0.99, num=max(1, progressive_epochs - 1))) + [1.0]
            for ep, frac in enumerate(fracs, start=1):
                if frac >= 1.0:
                    X_sub, y_sub = X_tr, y_tr
                else:
                    X_sub, _, y_sub, _ = train_test_split(
                        X_tr, y_tr, train_size=float(frac), stratify=y_tr, random_state=seed + 900 + ep
                    )
                pipe = ImbPipeline(
                    [
                        ("preprocessor", _make_preprocessor(X_sub)),
                        ("sampler", SMOTE(random_state=seed, k_neighbors=k_sm)),
                        ("model", clone(clf)),
                    ]
                )
                pipe.fit(X_sub, y_sub)
                p_tr = pipe.predict_proba(X_sub)
                p_va = pipe.predict_proba(X_va)
                train_ll.append(float(log_loss(y_sub, p_tr, labels=[0, 1])))
                val_ll.append(float(log_loss(y_va, p_va, labels=[0, 1])))
                epochs.append(ep)

        best_ep = int(np.argmin(val_ll)) + 1 if val_ll else None
        min_val_ll = float(np.min(val_ll)) if val_ll else None

        summary[name] = {
            "curve_mode": curve_mode,
            "n_epochs": len(epochs),
            "final_train_log_loss": float(train_ll[-1]) if train_ll else None,
            "final_val_log_loss": float(val_ll[-1]) if val_ll else None,
            "best_epoch_val_log_loss": best_ep,
            "min_val_log_loss": min_val_ll,
        }
        keras_curves[name] = {"train_loss": list(train_ll), "val_loss": list(val_ll)}

        plt.figure(figsize=(10.5, 6.4))
        plt.plot(epochs, train_ll, "o-", lw=2.4, ms=7, label="Eğitim log-loss", color="#1f77b4")
        plt.plot(epochs, val_ll, "s-", lw=2.4, ms=7, label="Doğrulama log-loss", color="#ff7f0e")
        if best_ep is not None:
            plt.axvline(
                best_ep,
                color="#6A3D9A",
                ls="--",
                lw=2.2,
                label=f"Önerilen early stopping (epoch {best_ep}, min doğrulama log-loss)",
            )
        plt.xlabel("Epoch (iterasyon / aşama)", fontsize=16, fontweight="semibold")
        plt.ylabel("Log loss", fontsize=16, fontweight="semibold")
        mode_note = {
            "boosting_staged_predict_proba": "Boosting aşaması (staged_predict_proba)",
            "random_forest_n_estimators": "Ağaç sayısı artışı (tam yeniden eğitim)",
            "progressive_stratified_train_fraction": "Stratified artan eğitim verisi oranı",
        }.get(curve_mode, curve_mode)
        plt.title(
            f"{name.replace('_', ' ').title()} — epoch vs loss\n({mode_note})",
            fontsize=15,
            fontweight="bold",
            pad=8,
        )
        plt.grid(True, alpha=0.35)
        plt.legend(fontsize=11, loc="best")
        plt.xticks(epochs, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(paths.figures_dir / f"train_val_epoch_log_loss_{name}.png", dpi=150)
        plt.close()

        ax = axes_flat[idx]
        ax.plot(epochs, train_ll, "o-", lw=2.0, ms=5, label="Eğitim", color="#1f77b4")
        ax.plot(epochs, val_ll, "s-", lw=2.0, ms=5, label="Doğrulama", color="#ff7f0e")
        if best_ep is not None:
            ax.axvline(best_ep, color="#6A3D9A", ls="--", lw=1.8, label=f"Min val (ep.{best_ep})")
        ax.set_title(name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Log loss", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig_grid.suptitle(
        "Tüm modeller — epoch vs log-loss (mor çizgi: min doğrulama log-loss, early stopping önerisi)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "train_val_epoch_log_loss_all_models_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_keras_style_loss_curves(keras_curves, paths)

    return summary


def train_and_compare(df: pd.DataFrame, target_col: str) -> Dict:
    # Eğitim öncesi çıktı klasörlerini hazırla.
    paths = get_paths()
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.model_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    # ID sütunları modele katkı sağlamadığı ve sızıntı riski oluşturduğu için çıkarılır.
    drop_cols = [c for c in df.columns if c.lower() in {"id", "index", "record_id", "patient_id"}]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    pre = _make_preprocessor(X)
    # 10 katlı stratified CV (daha stabil ve güvenilir değerlendirme için)
    cv = StratifiedKFold(
    n_splits=10, 
    shuffle=True,
     random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Epoch grafiği ve AdaBoost erken-durdurma: hold-out %80 ile aynı X_tv alt kümesi (tutarlılık için).
    X_tv_tune, _, y_tv_tune, _ = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    classifiers = _build_classifiers()
    classifiers, adaboost_early_stop_meta = _tune_adaboost_n_estimators_val_early_stop(
        classifiers, X_tv_tune, y_tv_tune, SEED
    )
    samplers = _build_samplers()
    over_methods = {
        "SMOTE",
        "KMeansSMOTE",
        "RandomOverSampler",
        "ADASYN",
        "BorderlineSMOTE",
        "SVMSMOTE",
    }

    # Tüm model+örnekleme kombinasyonlarının sonuçları burada birikecek.
    rows = []
    for clf_name, clf in classifiers.items():
        # Önce örneklemesiz temel performansı ölç.
        base_pipe = ImbPipeline([("preprocessor", pre), ("model", clone(clf))])
        base_cv = cross_validate(base_pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "classifier": clf_name,
                "sampling_category": "none",
                "sampling_method": "-",
                "accuracy": float(np.mean(base_cv["test_accuracy"])),
                "precision": float(np.mean(base_cv["test_precision"])),
                "recall": float(np.mean(base_cv["test_recall"])),
                "f1": float(np.mean(base_cv["test_f1"])),
                "roc_auc": float(np.mean(base_cv["test_roc_auc"])),
            }
        )

        # Sonra aynı modeli her örnekleme yöntemi ile tekrar ölç.
        for sampler_name, sampler in samplers.items():
            pipe = ImbPipeline(
                [("preprocessor", pre), ("sampler", clone(sampler)), ("model", clone(clf))]
            )
            cv_scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            rows.append(
                {
                    "classifier": clf_name,
                    "sampling_category": "oversampling" if sampler_name in over_methods else "undersampling",
                    "sampling_method": sampler_name,
                    "accuracy": float(np.mean(cv_scores["test_accuracy"])),
                    "precision": float(np.mean(cv_scores["test_precision"])),
                    "recall": float(np.mean(cv_scores["test_recall"])),
                    "f1": float(np.mean(cv_scores["test_f1"])),
                    "roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
                }
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(paths.metrics_dir / "all_model_sampling_results.csv", index=False)

    per_model_test = evaluate_per_model_test_metrics(X, y, classifiers)
    (paths.metrics_dir / "per_model_test_metrics.json").write_text(
        json.dumps(per_model_test, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Her sınıflandırıcı için: örneklemesiz + en iyi over/under kombinasyonunu özetle.
    summary_rows = []
    for clf_name in classifiers:
        clf_df = results_df[results_df["classifier"] == clf_name]
        none_row = clf_df[clf_df["sampling_category"] == "none"].iloc[0]
        over_row = clf_df[clf_df["sampling_category"] == "oversampling"].sort_values(
            by="f1", ascending=False
        ).iloc[0]
        under_row = clf_df[clf_df["sampling_category"] == "undersampling"].sort_values(
            by="f1", ascending=False
        ).iloc[0]
        summary_rows.append(
            {
                "classifier": clf_name,
                "none_accuracy": float(none_row["accuracy"]),
                "none_f1": float(none_row["f1"]),
                "best_oversampling_method": str(over_row["sampling_method"]),
                "best_oversampling_accuracy": float(over_row["accuracy"]),
                "best_undersampling_method": str(under_row["sampling_method"]),
                "best_undersampling_accuracy": float(under_row["accuracy"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.metrics_dir / "classifier_summary_best_sampling.csv", index=False)

    # Özet tabloyu barplot için uzun formata çevir.
    plot_df = summary_df.melt(
        id_vars=["classifier"],
        value_vars=["none_accuracy", "best_oversampling_accuracy", "best_undersampling_accuracy"],
        var_name="setting",
        value_name="accuracy",
    )
    plot_df["setting"] = plot_df["setting"].map(
        {
            "none_accuracy": "Örneklemesiz",
            "best_oversampling_accuracy": "En iyi fazla örnekleme",
            "best_undersampling_accuracy": "En iyi az örnekleme",
        }
    )
    plt.figure(figsize=(12.5, 5.5))
    ax = sns.barplot(data=plot_df, x="classifier", y="accuracy", hue="setting")
    plt.ylim(0, 1.0)
    plt.title("Sınıflandırıcı Doğruluğu (Örneklemeli/Örneklemesiz)", fontsize=16)
    plt.xlabel("Sınıflandırıcı", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    _compact_legend_bar_charts(ax)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(paths.figures_dir / "accuracy_with_without_sampling.png", dpi=140)
    plt.close()

    # Recall öncelikli seçim (klinik: FN maliyeti); eşit recall'da F1 ile kırılım.
    best_recall_row = results_df.sort_values(by=["recall", "f1"], ascending=[False, False]).iloc[0]
    best_f1_row = results_df.sort_values(by="f1", ascending=False).iloc[0]

    best_recall_combo = _combo_dict_from_row(best_recall_row)
    best_f1_combo = _combo_dict_from_row(best_f1_row)

    recall_best_cv_scores, class_rep_recall_cv = _oof_eval_and_confusion_plot(
        _pipeline_for_result_row(best_recall_row, pre, samplers, classifiers),
        X,
        y,
        cv,
        "Karmaşıklık Matrisi (Recall En İyi — 10-fold CV OOF)",
        paths.figures_dir / "confusion_matrix_recall_cv_reference.png",
    )
    f1_best_cv_scores, class_rep_f1_cv = _oof_eval_and_confusion_plot(
        _pipeline_for_result_row(best_f1_row, pre, samplers, classifiers),
        X,
        y,
        cv,
        "Karmaşıklık Matrisi (F1 En İyi — 10-fold CV OOF)",
        paths.figures_dir / "confusion_matrix_f1_cv_reference.png",
    )

    vc = y.value_counts().sort_index()
    class_imbalance = {
        "counts": {int(k): int(v) for k, v in vc.items()},
        "ratio_positive": float(vc.get(1, 0) / max(len(y), 1)),
        "ratio_negative": float(vc.get(0, 0) / max(len(y), 1)),
        "notes": "Pozitif (diyabet) sınıfı azınlıkta; SMOTE ve sınıflandırma eşiği recall hedefi için kullanılır.",
    }

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2, random_state=SEED, stratify=y_tv)

    all_models_log_loss_cv = compute_all_models_cv_train_val_log_loss(X_tv, y_tv, classifiers)
    all_models_epoch_log_loss = plot_all_models_epoch_train_val_log_loss(X_tv, y_tv, classifiers, paths)

    gb_base = ImbPipeline(
        [
            ("preprocessor", clone(pre)),
            ("sampler", SMOTE(random_state=SEED, k_neighbors=3)),
            ("model", _hist_gb_deployment_estimator()),
        ]
    )
    cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring_full = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    # Referans: güçlü ama düzenlenmemiş HistGB (yüksek kapasite → CV’de train–val ayrımı genelde daha büyük)
    gb_baseline_overfit = ImbPipeline(
        [
            ("preprocessor", clone(pre)),
            ("sampler", SMOTE(random_state=SEED, k_neighbors=3)),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=SEED,
                    max_iter=320,
                    learning_rate=0.12,
                    max_depth=8,
                    min_samples_leaf=3,
                    l2_regularization=0.0,
                    early_stopping=False,
                    max_bins=255,
                ),
            ),
        ]
    )
    base_cv = cross_validate(
        clone(gb_baseline_overfit), X_train, y_train, cv=cv_tune, scoring=scoring_full, n_jobs=-1
    )
    baseline_cv_means = {
        m: float(np.mean(base_cv[f"test_{m}"])) for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    param_distributions = {
        "model__learning_rate": sp_uniform(0.02, 0.09),
        "model__max_depth": sp_randint(2, 6),
        "model__max_iter": sp_randint(100, 320),
        "model__l2_regularization": sp_uniform(0.2, 12.0),
        "model__min_samples_leaf": sp_randint(10, 48),
        # Erken durdurma: doğrulama platosu sonrası gereksiz iterasyonları kes
        "model__n_iter_no_change": sp_randint(8, 20),
    }
    search = RandomizedSearchCV(
        clone(gb_base),
        param_distributions=param_distributions,
        n_iter=36,
        cv=cv_tune,
        scoring="roc_auc",
        refit=True,
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    tuned_est = search.best_estimator_
    tuned_cv = cross_validate(clone(tuned_est), X_train, y_train, cv=cv_tune, scoring=scoring_full, n_jobs=-1)
    tuned_cv_means = {
        m: float(np.mean(tuned_cv[f"test_{m}"])) for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    hist_iter_stats = plot_hist_gb_learning_curves(tuned_est, paths)
    plot_hist_gb_keras_style_loss(tuned_est, paths)

    val_prob = tuned_est.predict_proba(X_val)[:, 1]
    optimal_threshold_val = optimize_threshold_recall_band(y_val, val_prob, low=0.90, high=0.95)

    y_val_pred = (val_prob >= optimal_threshold_val).astype(int)
    val_holdout_scores = _scores(y_val, y_val_pred, val_prob)
    class_rep_val = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
    conf_val = confusion_matrix(y_val, y_val_pred)
    row_sum_v = np.maximum(conf_val.sum(axis=1, keepdims=True), 1e-9)
    conf_pct_v = conf_val.astype(float) / row_sum_v
    conf_annot_v = np.array(
        [
            [
                f"{conf_val[i, j]}\n({format_display_float(conf_pct_v[i, j] * 100, 1)}%)"
                for j in range(conf_val.shape[1])
            ]
            for i in range(conf_val.shape[0])
        ]
    )
    plt.figure(figsize=(7.2, 6))
    sns.heatmap(
        conf_val,
        annot=conf_annot_v,
        fmt="",
        cmap="Greens",
        xticklabels=["Sağlıklı (0)", "Diyabet (1)"],
        yticklabels=["Sağlıklı (0)", "Diyabet (1)"],
        annot_kws={"size": 13},
    )
    plt.title("Karmaşıklık Matrisi — Doğrulama (eşik sonrası)", fontsize=17, fontweight="bold")
    plt.xlabel("Tahmin", fontsize=15)
    plt.ylabel("Gerçek", fontsize=15)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "confusion_matrix_validation_holdout.png", dpi=150)
    plt.close()

    final_inner = clone(tuned_est)
    final_inner.fit(X_tv, y_tv)
    final_model = ThresholdedBinaryClassifier(final_inner, optimal_threshold_val)

    y_test_pred = final_model.predict(X_test)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    deployment_test_scores = _scores(y_test, y_test_pred, y_test_prob)
    class_rep_test = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

    conf = confusion_matrix(y_test, y_test_pred)
    row_sum = np.maximum(conf.sum(axis=1, keepdims=True), 1e-9)
    conf_pct_row = conf.astype(float) / row_sum
    conf_annot = np.array(
        [
            [
                f"{conf[i, j]}\n({format_display_float(conf_pct_row[i, j] * 100, 1)}%)"
                for j in range(conf.shape[1])
            ]
            for i in range(conf.shape[0])
        ]
    )
    plt.figure(figsize=(7, 5.8))
    sns.heatmap(
        conf,
        annot=conf_annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Sağlıklı (0)", "Diyabet (1)"],
        yticklabels=["Sağlıklı (0)", "Diyabet (1)"],
        annot_kws={"size": 12},
    )
    plt.title("Karmaşıklık Matrisi — Test (HistGB + SMOTE + eşik)", fontsize=16, fontweight="bold")
    plt.xlabel("Tahmin Edilen Sınıf", fontsize=14)
    plt.ylabel("Gerçek Sınıf", fontsize=14)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "confusion_matrix_best_model.png", dpi=140)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(6.5, 5.5))
    auc_text = f"{auc(fpr, tpr):.3f}".replace(".", ",")
    plt.plot(fpr, tpr, label=f"AUC = {auc_text}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Yanlış Pozitif Oranı", fontsize=13)
    plt.ylabel("Doğru Pozitif Oranı", fontsize=13)
    plt.title("ROC Eğrisi — Test Verisi", fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "roc_curve_best_model.png", dpi=140)
    plt.close()

    cv_fold_gap_stats = plot_training_curves(
        X_tv,
        y_tv,
        clone(tuned_est),
        StratifiedKFold(10, shuffle=True, random_state=SEED),
        paths,
    )

    gap_cv = float(cv_fold_gap_stats.get("mean_train_minus_val_accuracy", 0.0)) if cv_fold_gap_stats else None
    gap_it = float(hist_iter_stats["mean_train_minus_val_score_curve"]) if hist_iter_stats else None
    k_gap = int(cv_fold_gap_stats.get("n_splits", 10)) if cv_fold_gap_stats else 10
    overfitting_note = (
        "Derin öğrenme (dropout, batch norm) yerine sklearn HistGradientBoosting kullanıldı. "
        "Erken durdurma, L2 (l2_regularization), düşük max_depth, yüksek min_samples_leaf ve satır alt örnekleme "
        "ile genelleme iyileştirildi. SMOTE sınıf dengesizliği için sentetik örnek üretir (tabular augmentation). "
        "Karşılaştırma modelinde (early_stopping kapalı) daha yüksek kapasite kullanıldı. "
        "Genel modellerde varyans azaltımı: RF max_depth=5, KNN n_neighbors=15 (distance ağırlıklı), SVM C düşürüldü, GBC subsample+sqrt max_features; sayısal sütunlarda IQR (1.5×) kıskaç + StandardScaler. "
        f"{k_gap}-fold stratified CV’de ortalama train−val accuracy farkı: {gap_cv:.4f} (küçük olması hedeflenir). "
        f"Boosting iterasyon skorlarında ortalama train−val farkı: {gap_it:.4f}."
        if gap_cv is not None and gap_it is not None
        else "Overfitting azaltma: HistGradientBoosting + early stopping + L2 + hiperparametre araması (roc_auc)."
    )
    cv_oof = StratifiedKFold(10, shuffle=True, random_state=SEED)
    oof_prob = cross_val_predict(clone(tuned_est), X, y, cv=cv_oof, method="predict_proba", n_jobs=-1)[:, 1]
    optimal_threshold_full = optimize_threshold_recall_band(y, oof_prob, low=0.90, high=0.95)

    final_deploy = clone(tuned_est)
    final_deploy.fit(X, y)
    joblib.dump(ThresholdedBinaryClassifier(final_deploy, optimal_threshold_full), paths.model_path)

    # Çalışma özeti ve ortam bilgileri tek JSON dosyasında saklanır.
    run_artifacts = {
        "seed": SEED,
        "target_column": target_col,
        "dropped_identifier_columns": drop_cols,
        "cv_folds": 10,
        "total_classification_runs": int(len(results_df)),
        "model_selection_notes": (
            "best_overall: 10-fold CV ızgarasında en yüksek recall (eşitlikte F1). "
            "best_f1_overall: F1 referansı. best_cv_scores: dağıtım modeli (HistGB+SMOTE+eşik) test hold-out."
        ),
        "best_overall": {
            **best_recall_combo,
            "selection_criterion": "recall_then_f1_tiebreak",
            "cv_scores": recall_best_cv_scores,
        },
        "best_recall_overall": {
            **best_recall_combo,
            "selection_criterion": "recall_then_f1_tiebreak",
            "cv_scores": recall_best_cv_scores,
        },
        "best_f1_overall": {
            **best_f1_combo,
            "selection_criterion": "f1",
            "cv_scores": f1_best_cv_scores,
        },
        "reference_recall_best_full_data_cv": recall_best_cv_scores,
        "classification_report_recall_cv_reference": class_rep_recall_cv,
        "reference_f1_best_full_data_cv": f1_best_cv_scores,
        "classification_report_f1_cv_reference": class_rep_f1_cv,
        "best_cv_scores": deployment_test_scores,
        "score_context": "best_cv_scores: dağıtım modeli (HistGradientBoosting + SMOTE + recall eşiği) — ayrılmış test (hold-out).",
        "deployment_model": {
            "type": "hist_gradient_boosting_smote_threshold",
            "classifier": "HistGradientBoostingClassifier",
            "selection_criterion": "roc_auc_tuning_recall_threshold",
            "recall_target_band": [0.90, 0.95],
            "optimal_threshold_validation_split": float(optimal_threshold_val),
            "optimal_threshold_full_data_oof": float(optimal_threshold_full),
        },
        "class_imbalance": class_imbalance,
        "validation_holdout_scores": val_holdout_scores,
        "classification_report_validation_holdout": class_rep_val,
        "gradient_boosting_hyperparameter_search": {
            "classifier": "HistGradientBoostingClassifier",
            "method": "RandomizedSearchCV",
            "n_iter": 36,
            "inner_cv_folds": 5,
            "scoring": "roc_auc",
            "best_params": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in search.best_params_.items()},
            "baseline_cv_means_train_portion": baseline_cv_means,
            "tuned_cv_means_train_portion": tuned_cv_means,
            "hist_gb_iteration_curve_summary": hist_iter_stats or {},
            "cv_fold_train_val_gap": cv_fold_gap_stats or {},
        },
        "overfitting_mitigation_summary": overfitting_note,
        "all_models_cv_log_loss_summary": {
            k: {
                "mean_train_log_loss": float(v["mean_train_log_loss"]),
                "mean_val_log_loss": float(v["mean_val_log_loss"]),
            }
            for k, v in all_models_log_loss_cv.items()
        },
        "all_models_epoch_log_loss_summary": all_models_epoch_log_loss,
        "all_models_epoch_log_loss_figure_grid": "train_val_epoch_log_loss_all_models_grid.png",
        "keras_style_loss_curve_files": [f"loss_curve_{n}.png" for n in classifiers.keys()]
        + ["loss_curve_hist_gradient_boosting.png"],
        "adaboost_validation_early_stop": adaboost_early_stop_meta,
        "classification_report_test_holdout": class_rep_test,
        "per_model_test_metrics_file": "per_model_test_metrics.json",
        "python_version": sys.version,
        "platform": os.name,
        "cpu_count": os.cpu_count(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
    }
    with open(paths.metrics_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_artifacts, f, ensure_ascii=False, indent=2)
    return run_artifacts


def plot_training_curves(
    X: pd.DataFrame,
    y: pd.Series,
    best_pipeline: ImbPipeline,
    cv: StratifiedKFold,
    paths: Paths,
) -> Dict[str, float]:
    """
    Stratified K-fold CV: her fold’da pipeline eğitilip train / validation accuracy ve (1−acc) loss çizilir.
    Overfitting için train ile val arasındaki ortalama farkı döndürür.
    """
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []
    fold_numbers: list[int] = []
    k = int(cv.n_splits)

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        fold_pipeline = clone(best_pipeline)
        fold_pipeline.fit(X_train_fold, y_train_fold)

        train_pred = fold_pipeline.predict(X_train_fold)
        val_pred = fold_pipeline.predict(X_val_fold)

        train_acc = float(accuracy_score(y_train_fold, train_pred))
        val_acc = float(accuracy_score(y_val_fold, val_pred))

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        fold_numbers.append(fold_num)

    plt.figure(figsize=(11.5, 6.8))
    plt.plot(
        fold_numbers,
        train_accuracies,
        marker="o",
        linewidth=2.6,
        markersize=10,
        label="Eğitim Accuracy",
        color="#5DA5DA",
    )
    plt.plot(
        fold_numbers,
        val_accuracies,
        marker="s",
        linewidth=2.6,
        markersize=10,
        label="Doğrulama Accuracy",
        color="#F28E2B",
    )

    plt.xlabel("CV Fold", fontsize=17, fontweight="semibold")
    plt.ylabel("Accuracy", fontsize=17, fontweight="semibold")
    plt.title(f"{k}-Fold CV — Eğitim vs doğrulama accuracy", fontsize=19, fontweight="bold", pad=12)
    plt.xticks(fold_numbers, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([max(0, min(min(train_accuracies), min(val_accuracies)) - 0.05), 1.05])
    plt.grid(True, alpha=0.35)
    plt.legend(loc="lower right", fontsize=15)

    for i, (train_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies)):
        plt.text(
            fold_numbers[i],
            train_acc + 0.013,
            format_display_float(train_acc, 3),
            ha="center",
            fontsize=13,
        )
        plt.text(
            fold_numbers[i],
            val_acc - 0.038,
            format_display_float(val_acc, 3),
            ha="center",
            fontsize=13,
        )

    plt.tight_layout()
    plt.savefig(paths.figures_dir / "training_validation_curves.png", dpi=150)
    plt.close()

    train_losses = [1.0 - acc for acc in train_accuracies]
    val_losses = [1.0 - acc for acc in val_accuracies]

    plt.figure(figsize=(11.5, 6.8))
    plt.plot(
        fold_numbers,
        train_losses,
        marker="o",
        linewidth=2.6,
        markersize=10,
        label="Eğitim Loss (1 − Accuracy)",
        color="#5DA5DA",
    )
    plt.plot(
        fold_numbers,
        val_losses,
        marker="s",
        linewidth=2.6,
        markersize=10,
        label="Doğrulama Loss (1 − Accuracy)",
        color="#F28E2B",
    )

    plt.xlabel("CV Fold", fontsize=17, fontweight="semibold")
    plt.ylabel("Loss", fontsize=17, fontweight="semibold")
    plt.title(f"{k}-Fold CV — Eğitim vs doğrulama loss", fontsize=19, fontweight="bold", pad=12)
    plt.xticks(fold_numbers, fontsize=14)
    plt.yticks(fontsize=14)
    lo_floor = max(0.0, min(min(train_losses), min(val_losses)) - 0.05)
    hi_ceil = max(max(train_losses), max(val_losses)) + 0.1
    plt.ylim([snap_display_scalar(lo_floor), hi_ceil])
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper right", fontsize=15)

    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        plt.text(
            fold_numbers[i],
            train_loss + 0.012,
            format_display_float(train_loss, 3),
            ha="center",
            fontsize=13,
        )
        plt.text(
            fold_numbers[i],
            val_loss - 0.03,
            format_display_float(val_loss, 3),
            ha="center",
            fontsize=13,
        )

    plt.tight_layout()
    plt.savefig(paths.figures_dir / "training_validation_loss.png", dpi=150)
    plt.close()

    gap_mean = float(np.mean(np.array(train_accuracies) - np.array(val_accuracies)))
    return {
        "n_splits": k,
        "mean_train_accuracy_cv": float(np.mean(train_accuracies)),
        "mean_val_accuracy_cv": float(np.mean(val_accuracies)),
        "mean_train_minus_val_accuracy": gap_mean,
        "per_fold_train_accuracy": train_accuracies,
        "per_fold_val_accuracy": val_accuracies,
    }


def generate_report() -> Path:
    # Üretilmiş metrik/özet dosyaları olmadan rapor üretme.
    paths = get_paths()
    metrics_path = paths.metrics_dir / "metrics_summary.json"
    summary_path = paths.metrics_dir / "classifier_summary_best_sampling.csv"
    full_path = paths.metrics_dir / "all_model_sampling_results.csv"
    if not metrics_path.exists() or not summary_path.exists() or not full_path.exists():
        raise FileNotFoundError("Rapor için önce `python main.py train` çalıştırın.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_recall = metrics.get("best_recall_overall") or metrics["best_overall"]
    best_f1 = metrics.get("best_f1_overall") or metrics["best_overall"]
    recall_cv = best_recall.get("cv_scores") or metrics.get("reference_recall_best_full_data_cv", {})
    f1_cv = best_f1.get("cv_scores") or metrics.get("reference_f1_best_full_data_cv", {})
    deploy = metrics["best_cv_scores"]
    lines = [
        "# Diyabet Tanı Raporu (Makale Benzeri)",
        "",
        "## Kullanım Özeti",
        "- 6 sınıflandırıcı kullanıldı: SVM, Lojistik Regresyon, KNN, Random Forest, AdaBoost, Gradient Boosting.",
        "- Her model için 14 yeniden örnekleme + örneklemesiz durum değerlendirildi.",
        "- Toplam çalıştırma sayısı: 90 (makaledekiyle uyumlu).",
        "- Izgara seçimi: recall (eşitlikte F1); F1 referansı ayrı raporlanır.",
        "- Dağıtım modeli: HistGradientBoosting + SMOTE + recall eşiği (%90–%95 bandı), test hold-out.",
        "",
        "## Recall odaklı ızgara şampiyonu (10-fold CV OOF)",
        f"- Model: `{best_recall['classifier']}`",
        f"- Örnekleme: `{best_recall['sampling_category']}` / `{best_recall['sampling_method']}`",
        f"- Recall: `{str(round(recall_cv.get('recall', 0), 4)).replace('.', ',')}`",
        f"- F1: `{str(round(recall_cv.get('f1', 0), 4)).replace('.', ',')}`",
        f"- ROC-AUC: `{str(round(recall_cv.get('roc_auc', 0), 4)).replace('.', ',')}`",
        "",
        "## F1 referansı (10-fold CV OOF)",
        f"- Model: `{best_f1['classifier']}`",
        f"- Örnekleme: `{best_f1['sampling_category']}` / `{best_f1['sampling_method']}`",
        f"- F1: `{str(round(f1_cv.get('f1', 0), 4)).replace('.', ',')}`",
        f"- Recall: `{str(round(f1_cv.get('recall', 0), 4)).replace('.', ',')}`",
        "",
        "## Dağıtım modeli (test hold-out)",
        f"- Recall: `{str(round(deploy['recall'], 4)).replace('.', ',')}`",
        f"- Precision: `{str(round(deploy['precision'], 4)).replace('.', ',')}`",
        f"- F1: `{str(round(deploy['f1'], 4)).replace('.', ',')}`",
        f"- ROC-AUC: `{str(round(deploy['roc_auc'], 4)).replace('.', ',')}`",
        "",
        "## Çıktı Dosyaları",
        "- `outputs/all_model_sampling_results.csv`",
        "- `outputs/classifier_summary_best_sampling.csv`",
        "- `outputs/figures/class_distribution_pie.png`",
        "- `outputs/figures/accuracy_with_without_sampling.png`",
        "- `outputs/figures/training_validation_curves.png`",
        "- `outputs/figures/training_validation_loss.png`",
        "- `outputs/figures/train_val_epoch_log_loss_all_models_grid.png`",
        "- `outputs/figures/train_val_epoch_log_loss_<model>.png` (epoch ekseni; CV fold grafiklerinden bağımsız)",
        "- `outputs/figures/loss_curve_<model>.png` (Keras tarzı train/validation Loss, 0–50 epoch)",
        "- `outputs/figures/loss_curve_hist_gradient_boosting.png` (üretim HistGB)",
        "- `outputs/figures/confusion_matrix_validation_holdout.png`",
        "- `outputs/figures/confusion_matrix_recall_cv_reference.png`",
        "- `outputs/figures/confusion_matrix_f1_cv_reference.png`",
        "- `outputs/figures/confusion_matrix_best_model.png`",
        "- `outputs/figures/roc_curve_best_model.png`",
    ]
    paths.report_path.write_text("\n".join(lines), encoding="utf-8")

    # EDA galeri raporunda kullanılacak görsel sırası ve başlıkları.
    gallery_items = [
        ("class_distribution.png", "Şekil 1. Hedef sınıf dağılımı (sayım grafiği)"),
        ("class_distribution_pie.png", "Şekil 2. Veri setindeki diyabet ve sağlıklı kayıt dağılımları"),
        ("age_group_outcome_distribution.png", "Şekil 3. Yaş gruplarına göre hastalık dağılımları"),
        ("correlation_heatmap.png", "Şekil 4. Korelasyon ısı haritası"),
        ("bmi_category_outcome_stacked.png", "Şekil 5. BMI kategorilerine göre hedef sınıf dağılımı (yığılmış sütun)"),
        ("glucose_outcome_boxplot.png", "Şekil 6. Hedef sınıfa göre glikoz dağılımı (kutu grafiği)"),
        ("bmi_outcome_violinplot.png", "Şekil 7. Hedef sınıfa göre BMI dağılımı (keman grafiği)"),
        ("feature_importance_rf.png", "Şekil 8. Random Forest özellik önemi"),
        ("train_val_epoch_log_loss_all_models_grid.png", "Şekil 8c. Epoch vs log-loss; mor çizgi = min doğrulama (early stopping önerisi)"),
        ("training_boosting_iteration_scores.png", "Şekil 8b. HistGB iterasyonlarında eğitim vs doğrulama skoru (early stopping)"),
        ("training_validation_curves.png", "Şekil 9. 10-fold stratified CV eğitim vs doğrulama accuracy (HistGB)"),
        ("training_validation_loss.png", "Şekil 10. 10-fold stratified CV eğitim vs doğrulama loss (1−accuracy, HistGB)"),
        ("confusion_matrix_validation_holdout.png", "Şekil 10b. Doğrulama kümesi karmaşıklık matrisi"),
        ("confusion_matrix_recall_cv_reference.png", "Şekil 11a. Karmaşıklık matrisi (recall en iyi, CV OOF)"),
        ("confusion_matrix_f1_cv_reference.png", "Şekil 11b. Karmaşıklık matrisi (F1 referansı, CV OOF)"),
        ("confusion_matrix_best_model.png", "Şekil 11. Karmaşıklık matrisi (test, HistGB+SMOTE+eşik)"),
        ("roc_curve_best_model.png", "Şekil 12. ROC eğrisi ve AUC"),
    ]
    for clf_name in ["svm", "logreg", "knn", "random_forest", "adaboost", "gradient_boosting"]:
        lf = f"loss_curve_{clf_name}.png"
        if (paths.figures_dir / lf).exists():
            gallery_items.append(
                (lf, f"Ek (Loss). {clf_name.replace('_', ' ').title()} — train/validation (Keras tarzı, log-loss)")
            )
    if (paths.figures_dir / "loss_curve_hist_gradient_boosting.png").exists():
        gallery_items.append(
            (
                "loss_curve_hist_gradient_boosting.png",
                "Ek (Loss). HistGradientBoosting (üretim) — train/validation (Keras tarzı)",
            )
        )
    for clf_name in ["svm", "logreg", "knn", "random_forest", "adaboost", "gradient_boosting"]:
        fn_e = f"train_val_epoch_log_loss_{clf_name}.png"
        if (paths.figures_dir / fn_e).exists():
            gallery_items.append(
                (fn_e, f"Ek. {clf_name.replace('_', ' ').title()} — epoch vs log-loss (min doğrulama = early stopping önerisi)")
            )

    important_group_plots = sorted(paths.figures_dir.glob("*_grouped_outcome_distribution.png"))
    for path_obj in important_group_plots:
        if path_obj.name == "age_group_outcome_distribution.png":
            continue
        feature_name = path_obj.stem.replace("_grouped_outcome_distribution", "").replace("_", " ").title()
        gallery_items.append((path_obj.name, f"Ek Görsel. {feature_name} gruplarına göre hastalık dağılımı"))

    # Markdown tabanlı görsel galeri raporu.
    gallery_lines = [
        "# EDA Görsel Galerisi",
        "",
        "Bu dosya, projede üretilen EDA ve model değerlendirme görsellerini şekil bazında listeler.",
        "",
    ]
    for fname, caption in gallery_items:
        gallery_lines.extend(
            [
                f"## {caption}",
                "",
                f"![{caption}](figures/{fname})",
                "",
            ]
        )
    paths.eda_gallery_report_path.write_text("\n".join(gallery_lines), encoding="utf-8")
    return paths.report_path


def predict_single(sample: Dict[str, float | int]) -> Dict:
    # Tek bir gözlem için modelden sınıf tahmini + olasılık üret.
    paths = get_paths()
    if not paths.model_path.exists():
        raise FileNotFoundError("Model bulunamadı. Önce train komutunu çalıştırın.")
    model = joblib.load(paths.model_path)
    X = pd.DataFrame([sample])
    if isinstance(model, ThresholdedBinaryClassifier):
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0, 1])
        return {
            "prediction": pred,
            "diabetes_probability": proba,
            "decision_threshold": float(model.threshold),
        }
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None
    return {"prediction": pred, "diabetes_probability": proba}
