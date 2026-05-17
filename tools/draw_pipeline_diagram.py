"""Proje pipeline mimarisi diyagramı üretir."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "pipeline_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(20, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("#FAFBFC")

COLORS = {
    "data": "#D6EAF8",
    "data_edge": "#2874A6",
    "cli": "#E8DAEF",
    "cli_edge": "#7D3C98",
    "process": "#FDEBD0",
    "process_edge": "#CA6F1E",
    "model": "#D5F5E3",
    "model_edge": "#1E8449",
    "output": "#D1F2EB",
    "output_edge": "#117A65",
    "deploy": "#FADBD8",
    "deploy_edge": "#C0392B",
}


def box(x, y, w, h, text, fc, ec, fs=8.5, bold=False):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.8,
    )
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color="#1C2833",
        wrap=True,
    )


def arrow(x1, y1, x2, y2, color="#566573"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


# Title
ax.text(
    10,
    10.35,
    "Diyabet Risk Analizi — ML Pipeline Mimarisi",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color="#1B4F72",
)

# --- DATA ---
ax.text(1.2, 9.5, "1. VERİ", fontsize=11, fontweight="bold", color="#2874A6")
box(0.3, 7.8, 2.4, 0.9, "Kaggle API\n(kagglehub)", COLORS["data"], COLORS["data_edge"], 8)
box(0.3, 6.5, 2.4, 0.9, "Manuel CSV", COLORS["data"], COLORS["data_edge"], 8)
box(0.3, 4.8, 2.4, 1.2, "data/raw/\ndiabetes.csv", COLORS["data"], COLORS["data_edge"], 9, True)
box(0.3, 3.2, 2.4, 0.9, "main.py\ndownload", COLORS["cli"], COLORS["cli_edge"], 8)

arrow(1.5, 7.8, 1.5, 6.0)
arrow(1.5, 6.5, 1.5, 6.0)
arrow(1.5, 4.8, 1.5, 4.1)

# --- CLI ---
ax.text(3.8, 9.5, "2. CLI", fontsize=11, fontweight="bold", color="#7D3C98")
box(3.2, 7.0, 2.6, 1.8, "main.py\n─────────\ndownload | train\nreport | predict", COLORS["cli"], COLORS["cli_edge"], 8.5, True)
arrow(2.7, 5.4, 3.2, 7.5, "#7D3C98")

# --- TRAIN PIPELINE ---
ax.text(7.5, 9.5, "3. EĞİTİM AKIŞI (train)", fontsize=11, fontweight="bold", color="#CA6F1E")

steps = [
    ("Veri Yükleme\nload_data + hedef sütun", 5.5, 8.0),
    ("EDA (run_eda)\nEksik · Sınıf dengesi · IQR\nKorelasyon · Grafikler", 5.5, 6.7),
    ("Ön İşleme\nSayısal: Median → IQR → Scaler\nKategorik: Mode → OneHot", 5.5, 5.2),
    ("Model Karşılaştırma\n6 sınıflandırıcı × 15 örnekleme\n10-fold Stratified CV (90 run)", 5.5, 3.5),
    ("Seçim\nRecall ↑ (F1 tiebreak)\n+ F1 referans", 5.5, 2.0),
]

for i, (txt, x, y) in enumerate(steps):
    box(x, y, 4.2, 1.1 if i < 4 else 0.95, txt, COLORS["process"], COLORS["process_edge"], 8)
    if i < len(steps) - 1:
        arrow(x + 2.1, y, x + 2.1, steps[i + 1][2] + 1.1)

arrow(4.5, 7.9, 5.5, 8.55)

# HistGB branch
box(10.2, 5.8, 4.5, 1.5, "HistGradientBoosting\nRandomizedSearchCV (36 iter)\nSMOTE + Early Stopping + L2", COLORS["model"], COLORS["model_edge"], 8.5)
box(10.2, 3.8, 4.5, 1.3, "Eşik Optimizasyonu\nRecall %90–95 bandı\nThresholdedBinaryClassifier", COLORS["model"], COLORS["model_edge"], 8.5)
box(10.2, 2.0, 4.5, 1.1, "Model Kaydet\nmodels/best_model.joblib", COLORS["model"], COLORS["model_edge"], 9, True)

arrow(9.7, 4.0, 10.2, 6.5)
arrow(12.45, 5.8, 12.45, 5.1)
arrow(12.45, 3.8, 12.45, 3.1)

# Classifiers detail box
box(10.2, 7.5, 4.5, 1.6, "Sınıflandırıcılar:\nSVM · LogReg · KNN · RF\nAdaBoost · GradientBoosting\n\nÖrnekleme: SMOTE, ADASYN, ROS,\nTomekLinks, NearMiss, ... (14)", COLORS["process"], COLORS["process_edge"], 7.5)
arrow(9.7, 6.7, 10.2, 8.0)

# --- OUTPUTS ---
ax.text(15.5, 9.5, "4. ÇIKTILAR", fontsize=11, fontweight="bold", color="#117A65")
outputs = [
    "outputs/figures/*.png",
    "eda_summary.json",
    "metrics_summary.json",
    "all_model_sampling_results.csv",
    "classifier_summary_best_sampling.csv",
    "report.md · eda_gallery_report.md",
]
for i, o in enumerate(outputs):
    box(15.0, 8.0 - i * 1.05, 4.5, 0.85, o, COLORS["output"], COLORS["output_edge"], 8)

arrow(14.7, 6.55, 15.0, 7.5)
arrow(14.7, 2.55, 15.0, 5.0)

# --- DEPLOYMENT ---
ax.text(15.5, 1.85, "5. KULLANIM", fontsize=11, fontweight="bold", color="#C0392B")
box(14.8, 0.35, 2.1, 1.2, "Streamlit\napp.py\n────────\nTahmin · Analiz", COLORS["deploy"], COLORS["deploy_edge"], 7.5)
box(17.2, 0.35, 2.3, 1.2, "CLI\nmain.py predict\n────────\nTek örnek JSON", COLORS["deploy"], COLORS["deploy_edge"], 7.5)
arrow(12.45, 2.0, 14.8, 1.2)
arrow(12.45, 2.0, 17.2, 1.2)

# Legend
legend_items = [
    mpatches.Patch(facecolor=COLORS["data"], edgecolor=COLORS["data_edge"], label="Veri"),
    mpatches.Patch(facecolor=COLORS["cli"], edgecolor=COLORS["cli_edge"], label="CLI"),
    mpatches.Patch(facecolor=COLORS["process"], edgecolor=COLORS["process_edge"], label="İşlem"),
    mpatches.Patch(facecolor=COLORS["model"], edgecolor=COLORS["model_edge"], label="Model"),
    mpatches.Patch(facecolor=COLORS["output"], edgecolor=COLORS["output_edge"], label="Çıktı"),
    mpatches.Patch(facecolor=COLORS["deploy"], edgecolor=COLORS["deploy_edge"], label="Kullanım"),
]
ax.legend(handles=legend_items, loc="lower left", ncol=6, fontsize=9, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {OUT}")
