from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Grafik/metin: çok küçük negatifleri (ör. -0.001) sıfır gibi göster
DISPLAY_ZERO_EPS = 5e-4


@dataclass
class Paths:
    project_root: Path
    raw_data_path: Path
    model_path: Path
    metrics_dir: Path
    figures_dir: Path
    report_path: Path
    eda_gallery_report_path: Path


def get_paths(project_root: Path | None = None) -> Paths:
    root = project_root or Path(__file__).resolve().parents[1]
    return Paths(
        project_root=root,
        raw_data_path=root / "data" / "raw" / "diabetes.csv",
        model_path=root / "models" / "best_model.joblib",
        metrics_dir=root / "outputs",
        figures_dir=root / "outputs" / "figures",
        report_path=root / "outputs" / "report.md",
        eda_gallery_report_path=root / "outputs" / "eda_gallery_report.md",
    )


def snap_display_scalar(value: float, eps: float = DISPLAY_ZERO_EPS) -> float:
    """Grafik/metin için -eps..0 aralığındaki değerleri 0'a çeker (ör. -0.001 → 0)."""
    if -eps < value < 0:
        return 0.0
    if 0 <= value < eps:
        return 0.0
    return float(value)
