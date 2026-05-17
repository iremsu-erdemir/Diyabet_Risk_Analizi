"""
Bölüm 5 (ML modelleri) metnini outputs/metrics_summary.json ve
all_model_sampling_results.csv üzerinden güncel sayılarla üretir.

Kullanım: python tools/generate_report_section_5.py
Çıktı: outputs/report_section_5_models.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
METRICS_PATH = OUTPUTS / "metrics_summary.json"
RESULTS_CSV = OUTPUTS / "all_model_sampling_results.csv"
OUT_PATH = OUTPUTS / "report_section_5_models.txt"

MODEL_LABELS = {
    "svm": "SVM",
    "logreg": "Lojistik Regresyon",
    "knn": "k-NN",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "gradient_boosting": "GradientBoostingClassifier",
}

SAMPLING_TR = {
    "-": "kullanılmadı",
    "SMOTE": "SMOTE",
    "SVMSMOTE": "SVMSMOTE",
    "RandomOverSampler": "RandomOverSampler",
    "InstanceHardnessThreshold": "InstanceHardnessThreshold",
    "AllKNN": "AllKNN",
    "BorderlineSMOTE": "BorderlineSMOTE",
    "KMeansSMOTE": "KMeansSMOTE",
    "ADASYN": "ADASYN",
}


def fmt(x: float | None, decimals: int = 4) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{decimals}f}".replace(".", ",")


def best_row_per_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for clf in df["classifier"].unique():
        sub = df[df["classifier"] == clf]
        rows.append(sub.sort_values(metric, ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def metrics_row(r: pd.Series) -> str:
    return (
        f"{fmt(r['accuracy'])}\t{fmt(r['precision'])}\t{fmt(r['recall'])}\t"
        f"{fmt(r['f1'])}\t{fmt(r['roc_auc'])}"
    )


def main() -> None:
    if not METRICS_PATH.exists() or not RESULTS_CSV.exists():
        raise FileNotFoundError("Önce `python main.py train` çalıştırın.")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(RESULTS_CSV)
    roc_best = best_row_per_model(df, "roc_auc")
    recall_best = best_row_per_model(df, "recall")

    best_recall = metrics.get("best_recall_overall") or metrics["best_overall"]
    best_f1 = metrics.get("best_f1_overall") or metrics["best_overall"]
    recall_cv = best_recall.get("cv_scores") or metrics.get("reference_recall_best_full_data_cv", {})
    f1_cv = best_f1.get("cv_scores") or metrics.get("reference_f1_best_full_data_cv", {})
    deploy_test = metrics["best_cv_scores"]
    deploy_val = metrics.get("validation_holdout_scores", {})
    dep = metrics.get("deployment_model", {})
    gbh = metrics.get("gradient_boosting_hyperparameter_search", {})
    bp = gbh.get("best_params", {})
    hist_tuned = gbh.get("tuned_cv_means_train_portion", {})

    lines: list[str] = []
    w = lines.append

    w("5. KULLANILAN MAKİNE ÖĞRENMESİ MODELLERİ")
    w("")
    w(
        "Bu çalışmada diyabet tahmini problemi için farklı makine öğrenmesi algoritmaları ortak bir "
        "ön işleme ve değerlendirme altyapısı içerisinde karşılaştırılmıştır. Tüm modeller aynı veri "
        "dönüşüm adımlarını içeren bir pipeline yapısı üzerinden çalıştırılmıştır."
    )
    w(
        "Ön işleme adımları ColumnTransformer yapısı ile tanımlanmıştır. Kullanılan Pima Indians diyabet "
        "veri kümesinde tüm öngörücü değişkenler sayısal olduğundan kategorik encoding adımı bu çalışmada "
        "fiilen devreye girmemiştir; pipeline genelleştirilebilir olması için kategorik kolonlar ayrı kolonda "
        "tanımlanmış, ileride kategorik özellik içeren veri kümelerinde en sık değer ile doldurma ve "
        "one-hot encoding uygulanacak şekilde bırakılmıştır. Sayısal değişkenlerde eksik değerler medyan "
        "ile doldurulmuş, IQR tabanlı aykırı değer sınırlandırması uygulanmış ve StandardScaler ile "
        "ölçekleme yapılmıştır."
    )
    w(
        "Modelleme sürecinde imblearn.pipeline.Pipeline yapısı kullanılarak ön işleme ve sınıflandırıcılar "
        "tek bir yapı altında birleştirilmiştir. Sınıf dengesizliği bulunan senaryolarda SMOTE ve benzeri "
        "örnekleme yöntemleri deneysel olarak değerlendirilmiştir. Karşılaştırma ızgarasında SMOTE için "
        "k_neighbors=5, üretim (HistGradientBoosting) boru hattında ise daha küçük eğitim alt kümesine uyum "
        "için k_neighbors=3 kullanılmıştır."
    )
    w(
        "Model performansı 10 katlı stratified cross-validation ile ölçülmüş; accuracy, precision, recall, "
        "F1-score ve ROC-AUC metrikleri raporlanmıştır. Altı klasik sınıflandırıcı için örneklemesiz durum "
        "ve 14 yeniden örnekleme yöntemi birlikte değerlendirilerek toplam 90 model–örnekleme kombinasyonu "
        "oluşturulmuştur; HistGradientBoostingClassifier bu ızgaranın dışında tutulmuş, üretim modeli olarak "
        "ayrı bir RandomizedSearchCV ve eşik optimizasyonu hattında raporlanmıştır. Klinik tarama bağlamında "
        "yanlış negatif maliyeti yüksek olduğundan, 90 kombinasyonluk ızgarada birincil seçim ölçütü recall "
        "(eşit recall değerlerinde F1-score ile kırılım) olarak tanımlanmıştır. ROC-AUC odaklı karşılaştırma "
        "ayrıca sürdürülmüştür. Operasyonel tahminler için HistGradientBoosting + SMOTE (k_neighbors=3) + "
        "recall bandına göre ayarlanmış sınıflandırma eşiği kullanılmıştır."
    )
    w("")
    w("Tablo 5.1. Modellerin Karşılaştırmalı Performans Sonuçları (en yüksek ROC-AUC kombinasyonu, 10-fold CV)")
    w("Model\tAccuracy\tPrecision\tRecall\tF1-Score\tROC-AUC\tEn iyi örnekleme")
    for _, r in roc_best.iterrows():
        clf = MODEL_LABELS.get(r["classifier"], r["classifier"])
        samp = SAMPLING_TR.get(str(r["sampling_method"]), str(r["sampling_method"]))
        w(f"{clf}\t{metrics_row(r)}\t{samp}")
    w(
        f"HistGradientBoosting\t{fmt(hist_tuned.get('accuracy'))}\t{fmt(hist_tuned.get('precision'))}\t"
        f"{fmt(hist_tuned.get('recall'))}\t{fmt(hist_tuned.get('f1'))}\t{fmt(hist_tuned.get('roc_auc'))}\t"
        "SMOTE (5-fold CV, eğitim alt kümesi)"
    )
    w("")
    w(
        "a) HistGradientBoosting, 90 kombinasyonluk karşılaştırma ızgarasına dahil edilmemiştir. Tablodaki "
        "satır, RandomizedSearchCV sonrası eğitim alt kümesinde 5 katlı çapraz doğrulama ortalamalarını "
        "temsil eder; tam veri 10-fold ızgarası ile birebir karşılaştırılmamalıdır."
    )
    w(
        "Tablo 5.1 yorumu. Altı klasik model için, tüm örnekleme kombinasyonları arasından en yüksek "
        "ortalama ROC-AUC veren 10-fold stratified CV satırları seçilmiştir; ızgarada SMOTE k_neighbors=5 "
        "kullanılmıştır. Metrikler dört ondalık basamağa yuvarlanmıştır."
    )
    w("")
    w("Tablo 5.2. Kullanılan Modeller ve Temel Yapıları")
    w("Model\tSklearn Sınıfı\tAçıklama")
    w("SVM\tSVC\tRBF kernel ile maksimum marjin sınıflandırma")
    w("Logistic Regression\tLogisticRegression\tElastic net düzenlileştirme ile doğrusal model")
    w("KNN\tKNeighborsClassifier\tMesafe tabanlı sınıflandırma")
    w("Random Forest\tRandomForestClassifier\tAğaç tabanlı topluluk (bagging)")
    w("AdaBoost\tAdaBoostClassifier\tBoosting tabanlı zayıf öğreniciler")
    w("Gradient Boosting\tGradientBoostingClassifier\tHata azaltmaya dayalı boosting")
    w(
        "HistGradientBoosting\tHistGradientBoostingClassifier\tHistogram tabanlı hızlı boosting "
        "(üretim / dağıtım modeli; 90’lık karşılaştırma ızgarası dışı; RandomizedSearchCV + recall eşiği)"
    )
    w("")
    w("Tablo 5.3. Ortak Pipeline Yapısı")
    w("Bileşen\tKullanılan Yöntem")
    w("Eksik veri tamamlama\tSimpleImputer (median / most_frequent)")
    w("Aykırı değer işleme\tIQR clipping")
    w("Ölçekleme\tStandardScaler")
    w("Kategorik encoding (bu veri setinde uygulanmadı)\tOneHotEncoder (tüm öngörücüler sayısal)")
    w("Pipeline yapısı\tColumnTransformer + ImbPipeline")
    w("Sınıf dengesizliği (ızgara)\tSMOTE (k_neighbors=5) ve alternatif over/under-sampling")
    w("Sınıf dengesizliği (üretim)\tSMOTE (k_neighbors=3)")
    w("Doğrulama yöntemi\tStratified 10-fold Cross Validation")
    w("")
    w("Tablo 5.4. Model Değerlendirme Stratejisi")
    w("Yöntem\tAçıklama")
    w("Cross Validation\t10-fold StratifiedKFold")
    w("Train / Validation / Test\t%64 eğitim + %16 doğrulama / %20 test (stratified, seed=42)")
    w("Metrikler\tAccuracy, Precision, Recall, F1, ROC-AUC")
    w(
        "Karşılaştırma ızgarası\t6 sınıflandırıcı × (örneklemesiz + 14 örnekleyici) = 90 kombinasyon "
        "(HistGradientBoosting dahil değil)"
    )
    w("Örnekleme (ızgara)\tSMOTE (k_neighbors=5) ve alternatif over/under-sampling (model başına en iyi kombinasyon)")
    w("Pipeline\tPreprocess → (Sampler) → Model")
    w("Izgara birincil seçim\tEn yüksek ortalama recall (eşitlikte F1-score)")
    w("Izgara referans seçim\tEn yüksek ortalama F1-score (dengeli metrik karşılaştırması)")
    w(
        "Üretim modeli\tHistGradientBoosting + SMOTE (k_neighbors=3); RandomizedSearchCV (roc_auc); "
        "sınıflandırma eşiği recall %90–%95 bandına göre"
    )
    w("")
    bp_str = (
        f"learning_rate={fmt(bp.get('model__learning_rate'), 4)}, max_depth={int(bp.get('model__max_depth', 3))}, "
        f"max_iter={int(bp.get('model__max_iter', 0))}, min_samples_leaf={int(bp.get('model__min_samples_leaf', 0))}, "
        f"l2_regularization={fmt(bp.get('model__l2_regularization'), 4)}, "
        f"n_iter_no_change={int(bp.get('model__n_iter_no_change', 0))}"
    )
    w("Tablo 5.5. Modellerde Kullanılan Temel Hiperparametreler")
    w("Model\tKullanılan hiperparametreler")
    w("SVM\tkernel=rbf, C=0.45, gamma=scale, probability=True, random_state=42")
    w(
        "Lojistik Regresyon\tmax_iter=1200, C=0.7, penalty=elasticnet, l1_ratio=0.32, "
        "solver=saga, random_state=42"
    )
    w("k-NN\tn_neighbors=15, weights=distance, metric=minkowski, p=2")
    w(
        "Random Forest\tn_estimators=120, max_depth=5, min_samples_split=8, min_samples_leaf=6, "
        "max_features=sqrt, random_state=42"
    )
    w(f"AdaBoost\tn_estimators={metrics.get('adaboost_validation_early_stop', {}).get('adaboost_n_estimators_after_val_early_stop', 7)}, learning_rate=0.55, random_state=42 (doğrulama log-loss ile erken durdurma)")
    w(
        "GradientBoostingClassifier\tn_estimators=100, learning_rate=0.05, max_depth=3, "
        "min_samples_split=8, min_samples_leaf=6, subsample=0.85, max_features=sqrt, random_state=42"
    )
    w(
        f"HistGradientBoosting\tRandomizedSearchCV best_params_: {bp_str}; "
        "sabit: early_stopping=True, validation_fraction=0.18, max_bins=64, random_state=42; "
        "boru hattında SMOTE: k_neighbors=3, random_state=42"
    )
    w("")
    w(
        "Tablo 5.5’teki hiperparametreler, kaynak kodda tanımlanan yapılandırmalar ile "
        "RandomizedSearchCV.best_params_ çıktıları üzerinden derlenmiştir. Klasik modellerin "
        "karşılaştırma ızgarasında SMOTE k_neighbors=5 iken, üretim boru hattında k_neighbors=3 "
        "kullanılmıştır."
    )
    w("")
    w("Tablo 5.6. Modellerin Recall Odaklı Performansı (en yüksek recall kombinasyonu, 10-fold CV)")
    w("Model\tAccuracy\tPrecision\tRecall\tF1-Score\tROC-AUC\tEn iyi örnekleme")
    for _, r in recall_best.iterrows():
        clf = MODEL_LABELS.get(r["classifier"], r["classifier"])
        samp = SAMPLING_TR.get(str(r["sampling_method"]), str(r["sampling_method"]))
        w(f"{clf}\t{metrics_row(r)}\t{samp}")
    w("")
    w("Tablo 5.7. Izgara Şampiyonları ve Dağıtım Modeli")
    w("Rol\tModel\tÖrnekleme\tAccuracy\tPrecision\tRecall\tF1\tROC-AUC")
    w(
        f"Birincil (recall)\t{MODEL_LABELS.get(best_recall['classifier'], best_recall['classifier'])}\t"
        f"{SAMPLING_TR.get(best_recall['sampling_method'], best_recall['sampling_method'])}\t"
        f"{fmt(recall_cv.get('accuracy'))}\t{fmt(recall_cv.get('precision'))}\t{fmt(recall_cv.get('recall'))}\t"
        f"{fmt(recall_cv.get('f1'))}\t{fmt(recall_cv.get('roc_auc'))}"
    )
    w(
        f"F1 referansı\t{MODEL_LABELS.get(best_f1['classifier'], best_f1['classifier'])}\t"
        f"{SAMPLING_TR.get(best_f1['sampling_method'], best_f1['sampling_method'])}\t"
        f"{fmt(f1_cv.get('accuracy'))}\t{fmt(f1_cv.get('precision'))}\t{fmt(f1_cv.get('recall'))}\t"
        f"{fmt(f1_cv.get('f1'))}\t{fmt(f1_cv.get('roc_auc'))}"
    )
    w(
        f"Dağıtım (test hold-out)\tHistGradientBoosting\tSMOTE + eşik\t"
        f"{fmt(deploy_test.get('accuracy'))}\t{fmt(deploy_test.get('precision'))}\t"
        f"{fmt(deploy_test.get('recall'))}\t{fmt(deploy_test.get('f1'))}\t{fmt(deploy_test.get('roc_auc'))}"
    )
    w(
        f"Dağıtım (doğrulama)\tHistGradientBoosting\tSMOTE + eşik\t"
        f"{fmt(deploy_val.get('accuracy'))}\t{fmt(deploy_val.get('precision'))}\t"
        f"{fmt(deploy_val.get('recall'))}\t{fmt(deploy_val.get('f1'))}\t{fmt(deploy_val.get('roc_auc'))}"
    )
    w("")
    w("Tablo 5.8. Hiperparametre ve Örnekleme Seçimlerinin Performansa Etkisi (Tablo 5.1 ile tutarlı, ROC-AUC)")
    w("Model\tKritik hiperparametreler / örnekleme\tAccuracy\tF1-Score\tROC-AUC")
    for _, r in roc_best.iterrows():
        clf = MODEL_LABELS.get(r["classifier"], r["classifier"])
        samp = SAMPLING_TR.get(str(r["sampling_method"]), str(r["sampling_method"]))
        hp_notes = {
            "svm": "kernel=rbf, C=0.45, gamma=scale",
            "logreg": "C=0.7, penalty=elasticnet, l1_ratio=0.32",
            "knn": "n_neighbors=15, weights=distance",
            "random_forest": "max_depth=5, n_estimators=120, min_samples_leaf=6",
            "adaboost": "n_estimators=7 (doğrulama temelli durdurma), learning_rate=0.55",
            "gradient_boosting": "learning_rate=0.05, max_depth=3, subsample=0.85",
        }
        hp = hp_notes.get(r["classifier"], "")
        w(f"{clf}\t{hp}; örnekleme: {samp}\t{fmt(r['accuracy'])}\t{fmt(r['f1'])}\t{fmt(r['roc_auc'])}")
    w(
        f"HistGradientBoosting\t{bp_str}; SMOTE k_neighbors=3\t"
        f"{fmt(hist_tuned.get('accuracy'))}\t{fmt(hist_tuned.get('f1'))}\t{fmt(hist_tuned.get('roc_auc'))}"
    )
    w("")
    w("5.4 Bulguların Değerlendirilmesi")
    w("")
    w(
        f"Çapraz doğrulama sonuçları, tam veri üzerindeki 10 katlı stratified değerlendirmede en yüksek "
        f"ROC-AUC değerinin {fmt(roc_best.loc[roc_best['classifier']=='gradient_boosting', 'roc_auc'].iloc[0])} "
        f"ile GradientBoostingClassifier modelinde ve örnekleme yönteminin kullanılmadığı yapılandırmada "
        f"elde edildiğini göstermektedir. Random Forest ({fmt(roc_best.loc[roc_best['classifier']=='random_forest', 'roc_auc'].iloc[0])}) "
        f"ile SVM ve Lojistik Regresyon modelleri ({fmt(roc_best.loc[roc_best['classifier']=='svm', 'roc_auc'].iloc[0])} – "
        f"{fmt(roc_best.loc[roc_best['classifier']=='logreg', 'roc_auc'].iloc[0])} arası) benzer ayrım gücü bandında "
        f"yer almaktadır. AdaBoost ({fmt(roc_best.loc[roc_best['classifier']=='adaboost', 'roc_auc'].iloc[0])}) ve "
        f"k-NN ({fmt(roc_best.loc[roc_best['classifier']=='knn', 'roc_auc'].iloc[0])}) modelleri ise aynı ölçüt altında "
        f"daha düşük performans sergilemektedir."
    )
    w(
        f"HistGradientBoosting, 90 kombinasyonluk karşılaştırma ızgarasına dahil edilmemiş; raporlanan "
        f"{fmt(hist_tuned.get('roc_auc'))} ROC-AUC değeri 5 katlı iç çapraz doğrulama protokolüne aittir ve "
        f"Tablo 5.1’teki 10-fold ızgara satırlarıyla doğrudan sıralanmamalıdır. "
        f"Buna karşılık üretim modelinde early stopping, L2 düzenlileştirme ve recall bandına göre eşik seçimi "
        f"({fmt(dep.get('optimal_threshold_validation_split'), 4)} doğrulama; "
        f"{fmt(dep.get('optimal_threshold_full_data_oof'), 4)} tam veri OOF) ile tutulan test kümesinde recall "
        f"{fmt(deploy_test.get('recall'))}, doğrulama kümesinde recall {fmt(deploy_val.get('recall'))} olarak ölçülmüştür."
    )
    w(
        "k-NN modeline ilişkin log-loss özetinde, eğitim kümesinde son derece düşük ortalama log-loss değeri ile "
        "doğrulama kümesinde belirgin biçimde daha yüksek log-loss değerinin birlikte gözlemlenmesi, aşırı uyum "
        "(overfitting) riskinin k-NN modeli özelinde güçlü olduğuna işaret etmektedir."
    )
    w(
        f"Klinik öncelik (yüksek duyarlılık) açısından değerlendirildiğinde, altı klasik modelden oluşan "
        f"90 kombinasyonluk ızgarada en yüksek "
        f"ortalama recall değeri {MODEL_LABELS.get(best_recall['classifier'], best_recall['classifier'])} ve "
        f"{SAMPLING_TR.get(best_recall['sampling_method'], best_recall['sampling_method'])} ile "
        f"{fmt(recall_cv.get('recall'))} olarak elde edilmiştir (Tablo 5.7). Aynı protokolde F1 referansı "
        f"{MODEL_LABELS.get(best_f1['classifier'], best_f1['classifier'])} + "
        f"{SAMPLING_TR.get(best_f1['sampling_method'], best_f1['sampling_method'])} kombinasyonunda "
        f"F1={fmt(f1_cv.get('f1'))}, recall={fmt(f1_cv.get('recall'))} düzeyindedir; yani yüksek F1 ile en yüksek "
        f"recall aynı yapılandırmada çakışmamaktadır."
    )
    w(
        f"Operasyonel tahminlerde kullanılan HistGradientBoosting + SMOTE (k_neighbors=3) + eşikli model, tutulan test "
        f"kümesinde recall={fmt(deploy_test.get('recall'))} sunarak yanlış negatifleri azaltmayı hedeflemiş; "
        f"bu değer, ROC-AUC şampiyonu GradientBoosting’in örneklemsiz yapılandırmasındaki recall "
        f"({fmt(roc_best.loc[roc_best['classifier']=='gradient_boosting', 'recall'].iloc[0])}) değerinden belirgin "
        f"şekilde yüksektir. Precision ({fmt(deploy_test.get('precision'))}) düşüşü, yüksek duyarlılık hedefinin "
        f"doğal sonucu olarak değerlendirilmiştir."
    )
    w(
        "Topluluk ve klasik modeller karşılaştırıldığında, en yüksek ROC-AUC değerlerinin GradientBoostingClassifier "
        "ve RandomForestClassifier ile elde edilmesi, tabular diyabet veri kümesinde ağaç tabanlı birleşik modellerin "
        "güçlü ayrım kapasitesi sunduğunu göstermektedir. Recall odaklı seçimde ise SVM ve InstanceHardnessThreshold "
        "gibi under-sampling stratejilerinin pozitif sınıfı daha agresif yakaladığı görülmüştür. Sonuç olarak, "
        "raporlamada hem ROC-AUC (ayırt etme gücü) hem recall (klinik tarama) hem de dağıtım modeli metrikleri "
        "ayrı tablolarda sunulmuştur."
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Yazıldı: {OUT_PATH}")


if __name__ == "__main__":
    main()
