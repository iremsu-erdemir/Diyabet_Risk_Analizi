# Diyabet Risk Analizi

Pima Indians Diabetes veri kümesi üzerinde, ML iş akışına uygun diyabet (şeker hastalığı) sınıflandırma projesi. Komut satırı (CLI) ile eğitim/raporlama ve Streamlit ile interaktif tahmin arayüzü sunar.

## Özellikler

- Problem tanımı ve veri edinimi (Kaggle veya manuel CSV)
- EDA: eksik değer, sınıf dengesi, dağılım, korelasyon, aykırı değer (IQR)
- Ön işleme: imputation, standardizasyon, kategorik one-hot encoding
- Stratified train / validation / test bölmesi (`SEED = 42`)
- Birden fazla sınıflandırıcı + örnekleme yöntemi karşılaştırması (SMOTE ve türevleri)
- 10-fold stratified CV, recall odaklı model seçimi
- Hiperparametre ayarı (L2, öğrenme oranı, derinlik vb.)
- Hata analizi: confusion matrix, sınıf raporu, ROC
- Eğitim / doğrulama loss eğrileri ve overfitting izleme
- Model kaydı (`models/best_model.joblib`) ve CLI ile tek örnek tahmin
- Otomatik metin raporu ve görsel galeri
- Streamlit web arayüzü

## Kurulum

```bash
pip install -r requirements.txt
```

Bağımlılıklar: `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`, `kagglehub`, `streamlit` ve diğerleri (`requirements.txt`).

## Veri indirme

```bash
python main.py download
```

Varsayılan Kaggle veri kümesi: `amineipad/diabetes-dataset`. İnternet ve gerekirse [Kaggle API kimlik doğrulaması](https://www.kaggle.com/docs/api) gerekir.

Alternatif: CSV dosyasını `data/raw/diabetes.csv` yoluna manuel koyun.

## Eğitim ve karşılaştırma

```bash
python main.py train
```

Hedef sütun otomatik bulunamazsa:

```bash
python main.py train --target Outcome
```

Bu adım EDA grafiklerini, tüm model × örnekleme denemelerini, en iyi modelleri ve `models/best_model.joblib` dosyasını üretir. Streamlit arayüzü için önce eğitimin tamamlanmış olması gerekir.

## Rapor üretimi

```bash
python main.py report
```

`train` çalıştırıldıktan sonra metriklerden özet rapor ve EDA galerisi oluşturulur.

### Üretilen çıktılar (`outputs/`)

| Dosya | Açıklama |
|-------|----------|
| `eda_summary.json` | EDA özet istatistikleri |
| `metrics_summary.json` | En iyi modeller, CV ve dağıtım metrikleri |
| `all_model_sampling_results.csv` | Tüm model × örnekleme denemeleri |
| `classifier_summary_best_sampling.csv` | Model başına en iyi örnekleme özeti |
| `per_model_test_metrics.json` | Test hold-out metrikleri (Streamlit karşılaştırma tablosu) |
| `report.md` | Metin sonuç raporu |
| `eda_gallery_report.md` | Grafik galerisi (markdown) |
| `figures/*.png` | Dağılım, loss, confusion matrix, ROC vb. |

`outputs/` ve `models/*.joblib` `.gitignore` içindedir; depoda yoksa `train` ile yeniden üretin.

## Web arayüzü (Streamlit)

```bash
streamlit run app.py
```

Tarayıcıda varsayılan adres: [http://localhost:8501](http://localhost:8501)

### Menü

- **Tahmin:** Kişisel özelliklerle diyabet riski (ana model: HistGradientBoosting + SMOTE + recall eşiği)
- **Model Analizi:** Metrikler, model karşılaştırması, grafikler
- **Proje Hakkında:** Proje ve teknik özet

Tema ayarları: `.streamlit/config.toml`

## Tek örnek tahmin (CLI)

```bash
python main.py predict --sample-json "{\"Pregnancies\":2,\"Glucose\":120,\"BloodPressure\":72,\"SkinThickness\":20,\"Insulin\":79,\"BMI\":28.5,\"DiabetesPedigreeFunction\":0.45,\"Age\":33}"
```

PowerShell için çift tırnak kaçışına dikkat edin veya JSON'u tek tırnaklı bir dosyadan okuyun.

## Yardımcı araçlar (`tools/`)

```bash
# Pipeline mimarisi diyagramı → outputs/figures/pipeline_architecture.png
python tools/draw_pipeline_diagram.py

# Bölüm 5 model metni (eğitim sonrası) → outputs/report_section_5_models.txt
python tools/generate_report_section_5.py
```

## Tekrarlanabilirlik

- Rastgelelik sabiti: `SEED = 42` (`src/pipeline.py`)
- Stratified train / val / test bölmesi
- Ön işleme ve tahmin aynı sklearn `Pipeline` ile eğitim ve çıkarımda tutarlı

## Proje yapısı

```
Diyabet_Risk_Analizi-master/
├── main.py                      # CLI giriş noktası
├── app.py                       # Streamlit web uygulaması
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml              # Streamlit tema
├── data/
│   └── raw/
│       └── diabetes.csv         # Ham veri (indirme veya manuel)
├── src/
│   ├── pipeline.py              # EDA, eğitim, değerlendirme, rapor
│   └── paths.py                 # Yol sabitleri
├── models/
│   └── best_model.joblib        # Eğitim sonrası (gitignore)
├── outputs/                     # Eğitim/rapor çıktıları (gitignore)
│   ├── figures/
│   └── …
└── tools/
    ├── draw_pipeline_diagram.py
    └── generate_report_section_5.py
```

## Örnek kullanım

```bash
git clone <repo-url>
cd Diyabet_Risk_Analizi-master
pip install -r requirements.txt

python main.py download   # veya diabetes.csv'yi data/raw/ altına koy
python main.py train
python main.py report     # isteğe bağlı
streamlit run app.py
```

## Ders kriteri kontrol listesi

- Problem tipi açıkça tanımlandı (ikili sınıflandırma)
- Veri kaynağı ve hedef değişken belirtildi
- EDA: eksik değer, sınıf dengesi, aykırı değer, istatistik
- Train / val / test ayrımı ve deterministik seed
- Ön işleme eğitim ve testte aynı pipeline ile
- Baseline ve birden fazla model aynı metriklerle karşılaştırıldı
- Örnekleme yöntemleri (SMOTE vb.) değerlendirildi
- Hiperparametre optimizasyonu (regularizasyon, öğrenme oranı, derinlik)
- Hata analizi (confusion matrix, sınıf raporu, ROC)
- Model kaydetme, raporlama ve CLI arayüzü
- Streamlit tabanlı web arayüzü
- 10-fold CV ve eğitim / doğrulama loss grafikleri
