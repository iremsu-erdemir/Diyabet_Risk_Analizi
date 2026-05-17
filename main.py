from __future__ import annotations

import argparse
import json

from src.pipeline import (
    download_kaggle_dataset,
    generate_report,
    infer_target_column,
    load_data,
    predict_single,
    run_eda,
    train_and_compare,
)


def cmd_download(_: argparse.Namespace) -> None:
    """Kaggle veri kümesini indirip standart ham veri klasörüne kopyalar."""
    path = download_kaggle_dataset()
    print(f"Veri kümesi indirildi: {path}")


def cmd_train(args: argparse.Namespace) -> None:
    """EDA + eğitim + değerlendirme adımlarını uçtan uca çalıştırır."""
    # Veri kümesini yükle ve hedef sütunu kullanıcı argümanından veya otomatik tespitten al.
    df = load_data()
    target_col = args.target if args.target else infer_target_column(df)
    # Önce görsel/istatistiksel EDA çıktıları üretilir.
    run_eda(df, target_col)
    # Ardından model eğitimi, karşılaştırma ve metrik kayıt işlemleri yapılır.
    results = train_and_compare(df, target_col)
    print("Eğitim tamamlandı.")
    print(json.dumps(results["best_cv_scores"], indent=2, ensure_ascii=False))


def cmd_report(_: argparse.Namespace) -> None:
    """Eğitim sonrası metriklerden metin raporu ve görsel galeri üretir."""
    path = generate_report()
    print(f"Rapor oluşturuldu: {path}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Komut satırından verilen tek örnek için model tahmini yapar."""
    # JSON metnini sözlüğe çevirip tahmin fonksiyonuna gönder.
    sample = json.loads(args.sample_json)
    result = predict_single(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    """CLI komutlarını ve parametrelerini tanımlar."""
    parser = argparse.ArgumentParser(description="Şeker hastalığı tahmin projesi")
    sub = parser.add_subparsers(dest="command", required=True)

    # Veri indirme komutu.
    p_download = sub.add_parser("download", help="Kaggle veri kümesini indir")
    p_download.set_defaults(func=cmd_download)

    # Eğitim komutu.
    p_train = sub.add_parser("train", help="EDA + model eğitimi + tuning")
    p_train.add_argument(
        "--target", type=str, default=None, help="Hedef sütun adı (opsiyonel)"
    )
    p_train.set_defaults(func=cmd_train)

    # Rapor üretme komutu.
    p_report = sub.add_parser("report", help="Rapor dosyası oluştur")
    p_report.set_defaults(func=cmd_report)

    # Tek örnek tahmin komutu.
    p_predict = sub.add_parser("predict", help="Tek örnek için tahmin yap")
    p_predict.add_argument(
        "--sample-json",
        type=str,
        required=True,
        help='Örnek: \'{"Pregnancies":2,"Glucose":120,...}\'',
    )
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    """Uygulamanın giriş noktası: parser'ı çalıştırır ve ilgili komutu çağırır."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
