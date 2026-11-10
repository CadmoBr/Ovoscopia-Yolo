import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data.yaml"
RESULTS_CSV = ROOT / "metricas_treinamento.csv"


def validate_dataset() -> None:
    required_dirs = [ROOT / "train" / "images", ROOT / "valid" / "images", ROOT / "test" / "images"]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Diretórios de dados ausentes: {missing}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Arquivo de configuração do dataset não encontrado: {DATA_YAML}")


def save_metrics_csv(metrics_dict: dict, filename: str = "metricas_treinamento.csv") -> None:
    csv_path = ROOT / filename
    rows = []
    for key, value in metrics_dict.items():
        rows.append({"metric": key, "value": value})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Métricas salvas em: {csv_path}")


def train_model(epochs: int = 50, imgsz: int = 640, batch: int = 8) -> YOLO:
    validate_dataset()
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "runs" / "detect"),
        name="train_final",
        exist_ok=True,
    )
    return model


def evaluate_model(model: YOLO) -> dict:
    metrics = model.val(data=str(DATA_YAML), split="test")
    metrics_dict = metrics.results_dict
    print(metrics_dict)
    save_metrics_csv(metrics_dict)
    return metrics_dict


def predict_images(model: YOLO, images: list[str], conf: float = 0.25) -> None:
    for image in images:
        path = ROOT / image
        if not path.exists():
            print(f"Imagem não encontrada: {path}")
            continue

        model.predict(
            source=str(path),
            conf=conf,
            save=True,
            project=str(ROOT / "runs" / "detect"),
            name="predict_final",
        )


if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)
    predict_images(model, ["ovo.jpeg", "ovo2.jpg", "ovo3.jpg", "ovo4.jpg"])
