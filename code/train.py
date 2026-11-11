
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def preprocess_image(image):
    """Apply CLAHE to LAB luminance without changing geometry."""

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([
        enhanced_l,
        a_channel,
        b_channel
    ])

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )


def prepare_dataset(source_root, output_root):

    source_root = Path(source_root)
    output_root = Path(output_root)

    if not source_root.exists():
        raise FileNotFoundError(source_root)

    for split in ["train", "valid", "test"]:

        source_images = source_root / split / "images"
        source_labels = source_root / split / "labels"

        output_images = output_root / split / "images"
        output_labels = output_root / split / "labels"

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        images = sorted([
            p for p in source_images.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        labels = {
            p.stem: p
            for p in source_labels.glob("*.txt")
        }

        for image_path in images:

            if image_path.stem not in labels:
                raise FileNotFoundError(
                    f"Missing label for {image_path.name}"
                )

            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(
                    f"Cannot read {image_path}"
                )

            processed = preprocess_image(image)

            destination = output_images / image_path.name

            if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                success = cv2.imwrite(
                    str(destination),
                    processed,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            else:
                success = cv2.imwrite(
                    str(destination),
                    processed
                )

            if not success:
                raise RuntimeError(
                    f"Cannot save {destination}"
                )

        for label_path in labels.values():
            shutil.copy2(
                label_path,
                output_labels / label_path.name
            )

        print(
            f"{split}: {len(images)} images prepared"
        )

    data_yaml = output_root / "data.yaml"

    config = {
        "path": str(output_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 6,
        "names": CLASS_NAMES
    }

    data_yaml.write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    return data_yaml


class PCBCompetitionTrainer(DetectionTrainer):

    def validate(self):

        previous_best = self.best_fitness

        metrics, fitness = super().validate()

        if metrics is None:
            return metrics, fitness

        all_ap = np.asarray(
            self.validator.metrics.box.all_ap,
            dtype=np.float64
        )

        class_ids = np.asarray(
            self.validator.metrics.box.ap_class_index,
            dtype=int
        )

        if all_ap.shape != (6, 10):
            raise RuntimeError(
                f"Unexpected AP matrix shape: {all_ap.shape}"
            )

        if set(class_ids.tolist()) != set(range(6)):
            raise RuntimeError(
                "Validation must contain all six classes."
            )

        ap_by_class = np.full(
            (6, 10),
            np.nan,
            dtype=np.float64
        )

        for row, class_id in enumerate(class_ids):
            ap_by_class[class_id] = all_ap[row]

        competition_map = float(
            ap_by_class[:, :9].mean()
        )

        if not np.isfinite(competition_map):
            raise RuntimeError(
                "Invalid competition mAP@50:90."
            )

        metrics["metrics/mAP50-90(B)"] = competition_map

        fitness = competition_map

        self.best_fitness = (
            fitness if previous_best is None
            else max(float(previous_best), fitness)
        )

        print(
            f"Competition mAP@50:90: {competition_map:.6f}"
        )

        return metrics, fitness


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default="/kaggle/input/datasets/abankitabehera/pcb-defects"
    )

    parser.add_argument(
        "--preprocessed-root",
        default="./pcb_defects_clahe"
    )

    parser.add_argument(
        "--weights",
        default="yolo26m.pt"
    )

    parser.add_argument(
        "--project",
        default="./pcb_runs"
    )

    parser.add_argument(
        "--device",
        default="0,1"
    )

    args = parser.parse_args()

    data_yaml = prepare_dataset(
        args.dataset_root,
        args.preprocessed_root
    )

    if "," in args.device:
        device = [
            int(x.strip())
            for x in args.device.split(",")
        ]
    else:
        device = int(args.device)

    model = YOLO(args.weights)

    model.train(
        data=str(data_yaml),

        imgsz=896,
        epochs=120,
        patience=30,
        batch=8,
        device=device,
        workers=2,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2,
        cos_lr=True,

        hsv_h=0.005,
        hsv_s=0.08,
        hsv_v=0.12,

        degrees=0.0,
        translate=0.03,
        scale=0.10,

        fliplr=0.5,
        flipud=0.5,

        mosaic=0.10,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,

        augmentations=[],
        cls_pw=0.0,

        amp=True,
        cache=False,
        multi_scale=False,

        save=True,
        save_period=5,
        plots=True,
        val=True,

        project=args.project,
        name="YOLO26m_896_CLAHE_Competition",
        exist_ok=False,

        resume=False,
        seed=42,
        deterministic=True,

        trainer=PCBCompetitionTrainer
    )


if __name__ == "__main__":
    main()
