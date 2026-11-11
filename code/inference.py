
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output",
        default="predictions.csv"
    )

    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")

    args = parser.parse_args()

    source = Path(args.source)

    image_paths = sorted([
        p for p in source.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in {source}"
        )

    model = YOLO(args.weights)

    rows = []

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Cannot read {image_path}"
            )

        height, width = image.shape[:2]

        processed = preprocess_image(image)

        result = model.predict(
            source=processed,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            agnostic_nms=False,
            save=False,
            verbose=False
        )[0]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids
        ):

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(
                    f"Invalid class ID: {class_id}"
                )

            x1, y1, x2, y2 = map(float, box)

            x1 = float(np.clip(x1, 0, width))
            x2 = float(np.clip(x2, 0, width))
            y1 = float(np.clip(y1, 0, height))
            y2 = float(np.clip(y2, 0, height))

            if x2 <= x1 or y2 <= y1:
                continue

            rows.append({
                "image_name": image_path.name,
                "class_name": CLASS_NAMES[class_id],
                "x_min": round(x1, 2),
                "y_min": round(y1, 2),
                "x_max": round(x2, 2),
                "y_max": round(y2, 2),
                "confidence": round(float(confidence), 6)
            })

    columns = [
        "image_name",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence"
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        rows,
        columns=columns
    ).to_csv(output, index=False)

    print("Images processed:", len(image_paths))
    print("Detections:", len(rows))
    print("Predictions saved:", output)


if __name__ == "__main__":
    main()
