"""
predict.py

Batch-predict images in a folder and export a `predictions.csv` file with columns:
[image, class, confidence, x1, y1, x2, y2]

Usage:
    python predict.py --weights models/train9/weights/best.pt --source dataset_split/val/images --out predictions.csv --conf 0.5

"""
import argparse
from ultralytics import YOLO
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', default='predictions.csv')
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.predict(source=args.source, save=True, conf=args.conf)

    data = []
    for r in results:
        img_path = r.path
        # r.boxes can be empty; iterate safely
        for i in range(len(r.boxes)):
            box = r.boxes[i]
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            data.append([img_path, cls_name, conf, x1, y1, x2, y2])

    df = pd.DataFrame(data, columns=['image', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    df.to_csv(args.out, index=False)
    print('CSV saved as', args.out)