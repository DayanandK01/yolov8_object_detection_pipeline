"""
train.py

Simple script to train YOLOv8 using the ultralytics API.

Usage:
    python train.py --data dataset_split/data.yaml --epochs 10 --imgsz 640 --batch 16 --save-period 5

"""
import argparse
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset_split/data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-period', type=int, default=5)
    args = parser.parse_args()

    # Load model (local path or official weights name)
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        save_period=args.save_period
    )