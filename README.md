# YOLOv8 Object Detection Pipeline

## A simple, step-by-step YOLOv8 pipeline for:

1. Converting Pascal VOC XML annotations to YOLO format.

2. Splitting data into training/validation sets.

3. Training a YOLOv8 model.

4. Running predictions and exporting results.

# kaggle dataset link
 * https://www.kaggle.com/datasets/andrewmvd/helmet-detection?utm_source=chatgpt.com
## 1. Setup
### Clone this repo
```bash
python -m venv venv
```
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

## 2. Prepare the dataset
```bash
python prepare_dataset.py \
    --images ./archive/images \
    --ann ./archive/annotations \
    --out dataset_split \
    --train-ratio 0.8 \
```
### This creates a dataset_split/ folder with train and val images + labels, and a data.yaml file.

## 3. Train the model 
```bash
python train.py \
    --data dataset_split/data.yaml \
    --model yolov8n.pt \
    --epochs 10 \
    --imgsz 640 \
    --batch 16 \
```
### Your trained weights will be saved in runs/detect/train*/weights/best.pt.

## 4. Run predictions
```bash
python predict.py \
    --weights models/train9/weights/best.pt \
    --source dataset_split/val/images \
    --out predictions.csv \
    --conf 0.5 \
```
### Predicted images will be saved in a runs/predict folder, and results in predictions.csv.

# Tips

       Use absolute paths if you get FileNotFoundError.

       Do not commit large files (.pt, runs/) to GitHub.

       Adjust --conf to control detection confidence.
