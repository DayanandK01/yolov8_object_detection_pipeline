"""
prepare_dataset.py

Converts Pascal VOC .xml annotations (in annotations_dir) and images (images_dir)
into YOLO format under `dataset_split/` and writes a dataset YAML used by YOLOv8.

Usage:
    python prepare_dataset.py --images ./archive/images --ann ./archive/annotations --out dataset_split --train-ratio 0.8

"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import random
from utils import ensure_dir

DEFAULT_CLASS_MAP = {
    "With Helmet": "helmet",
    "Without Helmet": "no_helmet",
}


def convert_box(size, box):
    # size: (width, height); box: (xmin, xmax, ymin, ymax)
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def main(images_dir, annotations_dir, output_dir, train_ratio, classes_map):
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)

    for split in ["train", "val"]:
        ensure_dir(output_dir / split / "images")
        ensure_dir(output_dir / split / "labels")

    image_files = [f.name for f in images_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    random.shuffle(image_files)

    train_count = int(len(image_files) * train_ratio)
    splits = {
        "train": image_files[:train_count],
        "val": image_files[train_count:]
    }

    class_list = list(classes_map.values())

    for split, files in splits.items():
        for img_file in files:
            xml_file = annotations_dir / (Path(img_file).stem + '.xml')
            label_path = output_dir / split / 'labels' / (Path(img_file).stem + '.txt')

            if not xml_file.exists():
                # Skip if annotation missing (optionally you could create empty label file)
                continue

            tree = ET.parse(xml_file)
            root = tree.getroot()

            w = int(root.find('size/width').text)
            h = int(root.find('size/height').text)

            with open(label_path, 'w') as out_file:
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text.strip()
                    if cls_name not in classes_map:
                        continue
                    cls_id = class_list.index(classes_map[cls_name])
                    xmlbox = obj.find('bndbox')
                    b = (
                        float(xmlbox.find('xmin').text),
                        float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymin').text),
                        float(xmlbox.find('ymax').text)
                    )
                    bb = convert_box((w, h), b)
                    out_file.write(f"{cls_id} {' '.join(map(lambda x: f'{x:.6f}', bb))}\n")

            # copy image
            shutil.copy(images_dir / img_file, output_dir / split / 'images' / img_file)

    # write YAML
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"train: {output_dir}/train/images\n")
        f.write(f"val: {output_dir}/val/images\n\n")
        f.write(f"nc: {len(class_list)}\n")
        f.write(f"names: {class_list}\n")

    print('✅ Dataset prepared!')
    print('YAML saved to:', yaml_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--ann', required=True)
    parser.add_argument('--out', default='dataset_split')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    main(args.images, args.ann, args.out, args.train_ratio, DEFAULT_CLASS_MAP)