from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

def handle_robflow_dataset(paths: list[Path]):
    for path in paths:
        coco = COCO(path / Path('_annotations.coco.json'))
        img_dir = path
        image_id = 0

        for image_id in coco.imgs:
            img = coco.imgs[image_id]
            original_image = Image.open(os.path.join(img_dir, img['file_name']))
            image = np.array(original_image)

            # Utwórz czarny obraz o tych samych wymiarach co oryginalny obraz
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            # Pobierz ID kategorii i anotacji
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)

            # Narysuj anotacje na czarnym obrazie
            for ann in anns:
                mask = np.maximum(mask, coco.annToMask(ann) * 255)

            mask_image = Image.fromarray(mask)
            labels_dir = path / 'labels'
            labels_dir.mkdir(parents=True, exist_ok=True)
            mask_image.save(os.path.join(labels_dir, f"{image_id}.png"))
            images_dir = path / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)
            original_image.save(os.path.join(images_dir, f"{image_id}.png"))
