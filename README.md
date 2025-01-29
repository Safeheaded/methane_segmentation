
# ZPO Project - Segmentation of permeable and impermeable areas
This project presents a code responsible for training a unet segmentations model for detecting impermeable areas.

## Dataset
- trained on images provided by our lecturer and gathered by us with QGIS,
- 430 annotated images in total
- No preprocessing was added
- Links to datasets [Dataset1](https://drive.google.com/file/d/1ljnnWSF8dcQW_B2l2LCmRftnyGwotYba/view) [Dataset2](https://drive.google.com/file/d/1-Y9OTHWHCw8Gjus7c04yDKAPLKUpEgrq/view?fbclid=IwY2xjawIHbrdleHRuA2FlbQIxMAABHZA5QgUMeqGrJTDtK0tdnqeJcn0X3QgizQZpDTgbr3nv4avi5jE265_8dQ_aem_Qhj_ZKD1LhR5w386JcLscA) 

## Training
- we used unet with with ressnet34 weights and following parameters:
```
EPOCHS = 3000
BATCH_SIZE = 32
T_MAX = EPOCHS * 344
learning_rate = 2e-4
```
- We used only vertical and horizontal flip for augmentations,
- in order to start training, you need to run main.py script,
- We used python 3.10.11, all requirements are in `requiremenets.txt`,
- remember to create `.env` file with content according to `.env.example`

## Results

| Original | Mask | Prediction |
|----------|-------|-------------|
| ![zdj1](./images/zdj1.PNG) | ![zdj1_mask](./images/zdj1_mask.PNG) | ![zdj1_pred](./images/zdj1_pred.PNG) |
| ![zdj2](./images/zdj2.PNG) | ![zdj2_mask](./images/zdj2_mask.PNG) | ![zdj2_pred](./images/zdj2_pred.PNG) |
| ![zdj3](./images/zdj3.PNG) | ![zdj3_mask](./images/zdj3_mask.PNG) | ![zdj3_pred](./images/zdj3_pred.PNG) |

- Metrics we used to evaluate the model (training/metrics/epoch/valid/dataset_iou , training/metrics/epoch/valid/per_image_iou)

## Results in QGIS
| Original | Mask | Information |
|----------|-------|-------------|
| ![zdj1](./images/Zrzut%20ekranu%202025-01-17%20164455.png) | ![zdj1_mask](./images/Zrzut%20ekranu%202025-01-17%20164431.png) | ![zdj1_pred](./images/Zrzut%20ekranu%202025-01-17%20164402.png) |
| ![zdj2](./images/Zrzut%20ekranu%202025-01-17%20171439.png) | ![zdj2_mask](./images/Zrzut%20ekranu%202025-01-17%20171457.png) | ![zdj2_pred](./images/Zrzut%20ekranu%202025-01-17%20171322.png) |


## Trained model in ONNX ready for `Deepness` plugin
- Downloadable [model](https://drive.google.com/file/d/1wEOb0LlU485C1tyuDuCWIq7yiO5brLXB/view) 
- file used to convert the model to ONNX "file convert_to_onnx.py"

## Demo instructions and video
- a short video of running the model in Deepness (no need for audio), preferably converted to GIF
- We used read orthophotomap of Poznań. You can find it [hear](https://mapy.geoportal.gov.pl/imap/Imgp_2.html?SRS=2180&resources=map:wms@https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/TrueOrtho) 

## People
- Krzysztof Nosal,
- Patryk Marczak
