# DL_Project
# Hot vs Cold Drink Classifier

A Flask web app that classifies drink images as hot or cold using a fine-tuned MobileNetV2 model.

Where to obtain datasets?

https://www.kaggle.com/datasets/rajkumarl/tea-time - Works for hot drinks (Can select manually for even better accuracy)
https://www.kaggle.com/datasets/faseeh001/cold-drinks-inventory-dataset - Works for cold drinks (Manual selection recommended due to overfitting errors)

## Steps
1. Prepare dataset (`python prepare_dataset.py`)
2. Train model (`python train_model.py`)
3. Run app (`python app.py`)

Upload a drink image to get instant classification!
