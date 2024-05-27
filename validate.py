import tqdm
import pickle

import pandas as pd
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset

from sklearn.metrics import f1_score, jaccard_score

from utils.metrics import dice_score

import logging

logging.basicConfig(level=logging.INFO)

def validate(model_path: str, x: np.ndarray, y: np.ndarray, custom_objects: dict = {}, path_predictions: str = None):
    """
    Validate the model

    Args:
    model: tf.keras.Model: Model to validate
    x: np.ndarray: Input data
    y: np.ndarray: Target data
    """
    try:
        model = load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    
    SCORE = []

    y_preds = []

    for x, y in tqdm(zip(x, y), total=len(y)):
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        dice = dice_score(y, y_pred)
        SCORE.append([f1_value, jac_value, dice])

        y_preds.append(y_pred)

    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)

    logging.info(f"F1 Score : {score[0]}, Jaccard Score : {score[1]}, Dice Score : {score[2]}")

    df = pd.DataFrame(SCORE, columns=["Image", "F1", "Jaccard", "Dice"])
    df.to_csv("files/score.csv", index=None)
    
    with open(path_predictions + "predictions.pkl", "wb") as f:
        pickle.dump(y_preds, f)

