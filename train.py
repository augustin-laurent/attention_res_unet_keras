import numpy as np
import logging

from data.load_data import load_data_test, load_data_train, create_dataset

from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import AdamW
from tensorflow.data import Dataset

from utils.loss_function import BCEDice, iou_coeff, dice_coeff
from model.attentionresunet import Attention_ResUNet

logging.basicConfig(level=logging.INFO)

def train_model(model: Model, train_dataset: Dataset, valid_dataset: Dataset, save_dir: str, csv_dir: str, model_name: str, batch_size: int = 16, epochs: int = 20, learning_rate: float = 1e-4, class_number: int = 1, input_shape: tuple = (256, 256, 3), resume: bool = False) -> None:
    """
    Train the model

    Args:
    model: tf.keras.Model: Model to train
    train_dataset: tf.data.Dataset: Training dataset
    valid_dataset: tf.data.Dataset: Validation dataset
    save_dir: str: Directory to save the model
    csv_dir: str: Directory to save the training history
    model_name: str: Name of the model
    batch_size: int: Batch size for training
    epochs: int: Number of epochs to train the model
    learning_rate: float: Learning rate
    class_number: int: Number of classes
    input_shape: tuple: Input shape of the model
    dropout_rate: float: Dropout rate
    batch_norm: bool: Batch normalization
    resume: bool: Resume training
    """

    if resume:
        try:
            model = load_model(f"{save_dir}/{model_name}.keras")
        except Exception as e:
            logging.error(f"Error loading model: {e}. Training from scratch.")

    logging.info("Model summary")

    model.summary()

    model.compile(optimizer=AdamW(learning_rate), loss=BCEDice, metrics=[iou_coeff, dice_coeff])

    callbacks = [
        ModelCheckpoint(f"{save_dir}/{model_name}.keras", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5),
        EarlyStopping(monitor="val_loss", patience=10, verbose=1),
        CSVLogger(csv_dir)
    ]

    logging.info("Training the model")

    try:
        model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=valid_dataset, callbacks=callbacks)
    except Exception as e:
        logging.error(f"Error training the model: {e}")

    logging.info("Model training completed")

if __name__ == "__main__":
    path_dataset = "/mnt/d/CRCT"
    save_dir = "files"
    csv_dir = "files_log"
    epoch = 50
    batch_size = 16
    height = 256
    width = 256
    model = Attention_ResUNet(input_shape=(height, width, 3), class_number=1, dropout_rate=0.0, batch_norm=True)
    (train_x, train_y), (valid_x, valid_y) = load_data_train(path_dataset)
    logging.info(f"Train: ({len(train_x)},{len(train_y)})")
    (test_x, test_y) = load_data_test(path_dataset)
    logging.info(f"Test: ({len(test_x)},{len(test_y)})")

    logging.info("Creating training dataset")
    train_dataset = create_dataset(train_x, train_y, batch_size=batch_size, augment=True, resize=True, normalize=True, size=(height, width))
    logging.info("Creating validation dataset")
    valid_dataset = create_dataset(valid_x, valid_y, batch_size=batch_size, augment=False, resize=True, normalize=True, size=(height, width))

    logging.info("Training the model")
    train_model(model, train_dataset, valid_dataset, save_dir, csv_dir, "attention_resunet", batch_size=batch_size, epochs=epoch, learning_rate=1e-4, class_number=1, input_shape=(height, width, 3), resume=False)
    logging.info("Model training completed")
