import os
import numpy as np
import logging
import cv2

from glob import glob
from tqdm import tqdm

from .augment import augment_data

from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

SEED = 88

IMG_FORMAT = ["png", "jpg", "jpeg"]

logging.basicConfig(level=logging.INFO)

def read_img(path: str, resize: bool = False, normalize: bool = False, size: tuple = (256, 256)) -> np.ndarray:
    """
    Reads an image from a given path and optionally resizes and normalizes it.

    Parameters:
    path (str): The path to the image file.
    resize (bool, optional): If True, the image is resized to the given size. Defaults to False.
    normalize (bool, optional): If True, the image is normalized to have values between 0 and 1. Defaults to False.
    size (tuple, optional): The desired size of the image after resizing. Defaults to (256, 256).

    Returns:
    np.ndarray: The processed image as a numpy array.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if resize:
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    if normalize:
        img = img / 255.0
    img = img.astype(np.float32)
    return img

def read_mask(path: str, resize: bool = False, normalize: bool = False, size: tuple = (256, 256)) -> np.ndarray:
    """
    Reads a mask from a given path and optionally resizes and normalizes it.

    Parameters:
    path (str): The path to the mask file.
    resize (bool, optional): If True, the mask is resized to the given size. Defaults to False.
    normalize (bool, optional): If True, the mask is normalized to have values between 0 and 1. Defaults to False.
    size (tuple, optional): The desired size of the mask after resizing. Defaults to (256, 256).

    Returns:
    np.ndarray: The processed mask as a numpy array.
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if resize:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_CUBIC)
    if normalize:
        mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def load_data_train(path: str, num_slice_valid_set: int = 1, seed: int = SEED) -> tuple:
    """
    Loads the dataset from the given path and splits it into training and validation sets.

    Parameters:
    path (str): The path to the dataset.
    num_slice_valid_set (int, optional): The number of slices to use for the validation set. Defaults to 1.
    seed (int, optional): The seed for the random number generator. Defaults to SEED.

    Returns:
    tuple: A tuple containing the training and validation sets.
    """
    images = []
    masks = []
    for fmt in tqdm(IMG_FORMAT, desc="Loading images for training and validation sets"):
        images.extend(sorted(glob(os.path.join(path, f"train/HE_cell/*.{fmt}"))))
        masks.extend(sorted(glob(os.path.join(path, f"train/ERG_cell/*.{fmt}"))))
    logging.info(f"Found {len(images)} images and {len(masks)} masks for training and validation sets")
    
    ids = [img.split('_')[0] for img in images]
    unique_ids = list(set(ids))

    np.random.seed(seed)
    np.random.shuffle(unique_ids)
    valid_ids = unique_ids[:num_slice_valid_set]
    train_ids = unique_ids[num_slice_valid_set:]

    train_x = [img for img, id in zip(images, ids) if id in train_ids]
    train_y = [mask for mask, id in zip(masks, ids) if id in train_ids]
    valid_x = [img for img, id in zip(images, ids) if id in valid_ids]
    valid_y = [mask for mask, id in zip(masks, ids) if id in valid_ids]

    logging.info(f"Train: ({len(train_x)},{len(train_y)})")
    logging.info(f"Valid: ({len(valid_x)},{len(valid_y)})")

    return (train_x, train_y), (valid_x, valid_y)

def load_data_test(path: str) -> tuple:
    """
    Loads the test dataset from the given path.

    Parameters:
    path (str): The path to the dataset.

    Returns:
    tuple: A tuple containing the test images and masks.
    """
    images = []
    masks = []
    for fmt in tqdm(IMG_FORMAT, desc="Loading images for test dataset"):
        images.extend(sorted(glob(os.path.join(path, f"eval/HE_eval/*.{fmt}"))))
        masks.extend(sorted(glob(os.path.join(path, f"eval/ERG_eval/*.{fmt}"))))
    logging.info(f"Found {len(images)} images and {len(masks)} masks for testing")
    return images, masks

def create_dataset(images: list, masks: list, batch_size: int = 16, augment: bool = True, resize: bool = True, normalize: bool = True, size: tuple = (256, 256)) -> Dataset:
    """
    Creates a TensorFlow dataset from the given images and masks.

    Parameters:
    images (list): A list of paths to the images.
    masks (list): A list of paths to the masks.
    batch_size (int, optional): The batch size for the dataset. Defaults to 16.
    augment (bool, optional): If True, the dataset is augmented. Defaults to False.
    resize (bool, optional): If True, the images and masks are resized. Defaults to True.
    normalize (bool, optional): If True, the images and masks are normalized. Defaults to True.
    size (tuple, optional): The desired size of the images and masks after resizing. Defaults to (256, 256).

    Returns:
    Dataset: The TensorFlow dataset.
    """
    X = []
    Y = []

    for x, y in zip(images, masks):
        img = read_img(x, resize=resize, normalize=normalize, size=size)
        mask = read_mask(y, resize=resize, normalize=normalize, size=size)
        if augment:
            img_aug, mask_aug = augment_data(img, mask)
            X.append(img_aug)
            Y.append(mask_aug)
        X.append(img)
        Y.append(mask)
    
    X = np.array(X)
    Y = np.array(Y)
    
    dataset = Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset