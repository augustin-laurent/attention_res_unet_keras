from tensorflow.keras.layers import Flatten
from tensorflow import reduce_sum
from tensorflow.keras.losses import binary_crossentropy

def dice_loss(y_true, y_pred, smooth: float = 1.0):
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = reduce_sum(y_true * y_pred)
    return ((2. * intersection + smooth) / (reduce_sum(y_true) + reduce_sum(y_pred) + smooth))

def dice_coeff(y_true, y_pred, smooth: float = 1.0):
    loss = dice_loss(y_true, y_pred, smooth=smooth)
    return 1 - loss

def bce_dice_loss(y_true, y_pred, smooth: float = 1.0):
    bce = binary_crossentropy(y_true, y_pred)
    loss = bce + dice_coeff(y_true, y_pred, smooth=smooth)
    return loss

def iou_coeff(y_true, y_pred, smooth: float = 1.0):
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = reduce_sum(y_true * y_pred)
    union = reduce_sum(y_true) + reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)