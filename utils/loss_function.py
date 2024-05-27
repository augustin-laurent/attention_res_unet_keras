from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Flatten
from tensorflow import reduce_sum

def dice_loss(y_true, y_pred, smooth: float = 1.0):
    """
    Computes the Dice loss between the true and predicted values.

    The Dice loss is a measure of the overlap between two samples and is often used in image segmentation tasks. 
    A Dice loss of 0 signifies no overlap while a loss of 1 signifies perfect overlap.

    Parameters:
    y_true (Tensor): The ground truth values. 
    y_pred (Tensor): The predicted values.
    smooth (float, optional): A smoothing factor to avoid division by zero and to control the trade-off between precision and recall. Defaults to 1.0.

    Returns:
    float: The computed Dice loss.
    """
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = reduce_sum(y_true * y_pred)
    return ((2. * intersection + smooth) / (reduce_sum(y_true) + reduce_sum(y_pred) + smooth))

def dice_coeff(y_true, y_pred, smooth: float = 1.0):
    """
    Computes the Dice coefficient between the true and predicted values.

    The Dice coefficient is a measure of the overlap between two samples and is often used in image segmentation tasks. 
    A Dice coefficient of 0 signifies no overlap while a coefficient of 1 signifies perfect overlap.

    Parameters:
    y_true (Tensor): The ground truth values. 
    y_pred (Tensor): The predicted values.
    smooth (float, optional): A smoothing factor to avoid division by zero and to control the trade-off between precision and recall. Defaults to 1.0.

    Returns:
    float: The computed Dice coefficient.
    """
    loss = dice_loss(y_true, y_pred, smooth=smooth)
    return 1 - loss

def bce_dice_loss(y_true, y_pred, smooth: float = 1.0):
    """
    Computes the combined Binary Cross-Entropy (BCE) and Dice loss between the true and predicted values.

    This loss function is often used in image segmentation tasks where both the overlap between the predicted and true values (Dice loss) and the correctness of each individual prediction (BCE) are important.

    Parameters:
    y_true (Tensor): The ground truth values. 
    y_pred (Tensor): The predicted values.
    smooth (float, optional): A smoothing factor to avoid division by zero and to control the trade-off between precision and recall in the Dice loss calculation. Defaults to 1.0.

    Returns:
    float: The computed combined BCE and Dice loss.
    """
    bce = binary_crossentropy(y_true, y_pred)
    loss = bce + dice_coeff(y_true, y_pred, smooth=smooth)
    return loss

def iou_coeff(y_true, y_pred, smooth: float = 1.0):
    """
    Computes the Intersection over Union (IoU) coefficient between the true and predicted values.

    The IoU coefficient is a measure of the overlap between two samples and is often used in image segmentation tasks. 
    An IoU coefficient of 0 signifies no overlap while a coefficient of 1 signifies perfect overlap.

    Parameters:
    y_true (Tensor): The ground truth values. 
    y_pred (Tensor): The predicted values.
    smooth (float, optional): A smoothing factor to avoid division by zero. Defaults to 1.0.

    Returns:
    float: The computed IoU coefficient.
    """
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = reduce_sum(y_true * y_pred)
    union = reduce_sum(y_true) + reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)