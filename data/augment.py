import numpy as np

import albumentations as A

def augment_data(image: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Augments the input image and mask using a variety of transformations.

    The transformations include horizontal and vertical flips, elastic and perspective transformations, piecewise affine transformations, shift-scale-rotate transformations, grid and optical distortions, blur, rotation, and transpose. Each transformation is applied with a certain probability.

    Parameters:
    image (np.ndarray): The input image to be augmented.
    mask (np.ndarray): The corresponding mask to be augmented.

    Returns:
    tuple: The augmented image and mask as a tuple.
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([A.ElasticTransform(),
                 A.Perspective(scale=(0.05, 0.1)),
                 #A.PiecewiseAffine(scale=(0.03, 0.05)),
                 A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.4, rotate_limit=45)],
                p=0.5),
        A.OneOf([A.GridDistortion(),
                 A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5)],
                p=0.5),
        A.Blur(blur_limit=3, p=1),
        A.Rotate(limit=40, p=0.5),
        A.Transpose(p=0.5)
    ])

    augmented = transform(image=image, mask=mask)

    return augmented['image'], augmented['mask']