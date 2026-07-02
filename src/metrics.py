import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import f1_score, roc_auc_score


def iou_score(y_true, y_pred, smooth=1e-7):
    """
    IoU для бинарной сегментации.
    y_true и y_pred: маски 0/1.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    return (intersection + smooth) / (union + smooth)


def dice_score(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient для сегментации.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()

    return (2 * intersection + smooth) / (total + smooth)


def hausdorff_distance(y_true, y_pred):
    """
    Расстояние Хаусдорфа между границами/масками.
    Чем меньше, тем лучше.
    """
    true_points = np.argwhere(y_true > 0)
    pred_points = np.argwhere(y_pred > 0)

    if len(true_points) == 0 or len(pred_points) == 0:
        return float("inf")

    forward = directed_hausdorff(true_points, pred_points)[0]
    backward = directed_hausdorff(pred_points, true_points)[0]

    return max(forward, backward)


def classification_f1(y_true, y_pred):
    """
    F1 для классификации.
    """
    return f1_score(y_true, y_pred, average="weighted")


def classification_auc(y_true, y_score):
    """
    AUC для бинарной классификации.
    y_score — вероятности класса 1.
    """
    return roc_auc_score(y_true, y_score)