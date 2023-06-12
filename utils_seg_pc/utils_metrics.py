import numpy as np


import torch

def mean_iou(cf_mtx):
    """
    cf_mtx(ndarray): shape -> (class_num, class_num), 混淆矩阵
    """
    #
    mIous = np.diag(cf_mtx) / (np.sum(cf_mtx, axis=1) + \
                              np.sum(cf_mtx, axis=0) - np.diag(cf_mtx))

    # 所有类别iou取平均
    mIou = np.nanmean(mIous)
    return mIous, mIou


def get_transform_label_preds(predictions, labels):
    # predictions_new = torch.flatten(predictions, 0, 1)
    predictions_new = predictions.argmax(1).cpu().numpy()
    labels_new = torch.flatten(labels, 0, 1).cpu().numpy()
    return predictions_new, labels_new

