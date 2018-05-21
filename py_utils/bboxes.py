import numpy as np

def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index (IOU) between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    if isinstance(bboxes1, (tuple, list)):
        bboxes1 = np.array(bboxes1)
    if isinstance(bboxes2, (tuple, list)):
        bboxes2 = np.array(bboxes2)

    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def bboxes_intersection(bboxes_ref, bboxes2):
    """Computing  interset(bboxes2, bboxes_ref) / area(bboxex_ref).
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.

    """
    # if isinstance(bboxes_ref, (list, tuple)):
    #     bboxes_ref = np.array(bboxes_ref)
    # if isinstance(bboxes2, (list, tuple)):
    #     bboxes2 = np.array(bboxes2)
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score


def bboxes_nms_multiclass(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


def bboxes_nms(scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    if isinstance(scores, list):
        scores = np.array(scores)

    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)

    keep_bboxes = np.ones(len(scores), dtype=np.bool)
    for i in range(len(scores)-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = (overlap < nms_threshold)
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return scores[idxes].tolist(), bboxes[idxes].tolist(), idxes


def self_jaccard(s_bbox, bboxes, bboxes_areas=None):
    """Compute jaccard between score a box and the anchors.
    """
    returnList = False
    if isinstance(s_bbox, (tuple, list)):
        s_bbox = np.array(s_bbox)
    if isinstance(bboxes, (tuple, list)):
        bboxes = np.array(bboxes)
        returnList = True
    if bboxes_areas is None:
        bboxes_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    int_ymin = np.maximum(bboxes[:, 1], s_bbox[1])
    int_xmin = np.maximum(bboxes[:, 0], s_bbox[0])
    int_ymax = np.minimum(bboxes[:, 3], s_bbox[3])
    int_xmax = np.minimum(bboxes[:, 2], s_bbox[2])
    h = np.maximum(int_ymax - int_ymin, 0.)
    w = np.maximum(int_xmax - int_xmin, 0.)
    inter_vol = h * w
    union_vol = bboxes_areas - inter_vol + (s_bbox[2] - s_bbox[0]) * (s_bbox[3] - s_bbox[1])
    scores = np.divide(inter_vol, union_vol)
    if returnList:
        scores = scores.tolist()
    return scores


def get_vols_xyxy(bboxes):
    return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])