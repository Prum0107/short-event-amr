def has_overlap(a_st, a_ed, b_st, b_ed):
    return not (a_ed <= b_st or b_ed <= a_st)


def intersection_1d(a_st, a_ed, b_st, b_ed):
    if not has_overlap(a_st, a_ed, b_st, b_ed):
        return 0.0
    return max(0.0, min(a_ed, b_ed) - max(a_st, b_st))


def iou_1d(a_st, a_ed, b_st, b_ed):
    inter = intersection_1d(a_st, a_ed, b_st, b_ed)
    if inter <= 0:
        return 0.0
    union = (a_ed - a_st) + (b_ed - b_st) - inter
    return inter / max(union, 1e-6)


def best_iou_for_window(pred_window, gt_windows):
    pred_st, pred_ed = pred_window[0], pred_window[1]
    best = 0.0
    for gt_st, gt_ed in gt_windows:
        best = max(best, iou_1d(pred_st, pred_ed, gt_st, gt_ed))
    return best


def recall_at_1_iou(pred_windows, gt_windows, threshold=0.5):
    if not pred_windows:
        return 0
    return int(best_iou_for_window(pred_windows[0], gt_windows) >= threshold)


def recall_at_k_iou(pred_windows, gt_windows, threshold=0.5, k=3):
    if not pred_windows:
        return 0
    for pred in pred_windows[:k]:
        if best_iou_for_window(pred, gt_windows) >= threshold:
            return 1
    return 0


def best_iou_among_topk(pred_windows, gt_windows, k=5):
    if not pred_windows:
        return 0.0
    best = 0.0
    for pred in pred_windows[:k]:
        best = max(best, best_iou_for_window(pred, gt_windows))
    return best


def compute_region_coverage(pred_windows, gt_windows):
    for pred in pred_windows:
        for gt in gt_windows:
            if iou_1d(pred[0], pred[1], gt[0], gt[1]) > 0:
                return 1
    return 0
