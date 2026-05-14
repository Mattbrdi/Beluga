import torch


def _as_bool_mask(mask):
    return torch.as_tensor(mask).bool()


def _duration_from_count(sample_count, sample_rate):
    sample_rate = torch.as_tensor(
        sample_rate,
        device=sample_count.device,
        dtype=torch.float32,
    )

    sample_count = sample_count.float()
    if sample_rate.ndim == 0 or sample_count.ndim == 0:
        return sample_count.sum() / sample_rate.float().mean()

    batch_size = sample_rate.numel()
    if sample_count.numel() == batch_size:
        return (sample_count.reshape(batch_size) / sample_rate.reshape(batch_size)).sum()

    return sample_count.sum() / sample_rate.float().mean()


def compute_iou(pred, target, eps=1e-8):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target).to(pred.device)

    intersection = torch.logical_and(pred, target).sum().float()
    union = torch.logical_or(pred, target).sum().float()

    if union == 0:
        return torch.ones((), device=pred.device)

    return intersection / (union + eps)


def compute_fp(pred, target, sample_rate):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target).to(pred.device)

    false_positive = torch.logical_and(pred, torch.logical_not(target))

    if false_positive.ndim <= 1:
        sample_count = false_positive.sum()
    else:
        sample_count = false_positive.flatten(start_dim=1).sum(dim=1)

    return _duration_from_count(sample_count, sample_rate)


def compute_mi(pred, target, sample_rate):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target).to(pred.device)

    missed_impulse = torch.logical_and(torch.logical_not(pred), target)

    if missed_impulse.ndim <= 1:
        sample_count = missed_impulse.sum()
    else:
        sample_count = missed_impulse.flatten(start_dim=1).sum(dim=1)

    return _duration_from_count(sample_count, sample_rate)


def compute_precision(pred, target, eps=1e-8):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target).to(pred.device)

    true_positive = torch.logical_and(pred, target).sum().float()
    predicted_positive = pred.sum().float()

    if predicted_positive == 0:
        return torch.ones((), device=pred.device) if target.sum() == 0 else torch.zeros((), device=pred.device)

    return true_positive / (predicted_positive + eps)


def compute_recall(pred, target, eps=1e-8):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target).to(pred.device)

    true_positive = torch.logical_and(pred, target).sum().float()
    actual_positive = target.sum().float()

    if actual_positive == 0:
        return torch.ones((), device=pred.device)

    return true_positive / (actual_positive + eps)


def compute_f1(pred, target, eps=1e-8):
    precision = compute_precision(pred, target, eps=eps)
    recall = compute_recall(pred, target, eps=eps)

    if precision + recall == 0:
        return torch.zeros((), device=precision.device)

    return 2 * precision * recall / (precision + recall + eps)


def compute_metrics(pred, target, sample_rate):
    return {
        "test_precision": compute_precision(pred, target),
        "test_recall": compute_recall(pred, target),
        "test_f1": compute_f1(pred, target),
        "test_iou": compute_iou(pred, target),
        "test_fp_duration": compute_fp(pred, target, sample_rate),
        "test_mi_duration": compute_mi(pred, target, sample_rate),
    }
