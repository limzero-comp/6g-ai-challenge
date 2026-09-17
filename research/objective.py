"""BCE warmup and optional smooth bit-score/lower-tail optimization."""
import math

import torch
import torch.nn.functional as F

from .link import B_MAX, bit_mask


def objective(bits, logits, lengths, score_weight=0.0, tail_weight=0.0, temperature=1.0):
    if not 0 <= score_weight <= 1 or tail_weight < 0 or temperature <= 0:
        raise ValueError("Invalid objective settings")
    bces, scores = [], []
    for target, pred, count in zip(bits, logits, lengths):
        target = target[:, :pred.shape[1]]
        mask = bit_mask(count, pred.shape[1]).to(pred.dtype)
        # Center the no-information loss at zero; source lengths are currently
        # exogenous, so this constant does not alter any model gradient.
        bce = ((F.binary_cross_entropy_with_logits(pred, target, reduction="none") - math.log(2.0)) * mask).sum(1) / B_MAX
        soft_correct = torch.sigmoid((2 * target - 1) * pred / temperature)
        soft_score = 0.5 + ((soft_correct - 0.5) * mask).sum(1) / B_MAX
        bces.append(bce)
        scores.append(soft_score)
    bce = torch.cat(bces).mean()
    score = torch.cat(scores)
    # CVaR is only a training surrogate. Select checkpoints using the actual p10.
    tail_count = max(1, math.ceil(0.1 * score.numel()))
    tail = torch.topk(score, tail_count, largest=False).values.mean()
    loss = (1 - score_weight) * bce + score_weight * (1 - score.mean()) + tail_weight * (1 - tail)
    return loss, {"bce": float(bce.detach()), "soft_score": float(score.mean().detach()),
                  "soft_tail": float(tail.detach())}
