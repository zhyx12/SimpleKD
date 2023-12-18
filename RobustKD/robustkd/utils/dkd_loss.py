# from https://github.com/megvii-research/mdistiller/blob/d3f5e9946d55443a7ff2c65b4b1966bc17efbf27/mdistiller/distillers/DKD.py
# MIT License
import torch
import torch.nn as nn
import torch.nn.functional as F


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature=1.0, fusion_type=None,
             prob_delta=0.0, add_extra_mask=False, num_extra_mask_class=1):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    if isinstance(logits_teacher, torch.Tensor):
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    else:
        if fusion_type == 'mean_logit':
            mean_logits = (logits_teacher[0] + logits_teacher[1]) / 2.0
            pred_teacher = F.softmax(mean_logits / temperature, dim=1)
        elif fusion_type == 'max_logit':
            mean_logits = torch.maximum(logits_teacher[0], logits_teacher[1])
            pred_teacher = F.softmax(mean_logits / temperature, dim=1)
        elif fusion_type == 'max_prob':
            prob_1 = F.softmax(logits_teacher[0] / temperature, dim=1)
            prob_2 = F.softmax(logits_teacher[1] / temperature, dim=1)
            pred_teacher = torch.maximum(prob_1, prob_2)
            pred_teacher /= pred_teacher.sum(dim=1, keepdims=True)
        else:
            pred_teacher = None
    #
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student + prob_delta)
    # tckd_loss = torch.mean(
    #     F.kl_div(log_pred_student, pred_teacher + prob_delta, reduction='none').sum(-1))
    tckd_loss = -torch.mean((log_pred_student * pred_teacher).sum(-1))
    if add_extra_mask:
        extra_mask = _add_extra_mask(logits_student, num_extra_mask_class)
        gt_mask = gt_mask * extra_mask
    if isinstance(logits_teacher, torch.Tensor):
        pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    else:
        if fusion_type == 'mean_logit':
            mean_logits = (logits_teacher[0] + logits_teacher[1]) / 2.0
            pred_teacher_part2 = F.softmax(mean_logits / temperature - 1000.0 * gt_mask, dim=1)
        elif fusion_type == 'max_logit':
            mean_logits = torch.maximum(logits_teacher[0], logits_teacher[1])
            pred_teacher_part2 = F.softmax(mean_logits / temperature - 1000.0 * gt_mask, dim=1)
        elif fusion_type == 'max_prob':
            prob_part2_1 = F.softmax(logits_teacher[0] / temperature - 1000.0 * gt_mask, dim=1)
            prob_part2_2 = F.softmax(logits_teacher[1] / temperature - 1000.0 * gt_mask, dim=1)
            pred_teacher_part2 = torch.maximum(prob_part2_1, prob_part2_2)
            pred_teacher_part2 /= pred_teacher_part2.sum(dim=1, keepdims=True)
        else:
            pred_teacher_part2 = None
    log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)
    # nckd_loss = torch.mean(F.kl_div(log_pred_student_part2, pred_teacher_part2 + prob_delta, reduction='none').sum(-1))
    nckd_loss = -torch.mean((log_pred_student_part2 * pred_teacher_part2).sum(-1))
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def _add_extra_mask(logits, num_extra_mask_class):
    with torch.no_grad():
        class_ind = torch.randperm(logits.shape[1], device='cuda:0')[0:logits.shape[0]]
        tmp_mask = _get_gt_mask(logits, class_ind)
        for i in range(1, num_extra_mask_class):
            class_ind = torch.randperm(logits.shape[1], device='cuda:0')[0:logits.shape[0]]
            tmp_mask *= _get_gt_mask(logits, class_ind)
        return tmp_mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def prob_product_kd(logits_student, logits_teacher, target,temperature=1.0,):
    gt_mask = _get_gt_mask(logits_student, target)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    pred_student_part2 = F.softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)
    loss = torch.mean((pred_teacher_part2 * pred_student_part2).sum(-1))
    return loss