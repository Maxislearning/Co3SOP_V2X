import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def focal_loss(pred, target, alpha=0.25, gamma=2.0, ignore_index=255):
    """Binary focal loss for a dense per-voxel target/non-target mask.
    pred: raw logits [B, 2, ...] (2 = non-target/target). target: same
    spatial shape as pred minus the class dim, values in {0, 1, ignore_index}.
    Plain BCE would let an all-empty prediction get a deceptively low loss
    since target voxels are a tiny fraction of the grid -- focal down-weights
    the easy (empty) majority instead.
    """
    mask = target != ignore_index
    target = target[mask].long()
    logits = pred.permute(0, 2, 3, 4, 1)[mask]  # [N, 2]
    ce = F.cross_entropy(logits, target, reduction='none')
    pt = torch.exp(-ce)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce).mean()


def dice_loss(pred, target, ignore_index=255, eps=1.0):
    """Binary Dice loss on the target-class softmax probability."""
    mask = target != ignore_index
    target = target[mask].float()
    prob = F.softmax(pred, dim=1)[:, 1][mask]  # P(target)
    intersection = (prob * target).sum()
    union = prob.sum() + target.sum()
    return 1 - (2 * intersection + eps) / (union + eps)

def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''

    bs = gt_occ.shape[0]
    gt_pts = []
    for i in range(bs):
        non_zeros = torch.nonzero(gt_occ[i])
        values = gt_occ[i][non_zeros[:,0],non_zeros[:,1],non_zeros[:,2]]
        pts = torch.cat([
            non_zeros,
            values.unsqueeze(1)
        ], dim=1)
        gt_pts.append(pts.float())
    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).type(torch.float) 
    for i in range(gt.shape[0]):
        coords = gt_pts[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_pts[i][:, 3]
    
    return gt

def geo_scal_loss(pred, ssc_target, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    # print(nonempty_target.sum())
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy_with_logits(precision, torch.ones_like(precision))
        + F.binary_cross_entropy_with_logits(recall, torch.ones_like(recall))
        + F.binary_cross_entropy_with_logits(spec, torch.ones_like(spec))
    )

def sem_scal_loss_with_weights(pred, ssc_target, weights=None):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy_with_logits(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy_with_logits(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy_with_logits(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            if weights!= None:
                loss_class = weights[i] * loss_class
            loss += loss_class
    return loss / count

def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy_with_logits(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy_with_logits(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy_with_logits(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count

