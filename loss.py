import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    Code borrowed from kaggle segmentation dataset
    """

    def __init__(self, sigmoid_required=False):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

        self.normalization = None
        if sigmoid_required:
            self.normalization = nn.Sigmoid()

    def forward(self, y_pred, y_true, mask=None):
        assert y_pred.size() == y_true.size()
        if self.normalization:
            y_pred = self.normalization(y_pred)
        if mask is not None:
            y_pred = y_pred * mask
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
                y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


class BCE_Dice_Loss(nn.Module):

    def __init__(self, sigmoid_required=False):
        super(BCE_Dice_Loss, self).__init__()
        self.bce = nn.BCELoss(reduction='mean')
        self.dsc = DiceLoss()

    def forward(self, y_pred, y_true):
        self.cross_entropy = self.bce(y_pred, y_true)
        self.dice = self.dsc(y_pred, y_true)
        return (self.cross_entropy + self.dice) / 2.0  # to normalize between 0 and 1


class BCE_Dice_Loss2(nn.Module):
    # Same as BCE_Dice_Loss, with normalization applied internally

    def __init__(self, alpha=1.0, beta=1.0, sigmoid_required=False):
        super(BCE_Dice_Loss2, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.normalization = None
        if sigmoid_required:
            self.normalization = nn.Sigmoid()

        self.bce = nn.BCELoss(reduction='mean')
        self.dsc = DiceLoss(sigmoid_required=False)

    def forward(self, y_pred, y_true, mask=None):
        if self.normalization:
            y_pred = self.normalization(y_pred)
        if mask is not None:
            # Mask must be applied after normalization
            y_pred = y_pred * mask
        self.cross_entropy = self.bce(y_pred, y_true)
        self.dice = self.dsc(y_pred, y_true)
        return self.alpha * self.cross_entropy + self.beta * self.dice


class BCELogits_Dice_Loss(nn.Module):

    def __init__(self, sigmoid_required=True):
        super(BCELogits_Dice_Loss, self).__init__()

        self.bce_logits = nn.BCEWithLogitsLoss()
        self.dsc = DiceLoss(sigmoid_required=sigmoid_required)

    def forward(self, y_pred, y_true, mask=None):
        # Mask should be applied after normalization. This is true for dice,
        # but not for bce with logits, where the mask is applied before normalization

        self.cross_entropy = self.bce_logits(y_pred, y_true)
        self.dice = self.dsc(y_pred, y_true, mask=mask)
        return self.cross_entropy + self.dice


class Generalized_Dice_Loss(nn.Module):

    def __init__(self, sigmoid_required=True, epsilon=1e-6):
        super(Generalized_Dice_Loss, self).__init__()
        self.epsilon = epsilon
        self.normalization = None
        if sigmoid_required:
            self.normalization = nn.Sigmoid()

    def forward(self, y_pred, y_true, mask=None):
        assert y_pred.size() == y_true.size(), "'y_pred' and 'y_true' must have the same shape"

        if self.normalization:
            y_pred = self.normalization(y_pred)
        if mask is not None:
            y_pred = y_pred * mask

        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)
        y_true = y_true.float()

        if y_pred.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            y_pred = torch.cat((y_pred, 1 - y_pred), dim=0)
            y_true = torch.cat((y_true, 1 - y_true), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = y_true.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (y_pred * y_true).sum(-1)
        intersect = intersect * w_l

        denominator = (y_pred + y_true).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        dsc = 2 * (intersect.sum() / denominator.sum())

        return 1. - torch.mean(dsc)