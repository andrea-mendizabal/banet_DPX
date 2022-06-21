import torch
from torch import nn
from scipy.spatial import distance
import numpy as np


class DiceCoefficient:
    """
    Standard DICE coefficient
    """

    def __init__(self, smooth=1.0, epsilon=1e-06, **kwargs):
        self.smooth = 1.0
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true, mask=None):
        assert y_pred.size() == y_true.size()

        if mask is not None:
            y_pred = y_pred * mask
        if not len(y_pred.size()) == 1:
            y_pred = y_pred[:, 0].contiguous().view(-1)
            y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum() + self.smooth
        denominator = (y_pred.sum() + y_true.sum() + self.smooth).clamp(min=self.epsilon)

        dsc = 2. * intersection / denominator
        dsc = dsc.cpu().detach().numpy()
        return float(dsc)


class DiceCoefficientSquared:
    """
    Extension of standard DICE from VNet paper, where denominator has (y_pred^2 + y_true^2).
    """

    def __init__(self, epsilon=1e-06, **kwargs):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true, mask=None):
        assert y_pred.size() == y_true.size()

        if mask is not None:
            y_pred = y_pred * mask

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        denominator = ((y_pred * y_pred).sum() + (y_true * y_true).sum()).clamp(min=self.epsilon)

        dsc = 2. * intersection / denominator
        dsc = dsc.cpu().detach().numpy()
        return float(dsc)


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            raise
            # target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        final_mean_iou = torch.mean(torch.tensor(per_batch_iou)).cpu().detach().numpy()
        return float(final_mean_iou)

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class TPR:
    """
    Computes True Positive Rate, also called Recall or Sensitivity, defined as:
    TP / ( TP + FN )
    """

    def __call__(self, y_pred, y_true, mask=None):
        assert y_pred.size() == y_true.size()

        if mask is not None:
            y_pred = y_pred * mask
        if not len(y_pred.size()) == 1:
            y_pred = y_pred[:, 0].contiguous().view(-1)
            y_true = y_true[:, 0].contiguous().view(-1)

        # In case the prediction tensor has not been rounded btw 0 and 1 yet:
        nonzero = y_pred[y_pred > 0].flatten()
        if len(nonzero) > 0 and not (torch.min(nonzero) == 1.):
            # print(torch.min( y_pred[y_pred > 0] ) )
            y_pred = torch.round(y_pred)

        intersection = (y_pred * y_true).sum().item()
        tpr = intersection / len(torch.nonzero(y_true))

        return float(tpr)


class MHD:
    """
    Computes the Mahalanobis distance between prediction and ground truth.
    Differently from standard MHD, here two sets are compared. Thus, MHD is computed
    between the means of the compared sets and S is the common covariance matrix.

    MHD(X, Y) = sqrt( (mu_x-mu_y)^T * S^(-1) * (mu_x-mu_y) )
    where mu_x and mu_y are the means of the point sets and S is given by
    S = (n1*S1 + n2*S2) / (n1+n2)
    where S1 and s" are the covariance matrices of the voxel sets and n1, n2 are the
    numbers of voxels in each set.
    """

    def __call__(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        if not (len(y_pred.size()) in [3, 5]):
            raise ValueError(f"MHD requires at least 3D data, but your input has {len(y_pred.size())} dimensions")

        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

        # In case the prediction tensor has not been rounded btw 0 and 1 yet:
        nonzero = y_pred[y_pred > 0].flatten()

        if len(nonzero):
            if not (np.min(nonzero) == 1.):
                # print(np.amin( y_pred[y_pred > 0] ))
                y_pred = np.round(y_pred)

            try:
                y_pred = y_pred[0][0]
                y_true = y_true[0][0]
            except:
                pass

            # Voxel coordinates for non zero elements
            x = np.where(y_pred)
            y = np.where(y_true)

            # Means of the distributions
            mu_x = np.mean(x, axis=1)
            mu_y = np.mean(y, axis=1)
            mu = mu_x - mu_y

            s1 = np.cov(x)
            s2 = np.cov(y)
            n1 = np.shape(x)[1]
            n2 = np.shape(y)[1]
            s = (n1 * s1 + n2 * s2) / (n1 + n2)
            s_inv = np.linalg.inv(s)

            # mhd = np.sqrt( np.dot( mu.T, np.dot( s_inv, mu ) ) )
            mhd_sq = np.dot(mu.T, np.dot(s_inv, mu))

            # Alternative implementation which allows to obtain pairwise mhd. The avg value is the same as mhd_sq
            # x_mu = x - mu_y.reshape((-1,1))
            # s2_inv = np.linalg.inv(s2)
            # mhd2 = np.dot(x_mu.T, s2_inv)
            # mhd2 = np.dot(mhd2, x_mu)
            # print(np.mean(mhd2))
            return mhd_sq

        else:
            return np.nan