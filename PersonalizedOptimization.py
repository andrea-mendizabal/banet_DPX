import torch
import numpy as np
import vtk
from vtk.util import numpy_support

from DeepPhysX_PyTorch.Network.TorchOptimization import TorchOptimization


class PersonalizedOptimization(TorchOptimization):

    def __init__(self, config):
        TorchOptimization.__init__(self, config)
        self.build_mask(filename='data/liver/voxelized_displacement.vts',
                        gridResolution=[31, 32, 26],
                        size=[0.29056687112897633, 0.2999399960041046, 0.243701246753335])

    def build_mask(self, filename='', gridResolution=[32, 32, 32], size=[0.3, 0.3, 0.3], device='cuda'):
        # Read sample
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()

        voxelized = reader.GetOutput()
        data = voxelized.GetPointData()

        preoperativeRaw = numpy_support.vtk_to_numpy(data.GetArray("preoperativeSurface"))
        preoperative_sdf = np.reshape(preoperativeRaw, (gridResolution[2], gridResolution[1], gridResolution[0], 1))
        # preoperative_sdf = np.transpose(preoperative_sdf, (3, 0, 1, 2))

        # We assume that the grid is perfectly regular
        max_mask_value = (size[0] / gridResolution[0]) * np.sqrt(3)
        self.mask = torch.from_numpy(preoperative_sdf <= max_mask_value).to(device)
        if not self.mask.any():
            raise IOError("Sample {} contains no internal points (no valid signed distance function?)".format(filename))

    def compute_loss(self, prediction, ground_truth, data):
        self.loss_value = self.loss(prediction, ground_truth, self.mask)
        return self.transform_loss(data)
