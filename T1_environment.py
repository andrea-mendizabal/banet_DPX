"""
#01 - Implementing an Environment
BanetEnvironment: simply set the current step index as training data.
"""

# Python related imports
from numpy import array
import numpy as np
from vtk import *
import os
import torch

# Session related imports
from loss import BCELogits_Dice_Loss
from eval_metrics import DiceCoefficient, DiceCoefficientSquared, MeanIoU, TPR, MHD

# DeepPhysX related imports
from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment


# Create an Environment as a BaseEnvironment child class
class BanetEnvironment(BaseEnvironment):

    def __init__(self,
                 ip_address='localhost',
                 port=10000,
                 instance_id=1, 
                 number_of_instances=1, 
                 as_tcp_ip_client=True,  
                 environment_manager=None):

        BaseEnvironment.__init__(self, 
                                 ip_address=ip_address, 
                                 port=port,
                                 instance_id=instance_id, 
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client, 
                                 environment_manager=environment_manager)

        self.nb_step = 0
        self.increment = 0
        # self.gridResolution = [27, 27, 32]
        self.gridResolution = [31, 32, 26]
        self.size = [0.29056687112897633, 0.2999399960041046, 0.243701246753335]
        # self.size = [0.25312511567026375, 0.25312511567026375, 0.30000013709068296] # should be the length of the box and not its resolution
        self.nb_points_in_grid = self.gridResolution[0]*self.gridResolution[1]*self.gridResolution[2]  # 23328
        self.nb_channels_in = 4

        # Evaluation metrics
        self.bce_loss_total = []
        self.pred_time      = []
        self.dice_test      = []
        self.dicesq_test    = []
        self.miou_test      = []
        self.inters_test    = []
        self.mhd_test       = []

    """
    INITIALIZING ENVIRONMENT - Methods will be automatically called it this order:
       - recv_parameters: Receive a dictionary of parameters that can be set in EnvironmentConfig
       - create: Create the Environment
       - init: Initialize the Environment if required
       - send_parameters: Same as recv_parameters, Environment can send back a set of parameters if required
       - send_visualization: Send initial visualization data (see Example/CORE/Features to add visualization data)
    """

    # Optional
    def recv_parameters(self, param_dict):
        # Set data size
        self.increment = param_dict['increment'] if 'increment' in param_dict else 1

    # MANDATORY
    def create(self):
        # Nothing to create in our BanetEnvironment
        pass

    # Optional
    def init(self):
        # Nothing to init in our BanetEnvironment
        pass

    # Optional
    def send_parameters(self):
        # Nothing to send back
        return {}

    # Optional
    def send_visualization(self):
        # Nothing to visualize (see Example/CORE/Features to add visualization data)
        return {}

    """
    ENVIRONMENT BEHAVIOR - Methods will be automatically called at each simulation step in this order:
       - step: Transition in simulation state, compute training data
       - check_sample: Check if current data sample is usable
       - apply_prediction: Network prediction will be applied in Environment
       - close: Shutdown procedure when data producer is no longer used
     Some requests can be performed here:
       - get_prediction: Get an online prediction from an input array
       - update_visualization: Send updated visualization data (see Example/CORE/Features to update viewer data)
    """

    # MANDATORY
    async def step(self):
        # ONLY CALLED IN PREDICTION ;)
        # Sending training data read from dataset
        self.set_training_data(self.sample_in, self.sample_out)

        # Store prediction
        prediction = self.get_prediction(self.sample_in)

        # Apply sigmoid to prediction to have values between 0 and 1
        prediction = torch.sigmoid(torch.tensor(prediction))

        # Binarize the prediction : values bigger than a threshold are set to one, others to zero
        # for i in range(prediction.shape[0]):
        #     if prediction[i] > 0.5:
        #         prediction[i] = 1
        #     else:
        #         prediction[i] = 0
        self.store_prediction_to_vts(prediction)
        # self.set_additional_dataset('prediction', prediction)
        self.compute_metrics(prediction)

        self.nb_step += self.increment

    def store_prediction_to_vts(self, prediction):
        # Create vtk grid
        grid = vtkStructuredGrid()
        grid.SetDimensions((self.gridResolution[0], self.gridResolution[1], self.gridResolution[2]))
        points = vtkPoints()
        points.SetNumberOfPoints(self.nb_points_in_grid)
        pID = 0
        start_x = -self.size[0] / 2
        start_y = -self.size[1] / 2
        start_z = -self.size[2] / 2
        d_x = self.size[0] / (self.gridResolution[0] - 1)
        d_y = self.size[1] / (self.gridResolution[1] - 1)
        d_z = self.size[2] / (self.gridResolution[2] - 1)
        for k in range(0, self.gridResolution[2]):
            for j in range(0, self.gridResolution[1]):
                for i in range(0, self.gridResolution[0]):
                    x = start_x + d_x * i
                    y = start_y + d_y * j
                    z = start_z + d_z * k
                    points.SetPoint(pID, x, y, z)
                    pID += 1
        grid.SetPoints(points)

        # Save the input to a vts file
        if self.nb_channels_in == 4:
            # Displacement of visible nodes
            displacement = vtkFloatArray()
            displacement.SetName("displacement")
            displacement.SetNumberOfComponents(3)
            displacement.SetNumberOfTuples(self.nb_points_in_grid)
            for i in range(self.nb_points_in_grid):
                displacement.SetTuple3(i, self.sample_in[i][0], self.sample_in[i][1], self.sample_in[i][2])
            grid.GetPointData().AddArray(displacement)

            # Preoperative surface
            preopSurf = vtkFloatArray()
            preopSurf.SetName("preopSurf")
            preopSurf.SetNumberOfComponents(1)
            preopSurf.SetNumberOfTuples(self.nb_points_in_grid)
            for i in range(self.nb_points_in_grid):
                preopSurf.SetTuple1(i, self.sample_in[i][3])
            grid.GetPointData().AddArray(preopSurf)

        else:
            # Multiple or single number of frames
            for j in range(self.nb_channels_in):
                # Intraoperative surfaces
                intraopSurf = vtkFloatArray()
                intraopSurf.SetName("intraopSurf" + str(j))
                intraopSurf.SetNumberOfComponents(1)
                intraopSurf.SetNumberOfTuples(self.nb_points_in_grid)
                for i in range(self.nb_points_in_grid):
                    intraopSurf.SetTuple1(i, self.sample_in[i][0 + j])
                grid.GetPointData().AddArray(intraopSurf)


        # Save the prediction to a vts file
        stiffness = vtkFloatArray()
        stiffness.SetName("stiffness")
        stiffness.SetNumberOfComponents(1)
        stiffness.SetNumberOfTuples(self.nb_points_in_grid)

        for i in range(self.nb_points_in_grid):
            stiffness.SetTuple1(i, prediction[i])
        grid.GetPointData().AddArray(stiffness)

        # Write grid to vts file
        filename = 'input_and_prediction_' + str(self.nb_step) + '.vts'
        filepath = os.path.join(os.getcwd() + '/predictions/', filename)
        print("Writing to {}".format(filepath))
        writer = vtkXMLStructuredGridWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(grid)
        writer.Update()

    def compute_metrics(self, pred):
        gt = torch.tensor(self.sample_out)
        pred = torch.tensor(pred).reshape(gt.shape)
        # gt = torch.sigmoid(gt)

        dsc_bce_loss = BCELogits_Dice_Loss()
        dice = DiceCoefficient()
        dicesq = DiceCoefficientSquared()
        # miou = MeanIoU()
        tpr = TPR()
        # mhd = MHD()

        loss = dsc_bce_loss(pred, gt)
        self.bce_loss_total.append(loss.item())

        # pred = torch.sigmoid(pred)

        self.dice_test.append(dice(pred, gt))
        self.dicesq_test.append(dicesq(pred, gt))
        # self.miou_test.append(miou(pred, gt))
        self.inters_test.append(tpr(pred, gt))
        # self.mhd_test.append(mhd(pred, gt))

        print("STATS OVER {} SAMPLES: ".format(self.nb_step))
        print(f"Final loss on test dataset: {np.mean(self.bce_loss_total):.8f} +- Std dev: {np.std(self.bce_loss_total):.8f} (Max error: {np.amax(self.bce_loss_total):.8f})")
        print(f"Final DICE on test dataset: {np.mean(self.dice_test):.8f} +- Std dev: {np.std(self.dice_test):.8f}")
        print(f"Final avg DICESQ on test dataset: {np.mean(self.dicesq_test):.8f}")
        # print(f"Final avg MIoU on test dataset: {np.mean(self.miou_test):.8f}")
        print(f'Final avg intersection on test dataset: {np.mean(self.inters_test) * 100:.1f}')
        # print(f"Final avg MHD on test dataset: {np.mean(self.mhd_test):.8f}")

    # Optional
    def check_sample(self, check_input=True, check_output=True):
        # Nothing to check in our BanetEnvironment
        return True

    # Optional
    def apply_prediction(self, prediction):
        # Nothing to apply in our BanetEnvironment
        print(f"Prediction at step {self.nb_step - 1} = {prediction}")

    # Optional
    def close(self):
        # Shutdown procedure
        print("Bye!")
