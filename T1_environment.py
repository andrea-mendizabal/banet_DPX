"""
#01 - Implementing an Environment
BanetEnvironment: simply set the current step index as training data.
"""

# Python related imports
from numpy import array
import numpy as np
from vtk import *
import os

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
        self.nb_preds = 20  # Same as nb_steps in BaseRunner
        self.gridSize = [27, 27, 32]
        self.size = [0.25312511567026375, 0.25312511567026375, 0.30000013709068296] # should be the length of the box and not its resolution
        self.nb_points_in_grid = self.gridSize[0]*self.gridSize[1]*self.gridSize[2]  # 23328
        self.nb_channels_in = 4

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
        # Sending training data read from dataset
        self.set_training_data(self.sample_in, self.sample_out)

        # Store prediction
        prediction = self.get_prediction(self.sample_in)
        self.store_prediction_to_vts(prediction)
        # self.set_additional_dataset('prediction', prediction)

        self.nb_step += self.increment

    def store_prediction_to_vts(self, prediction):
        # Create vtk grid
        grid = vtkStructuredGrid()
        grid.SetDimensions((self.gridSize[0], self.gridSize[1], self.gridSize[2]))
        points = vtkPoints()
        points.SetNumberOfPoints(self.nb_points_in_grid)
        pID = 0
        start_x = -self.size[0] / 2
        start_y = -self.size[1] / 2
        start_z = -self.size[2] / 2
        d_x = self.size[0] / (self.gridSize[0] - 1)
        d_y = self.size[1] / (self.gridSize[1] - 1)
        d_z = self.size[2] / (self.gridSize[2] - 1)
        for k in range(0, self.gridSize[2]):
            for j in range(0, self.gridSize[1]):
                for i in range(0, self.gridSize[0]):
                    x = start_x + d_x * i
                    y = start_y + d_y * j
                    z = start_z + d_z * k
                    points.SetPoint(pID, x, y, z)
                    pID += 1
        grid.SetPoints(points)

        # Save the input to a vts file
        if self.nb_channels_in == 1:
            pass
        elif self.nb_channels_in == 4:
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
