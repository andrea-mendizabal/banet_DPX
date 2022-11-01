"""
#03 - Configurations
Define configurations for Environment, Network and Dataset
"""

# Python imports
import torch

# DeepPhysX related imports
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
# from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_PyTorch.UNet.UNetConfig import UNetConfig


# Session imports
from T1_environment import BanetEnvironment
from T2_network import BanetNetwork, BanetOptimization
from loss import BCELogits_Dice_Loss
from PersonalizedOptimization import PersonalizedOptimization


# Create the Environment config
env_config = BaseEnvironmentConfig(environment_class=BanetEnvironment,      # The Environment class to create
                                   visualizer=None,                         # The Visualizer to use
                                   simulations_per_step=1,                  # The number of sub-steps to run
                                   use_dataset_in_environment=False,        # Dataset will not be sent to Environment
                                   param_dict={'increment': 1},             # Parameters to send at init
                                   as_tcp_ip_client=True,                   # Create a Client / Server architecture
                                   number_of_thread=1,                      # Number of Clients connected to Server
                                   ip_address='localhost',                  # IP address to use for communication
                                   port=10001)

# Create the Network config
net_config = UNetConfig(loss=BCELogits_Dice_Loss,
                        #optimization_class=PersonalizedOptimization,
                        lr=0.001,
                        optimizer=torch.optim.Adam,
                        network_name='BanetNetwork',
                        nb_dims=3,
                        input_size=[31, 32, 26],
                        # input_size=[27, 27, 32],
                        nb_input_channels=1,
                        nb_first_layer_channels=64,
                        nb_output_channels=1,
                        nb_steps=3,
                        two_sublayers=True,
                        border_mode='same',
                        # skip_merge=True,
                        save_each_epoch=True,
                        data_scale=1.0)


# Create the Dataset config
dataset_config = BaseDatasetConfig(partition_size=3, shuffle_dataset=True)

