"""
#07 - Prediction
Launch a running session.
"""

# Python imports
import os

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

# Tutorial related imports
from T3_configuration import BanetEnvironment, net_config


def launch_prediction():
    # Adapt the Environment config to avoid using Client / Server Architecture
    env_config = BaseEnvironmentConfig(environment_class=BanetEnvironment,
                                       visualizer=None,
                                       simulations_per_step=1,
                                       use_dataset_in_environment=False,
                                       param_dict={'increment': 1},
                                       as_tcp_ip_client=False)

    # Adapt the Dataset config with the existing dataset directory
    dataset_config = BaseDatasetConfig(dataset_dir=os.path.join(os.getcwd(), 'sessions/liver_test_4ch/dataset'),
                                       partition_size=3,
                                       shuffle_dataset=False,
                                       normalize=True)

    # Create the Pipeline
    pipeline = BaseRunner(session_name='sessions/banet_training_liver_test_4ch_1',
                          environment_config=env_config,
                          dataset_config=dataset_config,
                          network_config=net_config,
                          nb_steps=10)

    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_prediction()
