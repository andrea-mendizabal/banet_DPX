"""
#05 - Offline Training
Launch a training session with an existing Dataset.
"""

# Python related imports
import copy
import os

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig

# Tutorial related imports
from T3_configuration import env_config, net_config


def launch_training():
    # Adapt the Dataset config with the existing dataset directory
    dataset_config = BaseDatasetConfig(dataset_dir=os.path.join(os.getcwd(), 'sessions/training_data_liver_one_frame_from_three_non_consecutive_defs'),
                                       partition_size=3,
                                       shuffle_dataset=False,
                                       normalize=True)
    # Create the Pipeline
    pipeline_config = dict(
        session_dir=os.getcwd(),
        session_name='sessions/banet_training_liver_one_frame_from_three_non_consecutive_defs/',
        environment_config=env_config,
        dataset_config=dataset_config,
        network_config=net_config,
        nb_epochs=100,
        nb_batches=1052,
        batch_size=5,
    )
    pipeline = BaseTrainer(**pipeline_config)

    # Launch the Pipeline
    pipeline.execute()


if __name__ == '__main__':
    launch_training()
