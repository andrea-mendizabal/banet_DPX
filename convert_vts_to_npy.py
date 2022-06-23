import os
import numpy as np
from vtk import *
from vtk.util import numpy_support

####### VTK functions ##################################################################################################
def readGrid( grid_filename ):
	reader = vtkXMLStructuredGridReader()
	reader.SetFileName(grid_filename)
	reader.Update()
	return reader.GetOutput()


def getDataArray( grid_vtk, field_name ):
    # Get data array and convert it to numpy
    if not grid_vtk.GetPointData().HasArray(field_name):
        raise IOError("Grid {} does not contain the {} array".format(grid_vtk, field_name))
    return numpy_support.vtk_to_numpy( grid_vtk.GetPointData().GetArray(field_name) )

####### VTK functions ##################################################################################################


# Hardcoded data paths and parameters
path_data = '/media/andrea/data/post_doc_verona/banet/a_jeter'

path_to_save = 'sessions/a_jeter/dataset'
# path_to_save = 'sessions/testing_data_3_frames_1_ch/dataset'

displacement = True
vts_filename = 'voxelized_displacement_only.vts'
nb_channels_in = 3
# vts_filename = 'voxelized_displacement.vts'
# nb_channels_in = 4

# displacement = False
# vts_filename = 'voxelized.vts'
# nb_channels_in = 1
# nb_channels_in = 3

max_num_frames = 3
nb_channels_out = 1
# nb_points_in_grid = 23328  # 27x27x32
nb_points_in_grid = 25792  # 31x32x26


# Create directories and arrays
if not os.path.isdir(path_to_save):
    os.makedirs(path_to_save)
input_arr = np.empty((0, nb_points_in_grid, nb_channels_in))
output_arr = np.empty((0, nb_points_in_grid, nb_channels_out))

# Read all the sample folders
for sample in os.listdir(path_data):
    vts_file = path_data + '/' + sample + '/' + vts_filename
    # If vts file exists, get inputs and outputs
    if os.path.exists(vts_file):
        print("vts_file", vts_file)
        # Open sample file
        grid_vtk = readGrid(vts_file)

        # Store input numpy arrays (several cases)
        if not displacement:
            input = np.empty((nb_points_in_grid, 0))
            for num_frame in range(nb_channels_in):
                # If there is only one frame to export, export maximal deformation
                if nb_channels_in == 1:
                    num_frame = max_num_frames - 1
                data = getDataArray(grid_vtk, 'intraoperativeSurface' + str(num_frame)).reshape((nb_points_in_grid, 1))
                input = np.concatenate((input, data), axis=1)
        else:  # Working with displacements
            if nb_channels_in == 4:
                u = getDataArray(grid_vtk, 'displacement')
                sdf = getDataArray(grid_vtk, 'preoperativeSurface').reshape((nb_points_in_grid, 1))
                input = np.concatenate((u, sdf), axis=1)
            if nb_channels_in == 3:
                u = getDataArray(grid_vtk, 'displacement')
                input = u
            input_arr = np.concatenate((input_arr, np.array([input])))

        # Store output numpy arrays
        output = getDataArray(grid_vtk, 'stiffness').reshape((nb_points_in_grid, 1))
        output_arr = np.concatenate((output_arr, np.array([output])))

# Compute mean and std for further normalization
mean_inputs = input_arr.flatten().mean()
std_inputs = input_arr.flatten().std()
mean_outputs = output_arr.flatten().mean()
std_outputs = output_arr.flatten().std()
print("Inputs = {} +- {}".format(mean_inputs, std_inputs))
print("Outputs = {} +- {}".format(mean_outputs, std_outputs))

# Save numpy arrays
np.save(path_to_save + '/' + vts_filename[:-4] + '_training_IN_0.npy', input_arr)
np.save(path_to_save + '/' + vts_filename[:-4] + '_training_OUT_0.npy', output_arr)
print("There are {} samples in total.".format(input_arr.shape[0]))


