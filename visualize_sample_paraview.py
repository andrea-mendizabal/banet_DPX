# state file generated using paraview version 5.8.1

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

###########################################################
# CHOOSE THE SAMPLE TO VISUALIZE
NUM_SAMPLE = 0
NUM_SAMPLE_GT = 4
NUM_START = 10000
# PATH TO PREDICTION DIRECTORY
prediction_dir   = '/home/andrea/Projects/plugins-sofa/DeepPhysX/Application/banet_DPX/predictions'
# PATH TO GROUND TRUTH DIRECTORY
ground_truth_dir = '/media/andrea/data/post_doc_verona/banet/testing_dataset_different'#/media/andrea/data/post_doc_verona/banet/data_set_inference_preop+displ'


###########################################################
# Create full paths
import os
ground_basename  = 'voxelized_displacement.vts'
pred_basename    = 'input_and_prediction_' + str(NUM_SAMPLE) + '.vts'

ground_filename  = os.path.join( ground_truth_dir, f"{NUM_START + NUM_SAMPLE_GT:06d}", ground_basename )
pred_filename    = os.path.join( prediction_dir, pred_basename )

partial_deformed_surface_fln = os.path.join( ground_truth_dir, f"{NUM_START + NUM_SAMPLE_GT:06d}", 'partialDeformedSurface.vtp' )
##########################################################


#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [950, 638]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.21711664844151846, 0.23571683395394874, -0.34145682072820177]
renderView1.CameraViewUp = [-0.4149876986652965, 0.8504736541511524, 0.3232333113267679]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.2598076314591589
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Structured Grid Reader'
voxelizedvts = XMLStructuredGridReader(FileName=[ground_filename])
voxelizedvts.PointArrayStatus = ['displacement', 'preoperativeSurface', 'stiffness']
# voxelizedvts.PointArrayStatus = ['intraoperativeSurface0', 'stiffness']


# create a new 'XML Structured Grid Reader'
predvts = XMLStructuredGridReader(FileName=[pred_filename])
predvts.PointArrayStatus = ['stiffness']

# Load intraoperative surface and geometry at rest
partial_deformed_surface = XMLPolyDataReader(FileName=[partial_deformed_surface_fln])
Show(partial_deformed_surface)

# create a new 'Threshold'
threshold1 = Threshold(Input=voxelizedvts)
threshold1.Scalars = ['POINTS', 'stiffness']
threshold1.ThresholdRange = [0.9, 1.0]

# create a new 'Threshold'
threshold2 = Threshold(Input=predvts)
threshold2.Scalars = ['POINTS', 'stiffness']
threshold2.ThresholdRange = [0.48, 1]

#______________________________________________________________________________________________________________________

# show data from threshold1
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold1Display.Representation = 'Surface With Edges'
threshold1Display.AmbientColor = [1.0, 1.0, 0.0]
threshold1Display.ColorArrayName = [None, '']
threshold1Display.DiffuseColor = [1.0, 1.0, 0.0]
threshold1Display.OSPRayScaleArray = 'stiffness'
threshold1Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold1Display.SelectOrientationVectors = 'None'
threshold1Display.ScaleFactor = 0.010476190643385053
threshold1Display.SelectScaleArray = 'None'
threshold1Display.GlyphType = 'Arrow'
threshold1Display.GlyphTableIndexArray = 'None'
threshold1Display.GaussianRadius = 0.0005238095321692526
threshold1Display.SetScaleArray = ['POINTS', 'stiffness']
threshold1Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold1Display.OpacityArray = ['POINTS', 'stiffness']
threshold1Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold1Display.DataAxesGrid = 'GridAxesRepresentation'
threshold1Display.PolarAxes = 'PolarAxesRepresentation'
threshold1Display.ScalarOpacityUnitDistance = 0.02345339872615281

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold1Display.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold1Display.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

#______________________________________________________________________________________________________________________

# show data from threshold2
threshold2Display = Show(threshold2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold2Display.Representation = 'Surface With Edges'
threshold2Display.AmbientColor = [1.0, 0.0, 0.0]
threshold2Display.ColorArrayName = [None, '']
threshold2Display.DiffuseColor = [1.0, 0.0, 0.0]
threshold2Display.Opacity = 0.5
threshold2Display.OSPRayScaleArray = 'stiffness'
threshold2Display.OSPRayScaleFunction = 'PiecewiseFunction'
threshold2Display.SelectOrientationVectors = 'None'
threshold2Display.ScaleFactor = 0.010000000149011612
threshold2Display.SelectScaleArray = 'None'
threshold2Display.GlyphType = 'Arrow'
threshold2Display.GlyphTableIndexArray = 'None'
threshold2Display.GaussianRadius = 0.0005000000074505806
threshold2Display.SetScaleArray = ['POINTS', 'stiffness']
threshold2Display.ScaleTransferFunction = 'PiecewiseFunction'
threshold2Display.OpacityArray = ['POINTS', 'stiffness']
threshold2Display.OpacityTransferFunction = 'PiecewiseFunction'
threshold2Display.DataAxesGrid = 'GridAxesRepresentation'
threshold2Display.PolarAxes = 'PolarAxesRepresentation'
threshold2Display.ScalarOpacityUnitDistance = 0.02223639179391078

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold2Display.ScaleTransferFunction.Points = [-0.00018948502838611603, 0.0, 0.5, 0.0, 6.921183626218408e-08, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold2Display.OpacityTransferFunction.Points = [-0.00018948502838611603, 0.0, 0.5, 0.0, 6.921183626218408e-08, 1.0, 0.5, 0.0]

