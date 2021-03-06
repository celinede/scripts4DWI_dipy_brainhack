# scripts4DWI_dipy_brainhack
Scripts in python using Dipy to process and analyze DWI data. Those scripts got initiated during the BrainHack in Lausanne 2016.
The project is developed under the supervision of Roberto Toro at the Institut Pasteur in Paris with the help of Eleftherios Garyfallidis and Cameron Craddock.

data: the scripts are meant to compute and analyse the tractographies of the ferret DWI data recorded on a Bruker scanner (2dseq files).

The initial scripts are now accompanied by some additional python utility scripts for DWI data.

-- most of the scripts are still work in progress --

- analyze : load trk files, quickbundle, streamline length computation, filtering on the two previous information, get some information for plots
- common.py and det_tracking.py are scripts from Eleftherios Garyfallidis which I tried to adapt to my data in all the tracking something scripts 
- compareStreamlineLengthEndpointDistance : plot and compare the streamline lentgh to the endpoint distance to approximate streamline curvature
- connectivitymatrix : draft, work in progress
- densitymatrix : example of the use of dipy.utils.density_map
- findClosestPoint : project the streamline endpoints on the surface mesh
- findClosestPoint2dn : project the streamline endpoints on the surface mesh using the last two points of the streamlines as a direction for the projection
- getConnectivityMatrices_withSeedSpheres : build a connectivity matrix using as regions, spheres positionned on the mesh surface
- getEndpointsFromTrk : extract endpoints coordinates to a .txt file from a .trk file
- load&VisualizePly : load and visualise ply files
- load&VisualizeTrk : simple script to load and visualise existing trk files
- main_vtk_rw.py : main script for testing vtk_rw from a console
- meshInACube8 : build a connectivity matrix using as regions, cubes from a 2x2x2 grid built around the mesh
- meshInAGrid : build a connectivity matrix using as regions, cubes from a grid built around the mesh (choice of the number of division in every direction)
- modifyGradientTable : test to see if the gradient table could be the reason of the mistaken tractographies of my data with Dipy
- plot_surf_stat_map_3 : script to plot meshes and labels written by Julia Huntenburg (juhuntenburg, github)
- postprocessing4ExploreDTIdata : set of functions for further work after computing tractographies
- tracking_function : tracking with Dipy formated as functions
- translate&rotateEndpoints : adjust the endpoints position
- vtk_mesh_label_visualization : script using the create_fig function from plot_surf_stat_map_3 to plot mesh from my ferret data
- vtk_reformating : script to parse and reorganise messy vtk files. => one vertex per line (3 coordinates), one polygone per line as well
- vtk_rw : script written by Julia Huntenburg (juhuntenburg, github) to read and extract data from vtk files (vertex array, face array, data array). Those information are possible inputs for the create_fig function from plot_surf_stat_map_3
- Untitled 1 : example script using the functions from postprocessing4ExploreDTIdata 
