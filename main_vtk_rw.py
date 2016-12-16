from vtk_rw import read_vtk
import numpy as np

filename = '/Users/ghfc/Desktop/P8_F10_test.vtk'

#filename = '/Users/ghfc/Desktop/P8_F10_DTI.vtk'

vertex_array, face_array, data_array = read_vtk(filename)

#print vertex_array