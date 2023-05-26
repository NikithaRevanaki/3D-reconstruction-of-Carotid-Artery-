#!/usr/bin/env python
# coding: utf-8

# In[3]:


from cv2 import invert
import pyvista as pv
import numpy as np
import os, sys
from PIL import Image



#sys.path.insert(0, os.path.abspath('image_manipulation'))
from image_manipulation.image_manip import series_to_3d_array

def visualise_3d_grid(img_array):
	"""3d visualisation of a created volume
	   shows the outline of the volume and its slices in three planes

	Args:
		img_array (ndarray): images to be shown
	"""
	grid = pv.UniformGrid()
	img_array = np.transpose(img_array, (2, 1, 0))
	img_array[img_array<20]=0
	grid.origin = (0, 0, 0)
	grid.spacing = (1, 1, 1)
	grid.dimensions = img_array.shape
	grid.point_data["values"] = img_array.flatten(order="F")
	p = pv.Plotter(shape = (1, 2))
	p.set_background("white")
	p.add_mesh_isovalue(grid, widget_color='black')
	p.add_axes(color='b',x_color='r', y_color='g', z_color='b', xlabel='X', ylabel='Y', zlabel='Z', line_width=2, labels_off=False)
	p.remove_scalar_bar()
	p.view_vector([0, 0, -1], [0, -1, 0])

	p.subplot(0, 1)
	p.add_mesh_slice_orthogonal(grid, tubing=False, widget_color='white', cmap='gray')
	p.remove_scalar_bar()
	p.add_axes(color='b',x_color='r', y_color='g', z_color='b', xlabel='X', ylabel='Y', zlabel='Z', line_width=2, labels_off=False)
	p.view_vector([0, 0, -1], [0, -1, 0])
	p.show()

def show_grid(img_array, spacing=1):
	"""3d visualisation of array of images in a grid, zero values are not invisible

	Args:
		img_array (ndarray)): array to be shown
		spacing (int, optional): distance between images. Defaults to 1.
	"""
	grid = pv.UniformGrid()	#print(d.shape)
	grid.origin = (0, 0, 0)
	grid.spacing = (spacing, 1, 1)
	grid.dimensions = img_array.shape
	grid.point_data["values"] = img_array.flatten(order="F")
	p = pv.Plotter()
	p.set_background('white')
	p.add_volume(grid, cmap='gray')
	p.show()

def visualise_two_grids(img_array1, img_array2, spacing=1):
	"""shows two volumes in space

	Args:
		img_array1 (ndarray): first volume
		img_array2 (ndarray): second volume
		spacing (int, optional): the spacing in z axis. Defaults to 1.
	"""
	grid1 = pv.UniformGrid()
	grid1.origin = (0, 0, 0)
	grid1.spacing = (spacing, 1, 1)
	grid1.dimensions = img_array1.shape
	grid1.point_data["values"] = img_array1.flatten(order="F")
	grid2 = pv.UniformGrid()
	grid2.origin = (0, 0, 0)
	grid2.spacing = (spacing, 1, 1)
	grid2.dimensions = img_array2.shape
	grid2.point_data["values"] = img_array2.flatten(order="F")
	p = pv.Plotter()
	p.set_background("black")
	p.add_volume(grid1)
	p.add_volume(grid2)
	p.show()