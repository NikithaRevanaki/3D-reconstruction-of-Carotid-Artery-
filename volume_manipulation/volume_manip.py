#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, sys
from PIL import Image
import SimpleITK as sitk
import math

def save_numpy_array(array, name):
	"""saves a numpy array in the 'volume' directory

	Args:
		array (ndarray): saved array
		name (str): name of the array
	"""
	if not os.path.exists('volumes'):
		os.mkdir('volumes')
		print("Volume file created")
	with open(os.path.join('volumes', 'vol_'+name+'.npy'), 'wb') as f:
		np.save(f, array)

def load_numpy_array(name):
	"""loads a numpy array from root

	Args:
		name (str): name of the loaded array

	Returns:
		ndarray: loaded array
	"""
	# name = os.path.join('volumes', name)
	if os.path.exists(name):
		with open(name, 'rb') as f:
			array = np.load(f)
		return array
	else:
		print('Path '+name+' does not exist.')
		sys.exit(2)

def save_images_in_array(arr, filename):
	"""saves all slices of an array in a file as png images

	Args:
		arr (ndarray): array to be sliced and saved
		filename (str): output file
	"""
	if not os.path.exists(filename):
		os.mkdir(filename)
		print('Path created.')
	for slice_num in range(arr.shape[0]):
		# with open(os.path.join(filename, name), 'wb') as f:
		name = ''
		if int(slice_num) < 10:
			name = '0'
		name += str(slice_num)+'.png'
		slice = Image.fromarray(arr[slice_num, :, :]).convert('L')
		slice.save(os.path.join(filename, name))
	print('Images saved')

def shift_volumes(shift2, arr1, arr2):
	"""zero pad volumes after registration, 
	   this is done to deal with possible negative coordinate values

	Args:
		shift2 (int): the coordinate shift of the second array
		arr1 (ndarray): first shifted array
		arr2 (ndarray): second shifted array

	Returns:
		ndarray, ndarray: shifted arrays
	"""
	_, h1, w1 = arr1.shape #tr
	_, h2, w2  = arr2.shape #lo
	if(shift2 > 0):
		zero_pad = np.zeros((abs(int(shift2)), h2, w2), dtype=np.uint8)
		arr2 = np.concatenate((zero_pad, arr2))
	elif(shift2 < 0):
		zero_pad = np.zeros((abs(int(shift2)), h1, w1), dtype=np.uint8)
		arr1 = np.concatenate((zero_pad, arr1))
	return arr1, arr2

def lin_interp(img_array, coords):
	"""linear interpolation of parallel slices

	Args:
		img_array (ndarray): the interpolated slices
		coords (ndarray): the coordinates of the slices in the interpolated axis 

	Returns:
		ndarray: the resulting volume
	"""
	n, h, w = img_array.shape
	diffs = np.diff(coords)
	volume = np.empty([coords[-1]-coords[0]+1, h, w], dtype=np.int16)
	start_coord = 0
	img_array = img_array.astype(np.int16)
	for i in range(n-1):
		im1 = img_array[i, :, :]
		im2 = img_array[i+1, :, :]
		d = diffs[i]
		xnew = np.arange(0, d, 1).astype(np.int16)
		for j in range(d):
			volume[start_coord+j, :, :] = im1+xnew[j]*1/d*(im2-im1)
		start_coord+=d
	volume[-1, :, :] = img_array[-1, :, :]
	return volume.astype(np.uint8)

def dw_sum(vol1, vol2, coords1, coords2, transform):
	"""sums two volumes and applies weights to all elements

	Args:
		vol1 (ndarray): first volume
		vol2 (ndarray): second volume
		coords1 (ndarray): coordinates of original slices in the first volume
		coords2 (ndarray): coordinates of original slices in the second volume
		transform (sitk Transform): the Euler transform between the volumes

	Returns:
		ndarray: the fused volume
	"""
	vol1 = vol1.astype(np.float32)
	vol2 = vol2.astype(np.float32)
	res1 = np.ones(vol1.shape, dtype=np.float32)
	res2 = np.ones(vol2.shape, dtype=np.float32)
	d1 = np.diff(coords1)
	d2 = np.diff(coords2)
	if coords1[0]>0:
		res1[:, :, :coords1[0]]=0
		vol1[:, :, :coords1[0]]=0
	vol1, res1 = give_weights(coords1, vol1, res1, d1)
	vol1 = transform_vol(transform, vol2, vol1).astype(np.float32)
	res1 = transform_vol(transform, vol2, res1).astype(np.float32)
	vol2 = np.transpose(vol2, (2, 1, 0)) 
	res2 = np.transpose(res2, (2, 1, 0)) 
	vol2, res2 = give_weights(coords2, vol2, res2, d2)
	vol2 = np.transpose(vol2, (2, 1, 0)) 
	res2 = np.transpose(res2, (2, 1, 0))
	res_vol = (vol1+vol2)/(res1+res2)
	return res_vol.astype(np.uint8)

def give_weights(coords, vol, res, d):
	"""computes the weights of the summed volumes

	Args:
		coords (ndarray): coordinates of the original slices
		vol (ndarray): the volume to be weighted
		res (ndarray): array containing weights
		d (ndarray): the distances between slices

	Returns:
		ndarray, ndarray: a weighted volume and the weights
	"""
	for i in range(len(coords)-1):
		start = coords[i]
		if start > vol.shape[0]:
			break
		diff = d[i]
		divider = 2
		for j in range(1, diff):
			if start+j < vol.shape[0] and start+j >= 0:
				vol[start+j, :, :] /= divider
				res[start+j, :, :] = 1/divider
				if j < diff/2:
					divider+=1
					if diff%2==1 and j== int(diff/2):
						divider-=1
				else: divider-=1
			else: divider+=1
	return vol, res


def transform_vol(transform, fixed_image, moving_image):
	"""applies found transform to an image, returns the transformed image

	Args:
	    transform (sitk Transform): transform to be applied to the image
	    fixed_image (numpy array): fixed image
	    moving_image (numpy array): moving image image
	Returns:
		ndarray: transformed image
	"""
	
	fixed = sitk.Cast(sitk.GetImageFromArray(fixed_image), sitk.sitkFloat32)
	moving = sitk.Cast(sitk.GetImageFromArray(moving_image), sitk.sitkFloat32)
	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetTransform(transform)
	resampler.SetDefaultPixelValue(0)
	moving_output = resampler.Execute(moving)
	return sitk.GetArrayFromImage(moving_output)

def transform_coords(coords):
	"""transforms the coordinates which were obtained with registration

	Args:
		coords (ndarray): the slice coordinates

	Returns:
		ndarray: transformed coordinates
	"""
	shift = np.min(coords)
	coords -= shift
	if shift < 0:
		shift = -1*(round(-1*shift))
	else: shift = round(shift)
	coords = np.sort(np.round(coords)).astype(np.int16)
	while(any(coords[1:] == coords[:-1])):
		coords[1:][coords[1:] == coords[:-1]] = coords[:-1][coords[1:] == coords[:-1]]+1
	coords += shift
	return shift, coords

def snr(original, created):
	"""computes the signal to noise ration of a 3D volume

	Args:
		original (ndarray): original volume
		created (ndarray): crated volume

	Returns:
		float: the snr of the input volume
	"""
	created = created.astype(np.int64)
	original = original.astype(np.int64)
	a = np.sum(np.sum(original**2))
	b = np.sum(np.sum((original-created)**2))
	print(a, b, a/b)
	if b == 0: 
		return 0
	else: 
		return 10*math.log10(a/b)

def save_overlayed(im1, im2, it):
	"""saves a composite image made from two images

	Args:
		im1 (ndarray): first image
		im2 (ndarray): second image
		it (str): iteration number
	"""
	writer = sitk.ImageFileWriter()
	img1 = sitk.Cast(sitk.GetImageFromArray(im1), sitk.sitkFloat32)
	img2 = sitk.Cast(sitk.GetImageFromArray(im2), sitk.sitkFloat32)
	img1 = sitk.Cast(sitk.RescaleIntensity(img1), sitk.sitkUInt8)
	img2 = sitk.Cast(sitk.RescaleIntensity(img2), sitk.sitkUInt8)
	comp_img = sitk.Compose(img1, img2, img1//2. + img2//2.)
	writer.SetFileName('c'+it+'.png')
	writer.Execute(comp_img)
	
def save_cross_sections(vol1, vol2, it, type):
	"""saves a cross section of two volumes

	Args:
		vol1 (ndarray): first array
		vol2 (ndarray): second array
		it (str): interation number
		type (str): array type
	"""
	nm, hm, wm = vol1.shape
	nm =int(nm/2)
	hm=int(hm/2)
	wm=int(wm/2)
	Image.fromarray(vol1[nm, :, :]).save('im1'+it+type+'a'+'.png')
	Image.fromarray(vol1[:vol2.shape[0], :, wm].T).save('im3'+it+type+'a'+'.png')
	Image.fromarray(vol2[nm, :, :vol1.shape[2]]).save('im1'+it+type+'b'+'.png')
	Image.fromarray(vol2[:, :, wm].T).save('im3'+it+type+'b'+'.png')
	save_overlayed(vol1[nm, :, :], vol2[nm, :vol1.shape[1], :vol1.shape[2]], it+type+'1')
	save_overlayed(vol1[:, :, wm].T, vol2[:vol1.shape[0], :vol1.shape[1], wm].T, it+type+'3')


# In[ ]:




