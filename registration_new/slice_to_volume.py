#!/usr/bin/env python
# coding: utf-8

# In[2]:


import SimpleITK as sitk
import numpy as np

def register_slices(slice_array_vol, fixed_volume, num_of_images, coords, depth):
    """registers thin slices from one array in the second array 

    Args:
        slice_array_vol (ndarray): volume created from slice_array
        fixed_volume (ndarray): fixed volume for registration
        num_of_images (int): number of images to be registered
        coords (ndarray): coordinates of images in volume
        depth (int): the distance between two images

    Returns:
        ndarray: array of image coordinates
    """
    out_offsets = np.empty([num_of_images]).astype(np.float32)
    d = int(depth/4)
    fixed_volume = sitk.Cast(sitk.GetImageFromArray(fixed_volume), sitk.sitkFloat32)

    for img_num in range(num_of_images):
        if coords[img_num] < 0 or coords[img_num] >= slice_array_vol.shape[0]:
            out_offsets[img_num] = -float(coords[img_num])
        else:
            if coords[img_num]-d < 0:
                img = np.empty([2*d, slice_array_vol.shape[1], slice_array_vol.shape[2]], dtype=np.uint8)
                img[:d, :, :] = slice_array_vol[coords[img_num]:coords[img_num]+d, :, :]
                img[d:, :, :] = np.repeat(slice_array_vol[coords[img_num], :, :][np.newaxis], d, axis=0).astype(np.uint8)
            elif coords[img_num]+d > slice_array_vol.shape[0]:
                img = np.empty([2*d, slice_array_vol.shape[1], slice_array_vol.shape[2]], dtype=np.uint8)
                img[:d, :, :] = slice_array_vol[coords[img_num]-d:coords[img_num], :, :]
                img[d:, :, :] = np.repeat(slice_array_vol[coords[img_num], :, :][np.newaxis], d, axis=0).astype(np.uint8)
            else:
                img = slice_array_vol[coords[img_num]-d:coords[img_num]+d, :, :]
            if img_num < int(num_of_images/2):
                out_offsets[img_num] = slice_volume_registration(fixed_volume, img, coords[img_num], 'f', d)
            else:
                out_offsets[img_num] = slice_volume_registration(fixed_volume, img, coords[img_num], 's', d)
    return out_offsets



def slice_volume_registration(fixed_volume, slice, coord, type, d):
    
    """registers one slice in a volume with 3D translation

    Args:
        fixed_volume (numpy array): fixed volume
        slice (numpy array): moving slice
        coord (int): coordinate of slice in volume
        type (str): says whether the image is from the first or second half of the volume

    Returns:
        float: final offset
    """
    init_transform = sitk.TranslationTransform(3, [0., 0., -float(coord-d)])
    moving_image = sitk.Cast(sitk.GetImageFromArray(slice), sitk.sitkFloat32)
    lr=0.1
    iters=20
    minc=0.0001
    window=10
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=30)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.02)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(learningRate=lr, 
                                                numberOfIterations=iters, 
                                                convergenceMinimumValue=minc, 
                                                convergenceWindowSize=window)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(init_transform)
    try:
        out_transform = registration.Execute(fixed_volume, moving_image)
        offset = sitk.TranslationTransform(out_transform).GetOffset()[2]
    except RuntimeError: # when the slice is outside of the volume
        if type == 's':
            offset = -float(coord-d)
        elif type == 'f':
            offset = -(1+d)
    return -1*offset+d
 


# In[ ]:




