#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import numpy as np


# In[2]:


def create_images_with_registration(images, num_of_images, coords, fixed, params):
    """applies registration to all images in an array
        creates a volume, usint the information about the mutual transformations

    Args:
        images (ndarray): original images

    Returns:
        ndarray: registered images
    """
    volume = np.empty([coords[-1]-coords[0]+1,images.shape[1],images.shape[2]], dtype=np.uint8)
    start_coord = 0
    for i in range(num_of_images-1):
        spacing = coords[i+1] - coords[i]
        if spacing != 1:
            volume[start_coord:start_coord+spacing, :, :] = create_images(images[i, :, :], images[i+1, :, :], spacing, fixed[i], params[i])
        else:
            volume[start_coord, :, :] = images[i, :, :]
        start_coord += spacing
    volume[-1, :, :] = images[-1, :, :]
    return volume


# In[3]:


def find_transforms(images):
    """finds the similarity transform between every neighboring image
       in the input array, the transform parameters are later used for 
       interpolation

    Args:
        images (ndarray): interpolated images

    Returns:
        list, list: lists containing the parameters of the found transforms
    """
    fixed = list()
    params = list()
    for i in range(images.shape[0]-1):
        transform = sitk.Similarity2DTransform(find_similarity_transform(images[i, :, :], images[i+1, :, :]))
        fixed.append(transform.GetFixedParameters())
        params.append(transform.GetParameters())
    return fixed, params


# In[4]:


def find_similarity_transform(fixed_image, moving_image):
    """finds the rotation and translation and scaling between two images

    Args:
        fixed_image (ndarray): fixed image
        moving_image (ndarray): transformed image

    Returns:
        sitk Transform: transform to be applied to the moving image
    """
    lr=1
    step=0.001
    iters=100
    tolerance=0.0005
    fixed = sitk.Cast(sitk.GetImageFromArray(fixed_image), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(moving_image), sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=lr,
                                                          minStep=step,
                                                          numberOfIterations=iters,
                                                          gradientMagnitudeTolerance=tolerance)
    registration.SetOptimizerScalesFromIndexShift()
    init_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
    registration.SetInitialTransform(init_transform)
    registration.SetInterpolator(sitk.sitkLinear)
    out_transform = registration.Execute(fixed, moving)
    return out_transform


# In[5]:


def create_images(first_image, second_image, spacing, fixed, params):
    """interpolates two images usind transformation interpolation

    Args:
        first_image (ndarray): first image of interpolation
        second_image (ndarray): second image of interpolation
        spacing (int): the number of missing slices to be created
        fixed (list): list of fixed parameters
        params (list): list of parameters

    Returns:
        _type_: _description_
    """
    vol = np.empty([spacing, first_image.shape[0], first_image.shape[1]], dtype=np.uint8)
    scale, angle, dx, dy = params
    num_of_first_images = int((spacing)/2)
    vol[:num_of_first_images+1, :, :] = create_volume_part(first_image, num_of_first_images, fixed, (scale-1)/spacing, angle/spacing, dx/spacing, dy/spacing, '-')
    num_of_second_images = (spacing-1)-num_of_first_images
    vol[num_of_first_images+1:, :, :] = create_volume_part(second_image, num_of_second_images, fixed, (scale-1)/spacing, angle/spacing, dx/spacing, dy/spacing, '+')
    return vol


# In[6]:


def create_volume_part(image, num_of_images, fixed, orig_scale, orig_angle, x, y, symbol):
    """interpolates a half of the space between two images

    Args:
        image (ndarray): the image to be transformed
        num_of_images (int): the number of images to be created
        fixed (list): list of fixed parameters of the transformation
        orig_scale (float): the scale which changes the transform
        orig_angle (float): the angle which changes the transform
        x (float): the partial shift in x direction
        y (float): the partial shift in y direction
        symbol (str): type of computation to be applied

    Returns:
        ndarray: interpolated volume part
    """
    transform = sitk.Similarity2DTransform()
    transform.SetFixedParameters(fixed)
    angle = 0
    scale = 1
    tx=0
    ty=0
    sitk_img = sitk.Cast(sitk.GetImageFromArray(image), sitk.sitkFloat32)
    images = np.empty([num_of_images+1, image.shape[0], image.shape[1]], dtype=np.uint8)
    if symbol == '+':
        images[-1, :, :] = image
        idx = num_of_images-1
    elif symbol == '-':
        images[0, :, :] = image
        idx = 1
    for _ in range(num_of_images):
        if symbol == '+':
            scale += orig_scale
            angle += orig_angle
            tx += x
            ty += y
        elif symbol == '-':
            scale -= orig_scale
            angle -= orig_angle
            tx -= x
            ty -= y
        transform.SetParameters([scale, angle, tx, ty])
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        moving_output = resampler.Execute(sitk_img)
        images[idx, :, :] = sitk.GetArrayFromImage(sitk.Cast(moving_output, sitk.sitkUInt8))
        if symbol == '-':
            idx +=1
        elif symbol == '+':
            idx -=1
    if symbol == '+':
        return images[:-1, :, :]
    return images

