#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import numpy as np

def volume_registration(fixed, moving):
    """ registration of two volumes with Euler transformation

    Args:
        fixed (ndarray): fixed volume
        moving (ndarray): moving volume

    Returns:
        ndarray: transformed registered volume
    """
    lr=0.1
    iters=30
    minc=1e-4
    window=10
    fixed_image =  sitk.GetImageFromArray(fixed)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.GetImageFromArray(moving)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                moving_image, 
                                                sitk.Euler3DTransform(), 
                                                sitk.CenteredTransformInitializerFilter.MOMENTS) # initialization with centroids
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
    registration.SetInitialTransform(transform)
    out_transform = registration.Execute(fixed_image, moving_image)  
    return apply_transform(moving_image, moving_image, out_transform), out_transform, out_transform.GetInverse()

def apply_transform(fixed_image, moving_image, out_transform):
    """applies the found transform

    Args:
        fixed_image (ndarray): fixed image
        moving_image (ndarray): moving image
        out_transform (sitk Transform): found transform

    Returns:
        ndarray: transformed moving array
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(out_transform)
    moving_output = resampler.Execute(moving_image)
    moving_output = sitk.GetArrayFromImage(moving_output).astype(np.uint8)
    return moving_output
    


# In[ ]:




