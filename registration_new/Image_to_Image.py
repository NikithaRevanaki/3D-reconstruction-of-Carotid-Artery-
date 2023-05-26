import SimpleITK as sitk
import os
import numpy as np

def save_overlayed_images(images, p):
    """saves the composed image made of every two neighboring images

    Args:
        images (ndarray): array containing images
        p (str): output path
    """
    writer = sitk.ImageFileWriter()
    comp_path = p
    if not os.path.exists(comp_path):
        os.mkdir(comp_path)
    for i in range(images.shape[0]-1):
        img1 = sitk.Cast(sitk.GetImageFromArray(images[i, :, :]), sitk.sitkFloat32)
        img1 = sitk.Cast(sitk.RescaleIntensity(img1), sitk.sitkUInt8)
        img2 = sitk.Cast(sitk.GetImageFromArray(images[i+1, :, :]), sitk.sitkFloat32)
        img2 = sitk.Cast(sitk.RescaleIntensity(img2), sitk.sitkUInt8)
        comp_img = sitk.Compose(img1, img2, img1//2. + img2//2.)
        reg_name = "reg"+str(i)+'.png'
        writer.SetFileName(os.path.join(comp_path, reg_name))
        writer.Execute(comp_img)

def compute_diffs(images):
    """computes the absolute differences between images

    Args:
        images (ndarray): array containing the compared images
    """
    diffs = list()
    n = images.shape[1]*images.shape[2]
    for i in range(images.shape[0]-1):
        diffs.append(np.sum(np.sum(np.abs(images[i, :, :].astype(np.int16)-images[i+1, :, :].astype(np.int16))))/n)
    print(diffs)
    print(sum(diffs))
    
def register_image_set(image_array, num_of_images, lr, step, iters, tolerance):
    """gradually registers a series of images

    Args:
        image_array (ndarray): images in array
        num_of_images (int): number of images
        lr (float): learning rate of gradiend descent
        step (float): minimum step stopping condition
        iters (int): maximum number of iterations 
        tolerance (float): gradient magnitude tolerance

    Returns:
        npy array: registered images
    """
    image_indexes = np.arange(0, num_of_images, 1)
    middle_idx = int(num_of_images/2) # index of the image in the middle, will be used as a stationary image
    image_array = register_subset(image_indexes[middle_idx:], image_array, lr, step, iters, tolerance)
    image_array = register_subset(list(reversed(image_indexes[0:middle_idx+1])), image_array, lr, step, iters, tolerance)
    return image_array

def register_subset(indexes, images, lr, step, iters, tolerance):
    """creates a composite transform and applies it to the images

    Args:
        indexes (ndarray): indexes of images in array
        images (ndarray): images to be registered
        lr (float): learning rate of gradiend descent
        step (float): minimum step stopping condition
        iters (int): maximum number of iterations 
        tolerance (float): gradient magnitude tolerance

    Returns:
        ndarray: array of registered images
    """
    transforms = list()
    for i in range(0, len(indexes)-1): #images from the beginning of the list to middle-1
        transform = find_transform(images[indexes[i], :, :], images[indexes[i+1], :, :], lr, step, iters, tolerance)
        transforms.append(transform)
    for i in range(0, len(indexes)-1): 
        composite_transform = sitk.CompositeTransform(transforms[:i+1])
        images[indexes[i+1], :, :] = apply_transform(composite_transform, images[indexes[0], :, :], images[indexes[i+1], :, :])
    return images

def find_transform(fixed_image, moving_image, lr, step, iters, tolerance):
    """finds the rotation and translation between two images

    Args:
        fixed_image (ndarray): fixed image
        moving_image (ndarray): moving image
        lr (float): learning rate of gradiend descent
        step (float): minimum step stopping condition
        iters (int): maximum number of iterations 
        tolerance (float): gradient magnitude tolerance

    Returns:
        sitk Transform: transform to be applied to the moving image
    """

    fixed = sitk.Cast(sitk.GetImageFromArray(fixed_image), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(moving_image), sitk.sitkFloat32)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=lr,
                                                          minStep=step,
                                                          numberOfIterations=int(iters),
                                                          gradientMagnitudeTolerance=tolerance)
    registration.SetOptimizerScalesFromIndexShift()
    init_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform()) #euler - rotation, translation
    registration.SetInitialTransform(init_transform)
    registration.SetInterpolator(sitk.sitkLinear)
    out_transform = registration.Execute(fixed, moving)
    return out_transform

def apply_transform(transform, fixed_image, moving_image):
    """applies found transform to an image, returns the transformed image

    Args:
        transform (sitk Transform): transform to be applied to the image
        fixed_image (ndarray): fixed image
        moving_image (ndarray): moving image image
    Returns:
        ndarray: transfored image
    """
    fixed = sitk.Cast(sitk.GetImageFromArray(fixed_image), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(moving_image), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    moving_output = resampler.Execute(moving)
    img2 = sitk.Cast(moving_output, sitk.sitkUInt8)        
    return sitk.GetArrayFromImage(img2)


# In[ ]:




