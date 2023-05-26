#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 as cv
import numpy as np

def delete_plate(array, num_of_images):
    """deletes the plate from every image

    Args:
        path (str): path of images
        array (ndarray): array with images
    """
    for i in range(num_of_images):
        array[i, :, :] = find_plate(array[i, :, :].copy())
    return array


def find_plate(original_img):
    """finds a plate in the image and deletes it

    Args:
        original_img (ndarray): image to be changed

    Returns:
        ndarray: changed image
    """
    H, W = original_img.shape
    ret, img  = cv.threshold(original_img, 60, 255, cv.THRESH_BINARY)
    img = cv.erode(img, np.ones((11,11), np.uint8))
    img = cv.dilate(img, np.ones((11,11), np.uint8))
    img = cv.erode(img, np.ones((5,5), np.uint8))
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    best_contour = None
    second_best = None
    longest_side=0
    second_longest_side = 0
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        length = max([w, h])
        if length > longest_side:
            if longest_side>second_longest_side:
                second_best = best_contour
                second_longest_side = longest_side
            best_contour = cnt
            longest_side = length
        elif length>second_longest_side:
            second_best = cnt
            second_longest_side = length
    if best_contour is None:
        return img
    best_rect = cv.minAreaRect(best_contour)
    (center, (h, w), angle) = best_rect #!!! h, w is sometimes switched
    
    if w > W/2:
        best_contour = np.vstack((best_contour, np.reshape([H, W], (1, 1, 2))))
        best_contour = np.vstack((best_contour, np.reshape([0, W], (1, 1, 2))))
        best_rect = cv.minAreaRect(best_contour)
        (center, (h, w), angle) = best_rect
        box = np.int0(cv.boxPoints(best_rect))
        draw = cv.drawContours(np.full((img.shape), 1, dtype='uint8'), [box], 0, (0,0,0), -1)
        img = draw*original_img
    else:
        if second_best is not None:
            best_rect = np.vstack((best_contour, second_best))
        else:
            best_rect = best_contour
        best_rect = np.vstack((best_rect, np.reshape([H, W], (1, 1, 2))))
        best_rect = np.vstack((best_rect, np.reshape([0, W], (1, 1, 2))))
        cnt = cv.convexHull(best_rect)
        best_rect = cv.minAreaRect(cnt)
        box = np.int0(cv.boxPoints(best_rect))
        draw = cv.drawContours(np.full((img.shape), 1, dtype='uint8'), [box], 0, (0,0,0), -1)
        img = draw*original_img
    return img


def find_roi(array, num_of_images):
    """this function estites the diameter of the artery by finding 
    the diameter in every image and choosing the maximum found diameter
    as the result

    Args:
        path (str): path of images
        array (ndarray): array with images
        
    Returns:
        str: output path
        int: estimated diameter of the vein in pixels
    """
    diameters = []
    for i in range(num_of_images):
        diameter = find_roi_without_plate(array[i, :, :].copy())
        diameters.append(diameter)
    return int(max(diameters))


def find_roi_without_plate(original_img):
    """finds region of interest in image

    Args:
        original_img (ndarray): image with ROI

    Returns:
        ndarray: img with found ROI
    """
    img = cv.GaussianBlur(original_img, (7, 7), 0)
    img = cv.Canny(img, 10, 20)
    img = cv.dilate(img, np.ones((21,5), np.uint8))
    img = cv.erode(img, np.ones((4,21), np.uint8))
    img = cv.dilate(img, np.ones((10,10), np.uint8))
    img = cv.erode(img, np.ones((17,17), np.uint8))
    img = cv.dilate(img, np.ones((10,10), np.uint8))
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    best_volume = 0
    best_contour = None
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        if w*h > best_volume: # finds the middle of the vein
            best_volume = w*h
            best_contour = cnt
    if best_contour is None:
        return original_img
    best_rect = cv.minAreaRect(best_contour)
    (center, (width, height), angle) = best_rect
    diameter = np.min([width, height])
    box = np.int0(cv.boxPoints(best_rect))
    draw = cv.drawContours(np.full((img.shape), 0, dtype='uint8'), [box], 0, (255,255,255), -1)
    img = cv.bitwise_and(original_img, draw)
    draw = cv.drawContours(np.full((img.shape), 1, dtype='uint8'), [box], -1, (255,255,255), 2)
    return diameter


# In[ ]:




