"""
Library with all the functions for CoinCount!
"""

import cv2
from math import pi, sqrt
import numpy as np
import csv
import pickle
from typing import Any

def contour(image: Any, blur: int=3, threshold: int=110) -> list:
    """
    Returns the contours of object in a image.

    Parameters:
        image: Image to get contours from.
        blur (int): Value for Gaussian Blur. Must be odd and positive.
        threshold (int): Threshold value for conversion from grey image to binary image.

    Returns:
        (list): List with founded contours, gray image, blurred image and binary image.
            -> [[contours], GrayImage, BlurImage, BinaryImage]
    """
    try:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # to gray
        img_blur = cv2.GaussianBlur(img_gray, (blur, blur), 0)  # blur
        img_bnry = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY_INV)[1]  # binary
        cnts, _ = cv2.findContours(img_bnry, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get contours
        return [cnts, img_gray, img_blur, img_bnry]
    except:
        raise

def circle(contour: list, image: Any, accuracy: float=0.15, min_area: float=0) -> list:
    """
    Return if a contour is a circle or not.

    If TRUE it also returns area, perimeter and mean color.

    Parameters:
        contour (list): Contour to check.
        image: Image in which the contour appears.
        accuracy (float): Accuracy to determine if the contour is a circle or not. Must be between 0 and 1.
        min_area (float): Minimum area of circle.

    Returns:
        (list): List with result, area, perimeter, mean blue, mean green and mean red. ->
            [True/False, Area, Perimeter, MeanBlue, MeanGreen, MeanRed]

    """
    try:
        area = cv2.contourArea(contour) # area
        perimeter = cv2.arcLength(contour, True) # perimeter
        radius_area = sqrt(area/pi)     # radius from area
        radius_perimeter = perimeter / (2 * pi) # radius form perimeter
        if (radius_area * (1-accuracy)) <= radius_perimeter <= (radius_area * (1+accuracy)) and area >= min_area: # compare radius area perimeter
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            masked = cv2.bitwise_and(image, image, mask=mask)
            color = cv2.mean(masked, mask)
            return [True, area, perimeter, color[0], color[1], color[2]]
        else:
            return [False]
    except:
        raise

def rectangle(contour: list, accuracy: float=0.01, min_area: float=0) -> list:
    """
    Return if a contour is a rectangle or not.

    If TRUE it also returns the length of both sides.

    Parameters:
        contour (list):  Contour to check.
        accuracy (float): Accuracy to determine if the contour is a rectangle or not.
        min_area (float): Minimum area of rectangle.

    Returns:
        (list): List with result, length of side a and length of side b. -> [Ture/False, SideA, SideB]
    """
    try:
        length = []
        area = cv2.contourArea(contour) # area
        approx = cv2.approxPolyDP(contour, accuracy * cv2.arcLength(contour, True), True) # simplify contour
        if (len(approx) == 4) and area >= min_area:    # if four sides -> return length
            for i in range(0, len(approx) - 1):
                point = approx[i] - approx[i + 1]  # x and y form 0,0
                point = np.ndarray.tolist(point[0])  # point to array
                a_2 = abs(point[0]) ** 2  # calc a^2
                b_2 = abs(point[1]) ** 2  # calc b^2
                length.append(sqrt(a_2 + b_2))  # calc c and append to list

            # 4th line
            point = approx[3] - approx[0]  # x and y form 0,0
            point = np.ndarray.tolist(point[0])  # point to array
            a_2 = abs(point[0]) ** 2  # calc a^2
            b_2 = abs(point[1]) ** 2  # calc b^2
            length.append(sqrt(a_2 + b_2))  # calc c and append to list

            a = (length[0] + length[2]) / 2  # average length of side a
            b = (length[1] + length[3]) / 2  # average length of side b
            return [True, a, b]
        else:
            return [False]
    except:
        raise

def len_pixel(reference_mm: list, reference_pxl: list) -> float:
    """
    Returns the length of a pixel in mm, when there is a reference rectangle.

    Parameters:
        reference_mm (list): Length of reference rectangle in mm.
        reference_pxl (list): Length of reference rectangle in pixel.

    Returns:
        (float): Length of a pixel in mm.
    """
    try:
        reference_mm.sort()    # sort list reference_val
        reference_pxl.sort()    # sort list reference_pxl
        pxl_len_0 = reference_mm[0] / reference_pxl[0]   # calc len of pxl on short side
        pxl_len_1 = reference_mm[1] / reference_pxl[1]   # calc len of pxl on long side
        return (pxl_len_0 + pxl_len_1) / 2  # calc and return average len
    except:
        raise

def convert_len(length_of_pixel:float, length_in_pixel:float) -> float:
    """
    Returns a length of pixel in mm.

    Parameters:
        length_of_pixel (float): Length of one pixel in mm.
        length_in_pixel (float): Length of the to converting element in pixel.

    Returns:
        (float): Length of the element in pixel.
    """
    try:
        actual_length = length_of_pixel * length_in_pixel   # calc actual len
        return actual_length
    except:
        raise

def svm_readcsv(path: str, target_val: Any, list_data: list=None, list_target: list=None) -> list:
    """
    Reads data from csv to list and adds target to list.
    
    Used for Support Vector Machine.

    Parameters:
        path (str): Path of csv file.
        target_val (Any): Target of the values in csv file.
        list_data (list): List to extend data.
        list_target (list): List to extend target.

    Returns:
        (list): List with list of csv values and list of targets -> [[Csv], [Target]]
    """
    try:
        if list_data is None:   # if no data list -> crate data list
            data = []
        else:
            data = list_data
        if list_target is None: # if no target list -> crate target list
            target = []
        else:
            target = list_target

        file = open(path, 'r', newline='')  # open file
        reader = csv.reader(file)   # create reader

        for row in reader:  # add csv and target to list
            data.append(row)
            target.append(target_val)

        file.close()    # close file

        return [data, target]
    except:
        raise

def svm_writecsv(path: str, values: list) -> None:
    """
    Write values into csv file.

    Used for Support Vector Machine.

    Parameters:
         path (str): Path of csv file.
         values (list): Values to write to csv files.
    """
    try:
        file = open(path, 'a', newline='')  # open file
        writer = csv.writer(file)   # create writer
        writer.writerows(values)    # write values
        file.close()    # close file
    except:
        raise

def svm_writemodel(path: str, model: Any) -> None:
    """
    Saves model to file.

    Parameters:
        path (str): Path to save file.
        model (Any): Model to save.
    """
    try:
        file = open(path, 'wb') # open file
        pickle.dump(model, file)    # save file
        file.close()    # close file
    except:
        raise

def svm_readmodel(path: str) -> Any:
    """
    Returns saved model from file.

    Parameters:
         path (str): Path of file of saved model.

    Returns:
        (Any): Saved model.
    """
    try:
        file = open(path, 'rb') # open file
        model = pickle.load(file)   # load model
        file.close()    # close file
        return model
    except:
        raise