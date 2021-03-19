"""Program to count the value of coins on an image."""

from typing import Any

import cv2
from numpy import ndarray

import coincount_lib as ccl


def count(arg: Any = None) -> None:
    """
    Program to count the value of coins on an image.

    Parameters:
        arg (Any): Arguments from argparse.
    """
    try:    #set if not from argparse
        ref_mm = []
        if arg is None:
            ref_mm = [35, 50]
            image_path = './test.jpg'
            blur = 3
            threshold = 110
            acc_coin = 0.15
            acc_rect = 0.01
            model_path = './svm/model.sav'
        else:
            ref_mm.append(arg.refa)
            ref_mm.append(arg.refb)
            image_path = arg.image
            blur = arg.blur
            threshold = arg.threshold
            acc_coin = arg.acc_coin
            acc_rect = arg.acc_rect
            model_path = arg.model

        coin = []  # list to save values of coins
        number_coins = 0  # save number of coins
        number_rect = 0  # save number of rectangles
        money = 0  # value of money

        image = cv2.imread(image_path)  # read image
        cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))  # resize image
        cntr = ccl.contour(image, blur=blur, threshold=threshold)  # get contours

        cv2.imshow('blur', cv2.resize(cntr[2], (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5))))  # show blur
        cv2.imshow('bnry', cv2.resize(cntr[3], (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5))))  # show binary

        for c in cntr[0]:
            circ = ccl.circle(c, image, accuracy=acc_coin, min_area=700)  # check if contour is a circle
            rect = ccl.rectangle(c, accuracy=acc_rect, min_area=1000)  # check if contour is a rectangle

            if circ[0] is True:
                x, y, _, _ = cv2.boundingRect(c)
                cv2.drawContours(image, [c], -1, (0, 255, 0), 5)  # draw green around circles
                cv2.putText(image, str(number_coins), (x, y), cv2.FONT_ITALIC, 3, (0, 0, 0), 2)  # write number of coin
                coin.append([circ[1], circ[2], circ[3], circ[4], circ[5]])  # append values of circle to list
                number_coins += 1  # rise number of coins +1
            elif rect[0] is True:
                cv2.drawContours(image, [c], -1, (255, 0, 0), 5)  # draw blue around rectangle
                ref_pxl = [rect[1], rect[2]]  # write values of rectangle to list
                number_rect += 1  # rise number of rectangles +1
            else:
                cv2.drawContours(image, [c], -1, (0, 0, 255), 5)  # draw red around unknown

        if number_rect != 1:
            raise Exception('Error: There are less or more reference rectangles than 1.')

        cv2.imshow('image', cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5))))  # show image
        cv2.waitKey(0)

        image_ok = input('Image OK?: ')
        if image_ok == 'n' or image_ok == 'no' or image_ok == 'N' or image_ok == 'No' or image_ok == 'NO' or image_ok == '0' or image_ok == '-1':  # ask if img is ok if not close prg
            print('Program closed.')
            return

        len_pxl = ccl.len_pixel(ref_mm, ref_pxl)  # get length of pixel

        for c in coin:  # convert area and perimeter to mm
            for i in range(0, 2):
                c[i] = ccl.convert_len(len_pxl, c[i])

        model = ccl.svm_readmodel(model_path)  # load model

        pred = model.predict(coin)  # predict coins
        pred = ndarray.tolist(pred)  # to array

        for i in range(0, len(coin)):  # calc money
            print('Coin', i, ':', pred[i], 'cnts')
            if pred[i] == 1:
                money += 0.01
            elif pred[i] == 2:
                money += 0.02
            elif pred[i] == 5:
                money += 0.05

        print('Money:', money)

        cv2.destroyWindow('blur')
        cv2.destroyWindow('bnry')
        cv2.waitKey(0)
    except:
        raise


if __name__ == '__main__':
    try:
        count()
    except:
        raise
