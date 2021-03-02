"""
Program to add new coins to csv files.
"""

import cv2
import coincount_lib as ccl

if __name__ == '__main__':
    try:
        ref_mm = [35, 50]
        coin = []  # list to save values of coins
        number_coins = 0  # save number of coins
        number_rect = 0  # save number of rectangles

        image = cv2.imread('test.jpg')  # read image
        cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))  # resize image
        cntr = ccl.contour(image, threshold=90)  # get contours

        cv2.imshow('bnry', cv2.resize(cntr[3], (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))) # show binary

        for c in cntr[0]:
            circ = ccl.circle(c, image, min_area=700)  # check if contour is a circle
            rect = ccl.rectangle(c, min_area=1000)  # check if contour is a rectangle

            if circ[0] is True:
                x, y, _, _ = cv2.boundingRect(c)
                cv2.drawContours(image, [c], -1, (0, 255, 0), 5)  # draw green around circles
                cv2.putText(image, str(number_coins), (x, y), cv2.FONT_ITALIC, 3, (0, 0, 0), 2) # write number of coin
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

        cv2.imshow('image', cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5))))
        cv2.waitKey(0)

        len_pxl = ccl.len_pixel(ref_mm, ref_pxl)  # get length of pixel

        for c in range(0, len(coin)):  # convert area and perimeter to mm
            for i in range(0, 2):
                coin[c][i] = ccl.convert_len(len_pxl, coin[c][i])

        number_of_coin = int(input('Which coin should be added?: '))
        type_of_coin = int(input('What type of coin?: '))

        if type_of_coin == 1:
            path = './svm/csvfiles/1cnts.csv'
        elif type_of_coin == 2:
            path = './svm/csvfiles/2cnts.csv'
        elif type_of_coin == 5:
            path = './svm/csvfiles/5cnts.csv'
        elif type_of_coin == 10:
            path = './svm/csvfiles/10cnts.csv'
        elif type_of_coin == 20:
            path = './svm/csvfiles/20cnts.csv'

        ccl.svm_writecsv(path, [coin[number_of_coin]])

    except:
        raise