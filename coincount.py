"""Program to use all the CoinCount programs from the command line."""

import argparse

from count import count
from newcoin import new
from svm import svm

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-svm', '--SupportVectorMachine', action='store_true', dest='svm',
                            help='If set start program to create and save Support Vector Machine.')
        parser.add_argument('-n', '--new', action='store_true', dest='new',
                            help='If set start program to add new coins to csv files.')
        parser.add_argument('-c', '--count', action='store_true', dest='count',
                            help='If set start program to count values of coins in an image.')
        parser.add_argument('-m', '--model', type=str, dest='model', default='svm/model.sav',
                            help='Path to Support Vector Machine model.')
        parser.add_argument('-i', '--image', type=str, dest='image', default=None, help='Path to the image.')
        parser.add_argument('-ra', '--refA', type=float, dest='refa', default=None,
                            help='Value of first length in mm of reference rectangle.')
        parser.add_argument('-rb', '--refB', type=float, dest='refb', default=None,
                            help='Value of second length in mm of reference rectangle.')
        parser.add_argument('-b', '--blur', type=int, dest='blur', default=3, help='Value of Gaussian Blur.')
        parser.add_argument('-t', '--threshold', type=int, dest='threshold', default=110,
                            help='Threshold value for conversion from grey image to binary image.')
        parser.add_argument('-ac', '--acccoin', type=float, dest='acc_coin', default=0.15,
                            help='Accuracy to determine if contour is a coin or not.')
        parser.add_argument('-ar', '--accrect', type=float, dest='acc_rect', default=0.01,
                            help='Accuracy to determine if contour is a rectangle or not.')
        arg = parser.parse_args()

        if arg.svm is True:
            svm(arg)

        elif arg.new is True:
            if arg.image is None:
                print('Path of image is not set')
            elif arg.refa is None:
                print('Reference A is not set')
            elif arg.refb is None:
                print('Reference B is not set')
            else:
                new(arg)
        elif arg.count is True:
            if arg.image is None:
                print('Path of image is not set')
            elif arg.refa is None:
                print('Reference A is not set')
            elif arg.refb is None:
                print('Reference B is not set')
            else:
                count(arg)

        elif arg.count is True:
            pass

        else:
            print('No program chosen.')
    except:
        raise
