"""Program to create and save a new Support Vector Machine."""

from typing import Any

from sklearn import svm as svma
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import coincount_lib as ccl


def svm(arg: Any = None) -> None:
    """
    Program to create and save a new Support Vector Machine.

    Parameters:
        arg (Any): Arguments from argparse.
    """
    try:
        if arg is None:
            model_save = './svm/model.sav'
        else:
            model_save = arg.model

        data_target = ccl.svm_readcsv('./svm/csvfiles/1cnts.csv', 1)
        data_target = ccl.svm_readcsv('./svm/csvfiles/2cnts.csv', 2, data_target[0], data_target[1])
        data_target = ccl.svm_readcsv('./svm/csvfiles/5cnts.csv', 5, data_target[0], data_target[1])
        data_target = ccl.svm_readcsv('./svm/csvfiles/10cnts.csv', 10, data_target[0], data_target[1])
        data_target = ccl.svm_readcsv('./svm/csvfiles/20cnts.csv', 20, data_target[0], data_target[1])

        X_train, X_test, y_train, y_test = train_test_split(data_target[0], data_target[1],
                                                            test_size=0.25)  # split dataset

        model = svma.SVC(kernel='linear')  # create model

        model.fit(X_train, y_train)  # train model

        y_pred = model.predict(X_test)  # predict for test values

        print(classification_report(y_test, y_pred))  # classification report

        ccl.svm_writemodel(model_save, model)  # safe model
    except:
        raise


if __name__ == '__main__':
    try:
        svm()
    except:
        raise
