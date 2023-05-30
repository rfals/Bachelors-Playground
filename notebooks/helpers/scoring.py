import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, \
    accuracy_score, precision_score, recall_score, confusion_matrix, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns


class ensambleEvaluator:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def eval(self, model):
        y_pred = model.predict(self.X_test)
        y_pred_train = model.predict(self.X_train)

        print(confusion_matrix(self.y_test, y_pred))
        print("Test_Set")
        print(classification_report(self.y_test,y_pred))
        print("Train_Set")
        print(classification_report(self.y_train,y_pred_train))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()

    def train_val(self, model):
        y_pred = model.predict(self.X_test)
        y_pred_train = model.predict(self.X_train)
        scores = {"train_set": {"Accuracy" : accuracy_score(self.y_train, y_pred_train),
                        "Precision" : precision_score(self.y_train, y_pred_train),
                        "Recall" : recall_score(self.y_train, y_pred_train),                          
                        "f1" : f1_score(self.y_train, y_pred_train)},

            "test_set": {"Accuracy" : accuracy_score(self.y_test, y_pred),
                        "Precision" : precision_score(self.y_test, y_pred),
                        "Recall" : recall_score(self.y_test, y_pred),                          
                        "f1" : f1_score(self.y_test, y_pred)}}
    
        return pd.DataFrame(scores)

    