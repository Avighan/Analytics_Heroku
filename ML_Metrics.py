import pandas as pd
import numpy as np
import pdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix, \
    roc_curve, log_loss, balanced_accuracy_score, hamming_loss, hinge_loss, jaccard_score, explained_variance_score, \
    max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.preprocessing import LabelBinarizer


class Metrics:
    def __init__(self, y_test, type=None, y_pred=None):

        self.y_pred = y_pred
        self.y_test = y_test

        self.type = type

        self.metrics = {
            'Accuracy': accuracy_score,
            'Recall': recall_score,
            'Precision': precision_score,
            'F1score': f1_score,
            'Confusion': confusion_matrix,
            'AUC': roc_auc_score,
            'ROC': roc_curve,
            'logloss': log_loss,
            'balancedaccuracy': balanced_accuracy_score,
            'hammingloss': hamming_loss,
            'hingeloss': hinge_loss,
            'jaccardscore': jaccard_score,
            'mae': mean_absolute_error,
            'mse': mean_squared_error,
            'mape': self.mape,
            'mpe': self.mpe,
            'rmse': mean_squared_error,
            'medianabserror': median_absolute_error,
            'maxerror': max_error,
            'r2score': r2_score,
            'mean_squared_log_error': mean_squared_log_error,
            'explained_variance_score': explained_variance_score,
            'multiclass_roc_auc_score': self.multiclass_roc_auc_score,
            'multiclass_recall': self.multiclass_recall,
            'multiclass_precision': self.multiclass_precision,
            'multiclass_f1score': self.multiclass_f1score
            # 'adjusted_r2':self.adjusted_r2
        }

    def select_metrics(self, metrics_sel):
        self.metrics_sel = metrics_sel
        self.metric = self.metrics[metrics_sel]
        return self.metric

    def metrics_solve(self, estimator=None, test_x=None, metric=None, **kwargs):
        if estimator is not None and test_x is not None:
            self.y_pred = estimator.predict(test_x)
        if self.metrics_sel == 'rmse':
            if metric is not None:
                return np.sqrt(metric(self.y_test, self.y_pred, **kwargs))
            else:
                return np.sqrt(self.metric(self.y_test, self.y_pred, **kwargs))
        else:
            if metric is not None:
                return np.sqrt(metric(self.y_test, self.y_pred, **kwargs))
            else:
                return self.metric(self.y_test, self.y_pred, **kwargs)

    def get_metrics_keyword(self):
        return self.metrics

    def mape(self, y_test, y_pred, estimator=None, test_x=None):
        if estimator is not None and test_x is not None:
            self.y_pred = estimator.predict(test_x)
        mape = np.mean(np.abs((y_test - y_pred) / y_test) * 100)
        return mape

    def mpe(self, y_test, y_pred, estimator=None, test_x=None):
        if estimator is not None and test_x is not None:
            self.y_pred = estimator.predict(test_x)
        mpe = np.mean((y_test - y_pred) / y_test) * 100
        return mpe

    def adjusted_r2(self, y_test, y_pred, estimator=None, test_x=None, X=None):
        if X is None:
            X = self.y_test
        else:
            pass
        if estimator is not None and test_x is not None:
            self.y_pred = estimator.predict(test_x)
        self.metric = self.select_metrics('r2score')
        r_squared = self.metrics_solve()
        self.adjusted_r_squared = 1 - (1 - r_squared) * (len(self.y_test) - 1) / (len(self.y) - X.shape[1] - 1)
        return self.adjusted_r_squared

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def multiclass_recall(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return recall_score(y_test, y_pred, average=average)

    def multiclass_precision(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return precision_score(y_test, y_pred, average=average)

    def multiclass_f1score(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return f1_score(y_test, y_pred, average=average)

    def get_metric_list(self):
        return list(self.metrics.keys())