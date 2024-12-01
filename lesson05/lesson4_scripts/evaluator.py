import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

# class to evaluate models
class EvalClassifier:
    def __init__(self, model):
        self.model = model
    
    def get_confusion_matrix(self, y_true, y_pred, classes):
        class_idx = {label:index for index, label in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype = int)

        for true_label, pred_label in zip(y_true, y_pred):
            cm[class_idx[true_label]][class_idx[pred_label]] += 1
        return cm

    def plot_confusion_matrix(self, cm, classes):
        plt.figure(figsize = (9, 7))
        sbn.heatmap(data=cm, annot=True, cmap="Blues", fmt='.4f', cbar=True)
        # add legends with class names
        plt.xticks(ticks=np.arange(len(classes))+0.5, labels=classes, rotation=45)
        plt.yticks(ticks=np.arange(len(classes))+0.5, labels=classes, rotation=45)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.show()
    
    def calculate_metrics_by_class(self, confusion_matrix, classes):
        metrics_table = []

        for i in range(len(classes)):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            tn = np.sum(confusion_matrix) - (tp + fp + fn)

            # calculate the metrics
            accuracy = (tp + tn)/(tp + fp + fn + tn)
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            specificity = tn/(tn + fp)
            f1_score = 2 * (precision * recall)/(precision + recall)

            metrics_table.append([accuracy, precision, recall, specificity, f1_score])
        
        # generate data frame of metrics
        metrics_df = pd.DataFrame(metrics_table, index = classes, 
                                  columns = ["accuracy", "precision", "recall", "specificity", "f1-score"])

        return metrics_df
    
    def plot_metrics_by_class(self, metrics, title):
        plt.figure(figsize = (9, 7))
        sbn.heatmap(metrics, annot = True, fmt = ".4f", cmap = "Blues", cbar = True)
        # add legends with class names
        plt.xticks(ticks=np.arange(len(metrics.columns))+0.5, labels=metrics.columns, rotation=45)
        plt.yticks(ticks=np.arange(len(metrics.index))+0.5, labels=metrics.index, rotation=45)
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.title(title)
        plt.show()
        
    def calculate_metrics_xall(self, confusion_matrix, metrics):
        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
        precision = metrics["precision"].mean()
        recall = metrics["recall"].mean()
        specificity = metrics["specificity"].mean()
        f1_score = metrics["f1-score"].mean()

        metrics_table = []
        metrics_table.append([accuracy, precision, recall, specificity, f1_score])

        # generate data frame of metrics
        metrics_df = pd.DataFrame(metrics_table, index = ["Mean-metrics"],
                                  columns = ["accuracy", "precision", "recall", "specificity", "f1-score"])
        
        return metrics_df
    