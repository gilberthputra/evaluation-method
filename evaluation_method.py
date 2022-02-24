import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import math
    
def confusion_matrix(actual, predictions, draw = False):
    matrix_df = pd.crosstab(index = actual, 
                               columns = predictions, 
                               rownames = ['Actual'], 
                               colnames = ['Predicted'])

    if (draw):
        sns.heatmap(matrix_df, annot = True)

    return matrix_df

def compute_accuracy(matrix_df, predictions):
    accuracy = matrix_df.to_numpy().trace() * 100 / len(predictions)
    return accuracy

def recall_precision(matrix_df, level = 'micro'):
    arr = matrix_df.to_numpy()

    rows = np.sum(arr, axis = 1)
    columns = np.sum(arr, axis = 0)

    diagonals = np.diag(arr)

    if (level == 'micro'):
        recall = sum(diagonals) * 100 / sum(rows)
        precision = sum(diagonals) * 100 / sum(columns)
    elif (level == 'macro'):
        recall = sum((diagonals / rows)) * 100 / len(diagonals)
        precision = sum((diagonals / columns)) * 100 / len(diagonals)
    elif (level == 'weighted'):
        recall = sum((diagonals / rows) * (rows / np.sum(arr))) * 100
        precision = sum((diagonals / columns) * (rows / np.sum(arr))) * 100

    return recall, precision

def compute_f_score(recall, precision, beta = 1.0):
    numerator = (1 + math.pow(beta, 2)) * recall * precision
    denominator = (math.pow(beta, 2) * precision) + recall

    return numerator/denominator

def evaluate(actual, prediction, level = 'micro', beta = 1.0, draw = False):
    matrix_df = confusion_matrix(actual, prediction, draw = draw)
    accuracy = compute_accuracy(matrix_df, prediction)
    recall, precision = recall_precision(matrix_df, level)
    f_score = compute_f_score(recall, precision, beta)
    print(f"{level} | F-beta : {beta} \n \
          Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F-Score:{f_score:.3f} \n")