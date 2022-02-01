import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import math

class evaluate:
    def __init__(self, predictions, actual):
        self.predictions = predictions
        self.actual = actual
        
        self.accuracy = None
        self.recall = None
        self.precision = None
        self.f_score = None
    
    def confusion_matrix(self, draw = False):
        matrix_df = pd.crosstab(index = self.actual, 
                                   columns = self.predictions, 
                                   rownames = ['Actual'], 
                                   colnames = ['Predicted'])
        
        if (draw):
            sns.heatmap(matrix_df, annot = True)
            
        return matrix_df
    
    def compute_accuracy(self):
        self.accuracy = self.confusion_matrix().to_numpy().trace() * 100 / len(self.predictions)
    
    def recall_precision(self, level = 'micro'):
        arr = self.confusion_matrix().to_numpy()
        
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
        
        self.recall = recall
        self.precision = precision
    
    def compute_f_score(self, beta = 1.0):
        numerator = (1 + math.pow(beta, 2)) * self.recall * self.precision
        denominator = (math.pow(beta, 2) * self.precision) + self.recall
        
        self.f_score = numerator/denominator
    
    def result(self, level = 'micro', beta = 1.0):
        self.compute_accuracy()
        self.recall_precision(level)
        self.compute_f_score(beta)
        print(f"{level}\n Accuracy: {self.accuracy}, Recall: {self.recall}, Precision: {self.precision}, F1-Score:{self.f_score}")