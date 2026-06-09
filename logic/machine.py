import pandas as pd
import numpy as np
from scipy.io import arff
import random

class Machine():
    def __init__(self):
        self.tolerance = 0
        self.w1 = random.randint(-100, 100)
        self.w2 = random.randint(-100, 100)
        self.bias = random.randint(-100, 100)
        self.t = 0
        self.epoch = 0
        self.data = None
        self.class_val = None
        self.error_scores = []
    def load_data(self, path, test_size = 0.33):
        self.t = 0
        self.epoch = 0
        self.error_scores = []

        data_arff, metadata = arff.loadarff(path)
        df = pd.DataFrame(data_arff)
        df['class'] = df['class'].str.decode('utf-8').astype(int)

        self.data = df[['x1', 'x2']].to_numpy()
        self.class_val = df[['class']].to_numpy()
        self.calculate_error()
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def learn(self) -> None:
        if self.data is None:
            raise Exception("Data has not been loaded")
        x1 = float(self.data[self.t][0])
        x2 = float(self.data[self.t][1])
        d = int(self.class_val[self.t][0])
        if d == 0 : d = -1
        result = self.w1 * x1 + self.w2 * x2 + self.bias
        if result < 0 and d < 0 or result > 0 and d > 0:
            pass
        else:
            self.w1 = self.w1 + x1 * d
            self.w2 = self.w2 + x2 * d
        if self.t < self.class_val.shape[0] - 1:
            self.t += 1
        else:
            self.t = 0
            self.epoch += 1
    def calculate_error(self) -> None:
        wrong_predictions = 0
        total_samples = self.data.shape[0]
        
        for i in range(total_samples):
            x1 = float(self.data[i][0])
            x2 = float(self.data[i][1])
            d = int(self.class_val[i][0])
            if d == 0: d = -1
            
            result = self.w1 * x1 + self.w2 * x2 + self.bias
            
            if (result >= 0 and d < 0) or (result < 0 and d > 0):
                wrong_predictions += 1
                
        self.error_scores.append(wrong_predictions / total_samples)
    def learn_in_steps(self, steps : int) -> None:
        for i in range(steps):
            self.learn()

