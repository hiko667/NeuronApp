import pandas as pd
import numpy as np
from scipy.io import arff
import random

class Machine():
    def __init__(self):
        self.w1 = random.randint(-100, 100)
        self.w2 = random.randint(-100, 100)
        self.bias = random.randint(-100, 100)
        self.t = 0
        self.epoch = 0
        self.data = None
        self.class_val = None
    def load_data(self, path, test_size = 0.33):
        data_arff, metadata = arff.loadarff(path)
        df = pd.DataFrame(data_arff)
        df['class'] = df['class'].str.decode('utf-8').astype(int)

        self.data = df[['x1', 'x2']].to_numpy()
        self.class_val = df[['class']].to_numpy()
        print(self.class_val)

    def learn(self) -> None:
        if self.data is None:
            raise Exception("Data has not been loaded")
        x1 = float(self.data[self.t][0])
        x2 = float(self.data[self.t][1])
        d = int(self.class_val[self.t][0])
        result = self.w1 * x1 + self.w2 * x2 + self.bias
        if result < 0 and d == 0 or result > 0 and d == 0:
            pass
        else:
            pass
        self.t += 1
