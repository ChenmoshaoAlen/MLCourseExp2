import numpy as np
import pandas as pd
import sklearn.datasets as sd
import sklearn.model_selection as sms
import matplotlib.pyplot as plt
import random

X_train, y_train = sd.load_svmlight_file('a9a.txt',n_features = 123)
X_valid, y_valid = sd.load_svmlight_file('a9a.t.txt',n_features = 123)
