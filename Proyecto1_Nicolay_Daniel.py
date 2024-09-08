import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = os.getcwd()
print(path)

data = pd.read_csv("SeoulBikeData_utf8.csv")
print(data.head())