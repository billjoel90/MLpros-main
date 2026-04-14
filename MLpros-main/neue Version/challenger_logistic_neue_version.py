import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
from mpl_toolkits.mplot3d import Axes3D

# Datensatz laden
df = pd.read_csv("orings.csv")
display(df.head())

print("Spaltennamen:", df.columns.tolist())