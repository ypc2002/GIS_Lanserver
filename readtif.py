import math
import numpy as np
import os
from osgeo import gdal

fileName = "D:\\MS\\data\\beijing\\huairou2019.tif"
dataset = gdal.Open(fileName)
# projection = dataset.GetProjection()  # 投影
DN = dataset.ReadAsArray()
x = DN.shape[0]
y = DN.shape[1]
print(x, y)
Level1 = 0
Level2 = 0
Level3 = 0
Level4 = 0
Level5 = 0
for i in range(0, x):
    for j in range(0, y):
        if DN[i][j] < 2:
            Level1 = Level1 + 1
        if 2 <= DN[i][j] < 5:
            Level2 = Level2 + 1
        if 5 <= DN[i][j] < 10:
            Level3 = Level3 + 1
        if 10 <= DN[i][j] < 25:
            Level4 = Level4 + 1
        if DN[i][j] > 25:
            Level5 = Level5 + 1
print(Level1, Level2, Level3, Level4, Level5)

# transform = dataset.GetGeoTransform()  # 几何信息
