import cv2
import matplotlib as m
from matplotlib import pyplot as plt

a = cv2.imread(r'D:\Satellite-law-enforcement-Server\Satellite-law-enforcement-Server\static\tempImage\landUseImage\1\1665046430.7513893\1_landuseImage.png',0)

# print(a)

water = a[a==0]
print(water)
colors = [(191, 101, 33), (199, 131, 183), (120, 80, 173), (134, 121, 58), (28, 106, 203), (33, 61, 166),
                      (60, 216, 229), (91, 101, 113),
                      (77, 97, 31)]
color=colors
cmap = m.colors.ListedColormap(color)
# plt.imsave(arr=a, fname=outtif_LUCC, cmap=cmap, vmin=-0.5, vmax=8.5)