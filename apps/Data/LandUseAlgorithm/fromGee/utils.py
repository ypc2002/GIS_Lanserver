import  math
import os
import numpy as np
import matplotlib as m
import matplotlib.pyplot as plt
import cv2
def mercatorTolonlat(mercator):
    j = 0
    for i in mercator[0]:
        # print(i)
        x = float(i[0])
        y = float(i[1])
        x = x / 20037508.34 * 180
        y = y / 20037508.34 * 180
        y = 180 / math.pi * (2 * math.atan(math.exp(y * math.pi / 180)) - math.pi / 2)
        mercator[0][j][0]=x
        mercator[0][j][1]=y
        j=j+1
    return mercator


import cv2
import traceback

def resize(imagePatchesPath:str,flag):
    imagePatchesName = os.listdir(imagePatchesPath)
    # print(imagePatchesName)
    patchPath=imagePatchesPath + '/' + imagePatchesName[0]
    patch = cv2.imread(patchPath,flag)
    # print('patch',patch)
    h, w = patch.shape[:2]
    print('resize')
    for imagePatchName in imagePatchesName:
        patchPath = imagePatchesPath + '/' + imagePatchName
        patch = cv2.imread(patchPath,flag)
        img_resize=cv2.resize(patch,(h,w),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(patchPath,img_resize)



def mergeImage(imagePatchesPath:str,outImageName:str,bandsNums:int=3):
    try:
        imagePatchesName=os.listdir(imagePatchesPath)
        print('merge')
        print(imagePatchesName)
        lastW_index, lastH_index = imagePatchesName[-1][:-4].split('_')[-2:]

        print('hight',lastH_index)
        print('weidht',lastW_index)
        lastH_index = int(lastH_index)
        lastW_index = int(lastW_index)
        print(imagePatchesPath + '/' + imagePatchesName[0])
        if bandsNums==3:
            flag=1
            resize(imagePatchesPath,flag)
            h, w, d = cv2.imread(imagePatchesPath + '/' + imagePatchesName[0],flag).shape
            allImageArray = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w, d))
        elif bandsNums == 4:
            flag = -1
            resize(imagePatchesPath,flag)
            h, w= cv2.imread(imagePatchesPath + '/' + imagePatchesName[0], flag).shape
            allImageArray = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
        elif bandsNums==1:
            flag=0
            resize(imagePatchesPath,flag)
            h, w= (cv2.imread(imagePatchesPath + '/' + imagePatchesName[0],flag)).shape
            allImageArray = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
        for imagePatchName in imagePatchesName:
            patchPath=imagePatchesPath+'/'+imagePatchName
            patch=cv2.imread(patchPath,flag)
            startW_index,startH_index=patchPath[:-4].split('_')[-2:]
            startH_index = int(startH_index)
            startW_index = int(startW_index)
            allImageArray[startH_index*h:(startH_index+1)*h,startW_index*w:(startW_index+1)*w]=patch
        # import matplotlib as m
        # fig, ax = plt.subplots()
        print(allImageArray)
        cv2.imwrite(imagePatchesPath+'/'+outImageName,allImageArray)
        # color = ['white', 'green']
        # cmap1 = m.colors.ListedColormap(color)
        # im = plt.imshow(allImageArray,cmap=cmap1, vmin=0.15,vmax=1,aspect=1, interpolation="none")
        # fig.colorbar(im, ticks=range(2), orientation="horizontal")
        # plt.show()
        return allImageArray
    except Exception as e:
        print('合并错误')
        print(e.args)
        print(traceback.print_exc())
        return False

if __name__=='__main__':
    tifDir = r'D:/slw/s-law-server/Satellite-law-enforcement-Server/static/tempImage/landUseImage/ypc/1664721572.5760622/water'
    name = 'dd.png'
    b=mergeImage(tifDir, name, 4)
    outtif_LUCC_path = tifDir +'/'+ name
    # a = cv2.imread(outtif_LUCC_path, 0)
    # print(a)
    tifname = 'result.png'

    outtif_LUCC = 'D:\\slw\\s-law-server\\Satellite-law-enforcement-Server\\static\\tempImage\\landUseImage\\ypc\\1664721572.5760622\\water' + '\\' + tifname
    colors = ['white', 'blue']
    cmap = m.colors.ListedColormap(colors)

    plt.imsave(arr=b, fname=outtif_LUCC,vmin=-0.5,vmax=0.38, cmap=cmap)
    fig, ax = plt.subplots()
    im = plt.imshow(b, cmap=cmap, aspect=1,vmin=-0.5,vmax=0.38,interpolation="none")
    fig.colorbar(im, ticks=range(2), orientation="horizontal")
    plt.show()