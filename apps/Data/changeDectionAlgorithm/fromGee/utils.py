import  math
import os
import numpy as np
import matplotlib.pyplot as plt
#

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

def resize(imagePatchesPath:str):
    imagePatchesName = os.listdir(imagePatchesPath)
    print(imagePatchesName)
    patchPath=imagePatchesPath + '/' + imagePatchesName[0]
    patch = cv2.imread(patchPath)
    h, w = patch.shape[:2]
    for imagePatchName in imagePatchesName:
        patchPath = imagePatchesPath + '/' + imagePatchName
        patch = cv2.imread(patchPath)
        img_resize=cv2.resize(patch,(h,w),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(patchPath,img_resize)



def mergeImage(imagePatchesPath:str,outImageName:str,bandsNums:int=3):
    try:
        resize(imagePatchesPath)
        imagePatchesName=os.listdir(imagePatchesPath)
        print(imagePatchesName)
        lastW_index, lastH_index = imagePatchesName[-1][:-4].split('_')[-2:]
        # print('hight',lastH_index)
        # print('weidht',lastW_index)
        lastH_index = int(lastH_index)
        lastW_index = int(lastW_index)
        if bandsNums==3:
            flag=1
            h, w, d = cv2.imread(imagePatchesPath + '/' + imagePatchesName[0],flag).shape
            allImageArray = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w, d))
        elif bandsNums == 4:
            flag = -1
            h, w= cv2.imread(imagePatchesPath + '/' + imagePatchesName[0], flag).shape
            allImageArray = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
        elif bandsNums==1:
            flag=0
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
        cv2.imwrite(imagePatchesPath+'/'+outImageName,allImageArray)
        # color = ['white', 'green']
        # cmap1 = m.colors.ListedColormap(color)
        # im = plt.imshow(allImageArray,cmap='Set1',aspect=1, interpolation="none")
        # fig.colorbar(im, ticks=range(2), orientation="horizontal")
        # plt.show()
        return True
    except Exception as e:
        print('合并错误')
        print(e.args)
        print(traceback.print_exc())
        return False
import matplotlib as m
if __name__=='__main__':
    # mergeImage('D:\\slw\\s-law-server\\Satellite-law-enforcement-Server\\static\\tempImage\\landUseImage\\ypc\\1664703799.6769438',"g11sg.png",1)
    # mergeImage('D:/slw/s-law-server/Satellite-law-enforcement-Server/static/tempImage/landChangeImage/ypc/CVA','fsfsfs.png',1)
    # resize('D:\\slw\\s-law-server\\Satellite-law-enforcement-Server\\static\\tempImage\\landUseImage\\ypc\\1664703799.6769438')
    tifDir='D:\\slw\\s-law-server\\Satellite-law-enforcement-Server\\static\\tempImage\\landChangeImage\\ypc\\1664715960.5896826\\PCA\\'
    name='dd.png'
    mergeImage(tifDir, name, 1)
    outtif_LUCC_path = tifDir + name
    a = cv2.imread(outtif_LUCC_path, 0)

    tifname = 'result.png'

    outtif_LUCC = r'D:\slw\s-law-server\Satellite-law-enforcement-Server\static\tempImage\landChangeImage\ypc\1664715960.5896826\PCA'+'\\'+tifname
    colors = ['white','black']
    cmap = m.colors.ListedColormap(colors)

    plt.imsave(arr=a, fname=outtif_LUCC, cmap=cmap)
    fig, ax = plt.subplots()
    im = plt.imshow(a,cmap=cmap,aspect=1,interpolation="none")
    fig.colorbar(im, ticks=range(2), orientation="horizontal")
    plt.show()