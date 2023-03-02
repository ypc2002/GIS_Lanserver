import json
import  math
import numpy as np
import os
from osgeo import gdal

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
# print(mercatorTolonlat(a))


def getGeoJsonGeometry(GeoJson:json)->str:
    '''
    :param GeoJson:
    :return: geometry or geometries
    '''
    if GeoJson['type']=='FeatureCollection':
        # return str([feature['geometry'] for feature in GeoJson['features']])
        return str(GeoJson['features'][0]['geometry'])
    elif GeoJson['type']=='Feature':
        return str(GeoJson['geometry'])
    elif  GeoJson['type'] in ["Point","MultiPoint","LineString","MultiLineString","Polygon","MultiPolygon"]:
        return str(GeoJson)
    elif GeoJson['type']=="GeometryCollection":
        return str(GeoJson['geometries'])
    else:
        return False



import cv2
import traceback
def resize(imagePatchesPath:str):
    imagePatchesName = os.listdir(imagePatchesPath)
    print(imagePatchesName)
    patchPath=imagePatchesPath + '/' + imagePatchesName[0]

    patch = cv2.imread(patchPath)

    print(patch)
    h, w = patch.shape[:2]
    for imagePatchName in imagePatchesName:
        patchPath = imagePatchesPath + '/' + imagePatchName

        patch = cv2.imread(patchPath)

        img_resize=cv2.resize(patch,(h,w),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(patchPath,img_resize)


def resize_tif(imagePatchesPath:str):
    imagePatchesName = os.listdir(imagePatchesPath)
    print(imagePatchesName)
    print("dad")
    patchPath = imagePatchesPath + '/' + imagePatchesName[0]

    a, prj, trans = readTif(patchPath)
    d, h, w = a.shape
    #h 行 w 列  gdal读出的位 列*行
    print(h,w,a.shape)
    for imagePatchName in imagePatchesName:
        patchPath = imagePatchesPath + '/' + imagePatchName
        print("dadfsff")
        patch, prj, trans = readTif(patchPath)
        patch = np.transpose(patch, (1, 2, 0))
        # print(patch)
        print(patch.shape)
        img_resize = cv2.resize(patch, (w, h), interpolation=cv2.INTER_CUBIC)
        print(img_resize.shape)
        img_resize=np.transpose(patch, (2, 0, 1))
        print(img_resize.shape)
        to_tif(img_resize, prj, trans, patchPath)

import matplotlib as m
import matplotlib.pyplot as plt

def mergeImage(imagePatchesPath:str,outImageName:str,bandsNums:int=3):
    try:
        print("resiz success")
        path=imagePatchesPath + '/'+outImageName
        imagePatchesName=os.listdir(imagePatchesPath)
        print(imagePatchesName)

        lastW_index, lastH_index = imagePatchesName[-1][:-4].split('_')[-2:]

        lastH_index = int(lastH_index)
        lastW_index = int(lastW_index)
        if bandsNums==3:
            flag=1
            # resize_tif(imagePatchesPath)
            a,prj,trans=readTif(imagePatchesPath + '/' + imagePatchesName[0])
            d,h,w=a.shape
            # print(a)
            print(a.shape,d,h,w)
            allImageArray = np.zeros((d,(lastH_index + 1) * h, (lastW_index + 1) * w))
            r_merge = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
            g_merge = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
            b_merge = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
        elif bandsNums == 4:
            flag = -1
            h, w= cv2.imread(imagePatchesPath + '/' + imagePatchesName[0], flag).shape
            allImageArray = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
        elif bandsNums==1:
            flag=0
            resize(imagePatchesPath)
            h, w= (cv2.imread(imagePatchesPath + '/' + imagePatchesName[0],flag)).shape
            allImageArray = np.zeros(((lastH_index + 1) * h, (lastW_index + 1) * w))
        if bandsNums==3:

            for imagePatchName in imagePatchesName:
                patchPath=imagePatchesPath+'/'+imagePatchName
                patch,p,t=readTif(patchPath)
                # print(patch)
                r=patch[0]
                g=patch[1]
                b=patch[2]
                r=(r/(np.max(r)-np.min(r)))*255
                b = (b / (np.max(b) - np.min(b)))*255
                g = (g / (np.max(g) - np.min(g)))*255
                startW_index,startH_index=patchPath[:-4].split('_')[-2:]
                startH_index = int(startH_index)
                startW_index = int(startW_index)

                r_merge[startH_index * h:(startH_index + 1) * h, startW_index * w:(startW_index + 1) * w] = r
                g_merge[startH_index * h:(startH_index + 1) * h, startW_index * w:(startW_index + 1) * w] = g
                b_merge[startH_index * h:(startH_index + 1) * h, startW_index * w:(startW_index + 1) * w] = b
            allImageArray[0]=r_merge
            allImageArray[1]=g_merge
            allImageArray[2]=b_merge
            allImageArray=np.transpose(allImageArray,(1,2,0))
            allImageArray = TwoPercentLinear(allImageArray)
            cv2.imwrite(path,allImageArray.astype(int))
            # to_tif(allImageArray,prj,trans,path)
        else:
            for imagePatchName in imagePatchesName:
                patchPath = imagePatchesPath + '/' + imagePatchName
                patch = cv2.imread(patchPath, flag)
                startW_index, startH_index = patchPath[:-4].split('_')[-2:]
                startH_index = int(startH_index)
                startW_index = int(startW_index)
                allImageArray[startH_index * h:(startH_index + 1) * h, startW_index * w:(startW_index + 1) * w] = patch
            cv2.imwrite(imagePatchesPath + '/' + outImageName, allImageArray)

        return True
    except Exception as e:
        print('合并错误')
        print(e.args)
        print(traceback.print_exc())
        return False

def to_tif(mat, projection, tran, mapfile):
        """
        将数组转成tif文件写入硬盘
        :param mat: 数组
        :param projection: 投影信息
        :param tran: 几何信息
        :param mapfile: 文件路径
        :return:
        """

        row = mat.shape[0]  # 矩阵的行数
        columns = mat.shape[1]  # 矩阵的列数


        dim_z = mat.shape[2]  # 通道数

        driver = gdal.GetDriverByName('GTiff')  # 创建驱动
        # 创建文件
        dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_UInt16)
        dst_ds.SetGeoTransform(tran)  # 设置几何信息
        dst_ds.SetProjection(projection)  # 设置投影

        # 将数组的各通道写入tif图片
        for channel in np.arange(dim_z):
            map = mat[:, :, channel]
            dst_ds.GetRasterBand(int(channel + 1)).WriteArray(map)

        dst_ds.FlushCache()  # 写入硬盘
        dst_ds = None

def TwoPercentLinear(image, max_out=255, min_out=0):
    r,g,b = cv2.split(image)#分开三个波段
    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 98)#取得98%直方图处对应灰度
        low_value = np.percentile(gray, 2)#同理
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)#线性拉伸嘛
        return processed_gray
    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((r_p, g_p, b_p))#合并处理后的三个波段
    return np.uint8(result)

def readTif(fileName:str):

    dataset = gdal.Open(fileName)
    projection = dataset.GetProjection()  # 投影
    transform = dataset.GetGeoTransform()  # 几何信息
    if dataset == None:
        print(fileName + "文件无法打开")
        return
    return dataset.ReadAsArray(),projection,transform

if __name__=="__main__":
    mergeImage(r'D:\Satellite-law-enforcement-Server\Satellite-law-enforcement-Server\static\tempImage\sourceImage\1\1665058746.0593429\\','dd.png',3)
    # a=cv2.imread(r'D:\Satellite-law-enforcement-Server\Satellite-law-enforcement-Server\static\tempImage\sourceImage\1\1664975275.422603\remote_image_0_0')
    # print(np.unique(a))