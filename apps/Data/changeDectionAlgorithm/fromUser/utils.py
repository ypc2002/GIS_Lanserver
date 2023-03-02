import numpy as np
from osgeo import gdal
import cv2





#读取tif为矩阵
def readTif(filename:str)->np.array:
    dataset=gdal.Open(filename)
    if dataset==None:
        print(filename+"文件无法打开")
        return
    else:
        return dataset.ReadAsArray()

#按照矢量边界裁剪


def readPng_Jpg(filename:str)->np.array:
    img=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    return np.array(img)


def readImage(filePath):
    if 'tif' in filePath:
        Image=readTif(filePath)
        return Image
    elif 'png' in filePath:
        Image =readPng_Jpg(filePath)
        return Image


def TwoDtoOneD(Image: np.array, imgType:str):
    ImageShape = Image.shape
    result = []
    if imgType == 'tif':
        for x in range(ImageShape[1]):
            for y in range(ImageShape[2]):
                result.append(Image[:, x, y])
    elif imgType in ['jpg', 'png','oneBand']:
        for x in range(ImageShape[0]):
            for y in range(ImageShape[1]):
                result.append(Image[x, y])
    return np.array(result)

'''将图像一维转二维'''
def OneDtoTwoD(ImageDataList: np.array, aimShape):
    '''
    :param ImageDataList:
    :param aimShape: [width,height]
    :return:
    '''
    result = []
    for y in range(aimShape[1]):
        row = []
        for x in range(aimShape[0]):
            row.append(ImageDataList[y * aimShape[0] + x])
        result.append(row)
    return np.array(result)

'''合并图像'''
def mergeImage(imagePathList:list):
    #先判断合并图片的类型（每张图片大小一致）

    #先创建一个合并后大小的数组，然后合并

    #合并非tif类型
    for imagepath in imagePathList:
        image=readPng_Jpg(imagepath)
        imagepath=imagepath.split('_')
        x,y=imagepath[1],imagepath[2]

    #合并tif
    for imagepath in imagePathList:
        image=gdal.Open(imagepath)
        imagepath=imagepath.split('_')
        x,y=imagepath[1],imagepath[2]



'''二进制图片流转为矩阵'''
def ImageB2np(ImageB):

    return





#得到文件夹中各个图片的路径



#保存为jpg、png或者tif



# if __name__=='__main__':
    # from osgeo import gdal
    #
    # input_shape = r"G:\GISData\SLEData\土地利用湖北\湖北省.shp"
    # output_raster = r'G:\GISData\SLEData\土地利用湖北\2019-50r'
    # # tif输入路径，打开文件
    # input_raster = r"E:\50R_20190101-20200101.tif"
    # # 矢量文件路径，打开矢量文件
    # input_raster = gdal.Open(input_raster)
    # # 开始裁剪，一行代码，爽的飞起
    # ds = gdal.Warp(output_raster,
    #                input_raster,
    #                format='GTiff',
    #                cutlineDSName=input_shape,
    #                # cutlineWhere="FIELD = 'whatever'",#矢量文件筛选条件
    #                cropToCutline=True,# 保证裁剪后影像大小跟矢量文件的图框大小一致（设置为False时，结果图像大小会跟待裁剪影像大小一样，则会出现大量的空值区域）
    #                dstNodata=0)#掩膜像元的取值
    # filePath1 = './testData/image0_0.tif'
    # img=cv2.cvLoadimage(filePath1)
    # img=np.array(img)
    # print(img.shape)