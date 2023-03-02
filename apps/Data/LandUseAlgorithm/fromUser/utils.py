import numpy as np
from osgeo import gdal
import cv2
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset


import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.Up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '_vis'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }







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