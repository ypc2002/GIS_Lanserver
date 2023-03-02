import sys,os
# folderPath=os.getcwd()
# childrenFolderPath=os.listdir(folderPath)
# sys.path.extend([folderPath+'\\'+childrenFolderPath[i] for i in range(len(childrenFolderPath)-1)])

sys.path.append(os.getcwd()+'/apps/Data')

from extractBoundary.main_regularization import png2png

import argparse
import logging
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .utils import plot_img_and_mask,NestedUNet
import torch.utils.data as D


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m',
                        default='./apps/Data/LandUseAlgorithm/fromUser/models',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--batchSize', '-i',type=int, default=1, help='batch-size')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    # parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def caijian(image:np.array, size_w=256, size_h=256, step=256):  # step为步长，设置为256即相邻图片重叠50%
    '''
    :param image1Path: 裁剪图像的路径
    :param size_w: 裁剪成小块的宽
    :param size_h: 裁剪成小块的高
    :param step: 各个小块之间的步长
    :return: 小块组成的数组，每个小块左上角对应的行列元组数组
    '''
    patchList=[]
    patchNameList=[]
    size = image.shape
    # if size[0]<=size_w and size[1]<=size_h:
    #     patchList.append(image)
    #     patchNameList.append((0,0))
    # else:
    i = 0
    for h in range(0, size[1], step):
        star_h = h  # star_h表示起始高度，从0以步长step开始循环
        for w in range(0, size[2], step):
            star_w = w  # star_w表示起始宽度，从0以步长step开始循环
            end_h = star_h + size_h  # end_h是终止高度

            if end_h > size[1]:  # 如果边缘位置不够size_h的列
                # 以倒数size_h形成裁剪区域
                star_h = size[1] - size_h
                end_h = star_h + size_h
                i = i - 1
            end_w = star_w + size_w  # end_w是中止宽度
            if end_w > size[2]:  # 如果边缘位置不够size_w的行
                # 以倒数size_w形成裁剪区域
                star_w = size[2] - size_w
                end_w = star_w + size_w
                i = i - 1
            imageCropped = image[:,star_h:end_h, star_w:end_w]  # 执行裁剪操作

            i = i + 1
            patch_img = (star_h,star_w)
            patchList.append(imageCropped)
            patchNameList.append(patch_img)
        print(h)
    print("已成功执行！")
    return np.array(patchList),np.array(patchNameList)

class OurDataset(D.Dataset):
    def __init__(self, patchList, patchNameList):
        '''

        :param patchList: 同一张图像的不同小块
        :param patchNameList: 不同小块左上角对应的行列号
        '''
        self.patchList = patchList
        self.patchNameList=patchNameList
        self.length = len(patchList) if self.patchList!=None else 0
        # self.as_tensor = transforms.Compose([
        #     # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
        #     transforms.ToTensor(),
        # ])

    # 获取数据操作
    def __getitem__(self, index):
        image = self.patchList[index]
        # image=self.as_tensor(image)
        return image,self.patchNameList[index] #self.patchNameList[index] if len(self.patchNameList)!=0 else self.patchNameList
    # 数据集数量
    def __len__(self):
        return self.length

    def loadData(self,patchList, patchNameList=[]):
        '''
        重新更换数据
        '''
        self.patchList = patchList
        self.patchNameList = patchNameList
        self.length = len(patchList)

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




from enum import Enum

class models(Enum):
    '''存放模型名称、路径'''
    unetPP_building= 'unetPP'
    unetPP_building_path='apps/Data/LandUseAlgorithm/fromUser/models/checkpoint_epoch10.pth'



class landUseExtract:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''初始化模型'''
        unetPP_model= NestedUNet(in_ch=3,out_ch=2).to(device=self.device)
        self.unetPP=self.loadLandUseExtractModel(model_path=models.unetPP_building_path.value, netModel=unetPP_model)
        self.modelsDict={models.unetPP_building.value:self.unetPP,
                         }
        '''初始化数据对象'''
        self.dataSet = OurDataset(patchList=np.array([]), patchNameList=[])
        '''相关参数'''
        self.args = get_args()

    def get_dataloader(self,patchList, patchNameList,batch_size=1, shuffle=False, num_workers=0):
        self.dataSet.loadData(patchList=patchList,patchNameList=patchNameList)
        dataloader = D.DataLoader(self.dataSet, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True)
        return dataloader

    def loadLandUseExtractModel(self, model_path, netModel):
        model =netModel
        # 加载训练权重
        model.load_state_dict(torch.load(model_path,map_location=self.device))
        model.eval()
        return model

    def extract_building(self, net, full_img, scale_factor=1, size_w=256, size_h=256, step=256):

        img = self.dataSet.preprocess(full_img, scale_factor, is_mask=False)
        '''裁剪'''
        patch_list, patchNameList = caijian(image=img, size_w=size_w, size_h=size_h, step=step)
        dataLoader = self.get_dataloader(patchList=patch_list, patchNameList=patchNameList,batch_size=self.args.batchSize)
        probs=np.zeros((net.out_ch,img.shape[1],img.shape[2]))#整个图像的概率图
        with torch.no_grad():
            for patches, patchesName in dataLoader:
                output = net(patches.to(device=self.device, dtype=torch.float32))
                patches_probs = F.softmax(output, dim=1)
                for patch_probs,patchName in zip(patches_probs,patchesName):
                    probs[:,patchName[0]:patchName[0]+size_h,patchName[1]:patchName[1]+size_w]=patch_probs.cpu()
            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((full_img.height, full_img.width)),
                transforms.ToTensor()
            ])  # 转为原来图像的size
            probs=torch.tensor(probs)
            probs=tf(probs)
            #得到分类类别
            mask=torch.argmax(probs,dim=0).numpy().astype(np.uint8)*255
        return mask

    def extract_water(self):
        return
    def extract_plant(self):
        return
    # 主函数 调用提取建筑物
    def extract(self, modelKey, input_path, resultPath):
        print('in===============================')
        try:
            print('*********************************')
            # logging.info(f'Loading model {self.args.model}')
            # logging.info(f'Using device {self.device }')
            print('000000000000000000000000000')
            img = Image.open(input_path)
            print('11111111111111')
            '''判断提取地物类别'''
            if modelKey==models.unetPP_building.value:
                mask = self.extract_building(net=self.modelsDict.get(modelKey),
                                             full_img=img,
                                             scale_factor=self.args.scale)

            elif modelKey==models.net_plant_path.value:
                mask=self.extract_plant()
            elif modelKey==models.net_water_path.value:
                mask=self.extract_water()

            if not self.args.no_save:
                result = Image.fromarray(mask)
                result.save(resultPath)
                logging.info(f'Mask saved to {resultPath}')
                png2png(resultPath,resultPath)
            # if self.args.viz:
            #     logging.info(f'Visualizing results for image {resultPath}, close to continue...')
            #     plot_img_and_mask(img, mask)
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False


if __name__=='__main__':
    test=landUseExtract()
    test.extract(modelKey=models.unetPP_building.value, input_path='./testData/00CANH3NII.jpg', resultPath='result3.jpg')














