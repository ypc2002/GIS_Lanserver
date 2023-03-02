#DL变化检测
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms as T
import torch.utils.data as D
import cv2
import traceback
from apps.Data.extractBoundary.main_regularization import png2png,toSVG
# 根据任务要求，定义一个caijian函数
def caijian(image1:np.array,image2:np.array, size_w=256, size_h=256, step=256):  # step为步长，设置为256即相邻图片重叠50%
    '''
    :param image1Path: 裁剪图像的路径
    :param size_w: 裁剪成小块的宽
    :param size_h: 裁剪成小块的高
    :param step: 各个小块之间的步长
    :return: 小块组成的数组，每个小块左上角对应的行列元组数组
    '''
    patchList=[]
    patchNameList=[]
    size = image1.shape
    i = 0
    for h in range(0, size[0], step):
        star_h = h  # star_h表示起始高度，从0以步长step开始循环
        for w in range(0, size[1], step):
            star_w = w  # star_w表示起始宽度，从0以步长step开始循环
            end_h = star_h + size_h  # end_h是终止高度

            if end_h > size[0]:  # 如果边缘位置不够size_h的列
                # 以倒数size_h形成裁剪区域
                star_h = size[0] - size_h
                end_h = star_h + size_h
                i = i - 1
            end_w = star_w + size_w  # end_w是中止宽度
            if end_w > size[1]:  # 如果边缘位置不够size_w的行
                # 以倒数size_w形成裁剪区域
                star_w = size[1] - size_w
                end_w = star_w + size_w
                i = i - 1
            image1Cropped = image1[star_h:end_h, star_w:end_w]  # 执行裁剪操作
            image2Cropped=image2[star_h:end_h, star_w:end_w]
            #通道合并
            sampleCroped=np.concatenate((image1Cropped,image2Cropped),axis=2)
            i = i + 1
            patch_img = (star_h,star_w) # 用起始坐标来命名切割得到的图像，为的是方便后续标签数据抓取
            patchList.append(sampleCroped)
            patchNameList.append(patch_img)
        print(h)
    print("已成功执行！")
    return np.array(patchList),np.array(patchNameList)


'''图像预处理'''
def style_transfer(source_image, target_image):
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:, :, i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:, :, i])
        target_image_fshift = np.fft.fftshift(target_image_f)

        change_length = 1
        source_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
        int(h / 2) - change_length:int(h / 2) + change_length] = \
            target_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
            int(h / 2) - change_length:int(h / 2) + change_length]

        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)
        source_image_if[source_image_if > 255] = np.max(source_image[:, :, i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1, 0).swapaxes(1, 2)

    out = out.astype(np.uint8)
    return out


class OurDataset(D.Dataset):
    def __init__(self, patchList, patchNameList):
        '''

        :param patchList: 同一张图像的不同小块
        :param patchNameList: 不同小块左上角对应的行列号
        '''
        self.patchList = patchList
        self.patchNameList=patchNameList
        self.length = len(patchList) if self.patchList!=None else 0
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])

    # 获取数据操作
    def __getitem__(self, index):
        image = self.patchList[index]
        image_A = image[:, :, 0:3]
        image_B = image[:, :, 3:6]
        image_B_fromA = style_transfer(image_B, image_A)
        image = np.concatenate((image_A, image_B_fromA), axis=2)
        image=self.as_tensor(image)
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


'''模型列表'''
from enum import Enum
class models(Enum):
    unetPP='unetPP'
    unetPPPath='apps/Data/changeDectionAlgorithm/fromUser/models/UnetPlusPlus_changeDection.pth'
    # unetPPPath = 'models/UnetPlusPlus_changeDection.pth'

modelsPath={
    'unetPPPath':'apps/Data/changeDectionAlgorithm/fromUser/models/UnetPlusPlus_changeDection.pth'
}
class changeDection:
    def __init__(self):
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        '''初始化模型'''
        unetPP_model= smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=6,
            decoder_attention_type="scse",
            classes=1,
            activation="sigmoid",
        )
        self.unetPP=self.loadChangeDetectModel(model_path=models.unetPPPath.value,netModel=unetPP_model)
        self.modelsDict={models.unetPP.value:self.unetPP}
        '''初始化数据对象'''
        self.dataSet = OurDataset(patchList=np.array([]), patchNameList=[])
    def addModel(self,modelPath:str,netModel):
        '''添加模型'''
        return

    def get_dataloader(self,patchList, patchNameList,batch_size, shuffle=False, num_workers=0):
        self.dataSet.loadData(patchList=patchList,patchNameList=patchNameList)
        dataloader = D.DataLoader(self.dataSet, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=True)
        return dataloader

    def loadChangeDetectModel(self,model_path, netModel=smp.UnetPlusPlus):
        model =netModel
        # 将模型加载到指定设备DEVICE上

        model.to(self.DEVICE)
        # 加载训练权重
        model.load_state_dict(torch.load(model_path,map_location=self.DEVICE))
        model.eval()
        return model

    def predect(self,modelKey:str,image1Path:str,image2Path:str,resultPath:str):
        '''
        :param modelKey: 选择的模型的key
        :param image1Path: 图片1路径
        :param image2Path: 图片2路径
        :param resultPath: 结果保存路径
        '''
        '''相关参数'''
        size_w=256 #32的倍数
        size_h=256
        step=256
        batch_size=2
        threshold=0.25
        try:
            '''读取图片'''
            img1 = cv2.imread(image1Path, cv2.IMREAD_UNCHANGED)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(image2Path, cv2.IMREAD_UNCHANGED)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            '''判断shape,大于256*256，裁剪为'''
            if img1.shape[0]==img2.shape[0] and img1.shape[1]==img2.shape[1]:
                patchList, patchNameList=[],[]
                if img1.shape[0]>size_h or img1.shape[1]>size_w:
                    '''裁剪'''
                    patchList,patchNameList=caijian(image1=img1,image2=img2,size_w=size_w,size_h=size_h,step=step)
                else:
                    '''不需要裁剪'''
                    patchList=np.array([np.concatenate((img1,img2),axis=2)])
                    patchNameList=np.array([(0,0)])
                '''加载到dataloader对象'''
                dataloader=self.get_dataloader(patchList=patchList,patchNameList=patchNameList,batch_size=batch_size)
                predictChangeImgList=[]#存储预测结果
                predictChangeImgNameList=[]#存储预测结果在原始影响中对应的位置
                '''变化检测'''
                for sample,indexName in dataloader:
                    with torch.no_grad():
                        sample = sample.to(self.DEVICE)
                        output = self.modelsDict.get(modelKey)(sample).cpu().data.numpy()
                    '''将每个batch的所有结果存到predictChangeImgList中'''
                    for i in range(output.shape[0]):
                        pred = output[i]
                        threshold = threshold
                        pred[pred >= threshold] = 255
                        pred[pred < threshold] = 0
                        pred = np.uint8(pred)
                        pred = pred.reshape((size_w, size_w))
                        predictChangeImgList.append((pred))
                        predictChangeImgNameList.append(indexName[i])
                '''保存结果'''
                result = np.zeros(img1.shape[0:-1])  # 存储结果的矩阵
                for predictChangeImg, predictChangeImgName in zip(predictChangeImgList, predictChangeImgNameList):
                    startH = predictChangeImgName[0]
                    startW = predictChangeImgName[1]
                    result[startH:startH + size_h, startW:startW + size_w] = predictChangeImg
                cv2.imwrite(resultPath, result)
                png2png(resultPath,resultPath)
                # toSVG(resultPath,resultPath)
            else:
                '''错误处理'''
                print('图片大小不一致，无法正确匹配检测！')
                return False
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False
        return True





if __name__=='__main__':
    CD=changeDection()
    # CD.predect(models.unetPP.value,'./testData/13.tif','./testData/13(2).tif','./testData/result1.png')
    # CD.predect(models.unetPP.value, './testData/CD1.png', './testData/CD2.png', './testData/result2.png')
    CD.predect(models.unetPP.value, './testData/train_1.png', './testData/train_1(1).png', './testData/result2.png')

    # A=2
    # print(A[1] if A==1 else None)