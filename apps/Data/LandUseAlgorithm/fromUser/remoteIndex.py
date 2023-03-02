import matplotlib.pyplot as plt
import numpy as np

'''指数法'''
class RSIndex:
    def __init__(self,image:np.array,R:int,G:int,B:int,NIR:int,*,MIR:int=None,TIR:int=None):
        self.R=image[R]
        self.G=image[G]
        self.B=image[B]
        self.NIR=image[NIR]
        if MIR!=None:
            self.MIR=image[MIR]
        if TIR!=None:
            self.TIR=image[TIR]
    #归一化差分水体指数
    def NDMI(self):
        return (self.G-self.NIR)/(self.G+self.NIR)
    #植被水分指数：用于研究植被含水量，旱情监测
    def NDWI(self):
        return (self.NIR-self.MIR)/(self.NIR+self.IR)
    #归一化差异水体指数：NDMI的改进
    def MNDWI(self):
        return (self.G-self.MIR)/(self.G+self.MIR)
    #比值植被指数
    def RVI(self):
        return self.NIR/self.R
    #归一化植被指数
    def NDVI(self):
        return (self.NIR-self.R)/(self.NIR+self.R)
    #土壤调节植被指数
    def SAVI(self,L=0.5):
        return (self.NIR-self.R)*(1+L)/(self.NIR+self.R+L)
    #建筑指数（不透水面）
    def NDBI(self):
        return (self.MIR-self.NIR)/(self.MIR+self.NIR)
    def NDISI(self):
        '''徐涵秋提出的NDISI来提取不透水面信息。该指数采用复合波段的方法构成，能有效区别不透水面和土壤，并且可以抑制沙土和水体信息的影响，因此不需要预先进行
      剔除，较好地提高了信息的纯度'''
        MNDWI=self.MNDWI()
        return (self.TIR(MNDWI+self.NIR+self.MIR)/3)/(self.TIR+(MNDWI+self.NIR+self.MIR))

#差值

# from utils import readTif,readPng_Jpg,readImage
# def readImage(filePath1,filePath2):
#     if 'tif' in filePath1:
#         Image1=readTif(filePath1)
#         Image2=readTif(filePath2)
#         return Image1,Image2
#     elif 'png' in filePath1:
#         Image1 =readPng_Jpg(filePath1)
#         Image2=readPng_Jpg(filePath2)
#         return Image1,Image2

if __name__=='__main__':
    filePath1='./testData/image0_0.tif'
    filePath2 = './testData/image_10_0.tif'
    Image1=readImage(filePath=filePath1)
    R_index1=RSIndex(image=Image1,R=4,G=3,B=3,NIR=8)
    NDVI=R_index1.NDVI()
    plt.imshow(NDVI)
    plt.show()
