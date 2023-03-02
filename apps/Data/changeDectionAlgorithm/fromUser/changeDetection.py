import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import SpectralClustering,AgglomerativeClustering,Birch,MiniBatchKMeans
from utils import readTif, readPng_Jpg,TwoDtoOneD,OneDtoTwoD
from remoteIndex import RSIndex

class PCADiffer(object):
    # def TwoDtoOneD(self,Image: np.array, imgType:str):
    #     ImageShape = Image.shape
    #     result = []
    #     if imgType == 'tif':
    #         for x in range(ImageShape[1]):
    #             for y in range(ImageShape[2]):
    #                 result.append(Image[:, x, y])
    #     elif imgType in ['jpg', 'png']:
    #         for x in range(ImageShape[0]):
    #             for y in range(ImageShape[1]):
    #                 result.append(Image[x, y])
    #     return np.array(result)
    #
    # '''将图像一维转二维'''
    #
    # def OneDtoTwoD(self,ImageDataList: np.array, aimShape):
    #     '''
    #     :param ImageDataList:
    #     :param aimShape: [width,height]
    #     :return:
    #     '''
    #     result = []
    #     for y in range(aimShape[1]):
    #         row = []
    #         for x in range(aimShape[0]):
    #             row.append(ImageDataList[y * aimShape[0] + x])
    #         result.append(row)
    #     return np.array(result)
    def readImage(self,filePath1,filePath2):
        if 'tif' in filePath1:
            Image1=readTif(filePath1)
            Image2=readTif(filePath2)
            return Image1,Image2
        elif 'png' in filePath1:
            Image1 =readPng_Jpg(filePath1)
            Image2=readPng_Jpg(filePath2)
            return Image1,Image2


    def calcu(self, Image1:np.array, Image2:np.array, n_components:int=1, imgType='png'):
        ImageShape=Image1.shape
        Image_1=TwoDtoOneD(Image=Image1, imgType=imgType)
        Image_2=TwoDtoOneD(Image=Image2, imgType=imgType)
        ipca=IncrementalPCA(n_components=n_components, batch_size=10)
        Image1_X=ipca.fit_transform(Image_1)
        Image2_X = ipca.fit_transform(Image_2)
        ImageDiff=Image2_X-Image1_X
        # clustering = MiniBatchKMeans(n_clusters=2).fit(ImageDiff)  快一点
        clustering=Birch(n_clusters=2).fit(ImageDiff) #稳一点
        # 固定类别颜色（不变黑、增多白）
        # cluster_center = clustering.cluster_centers_.reshape(clustering.cluster_centers_.shape[0])
        # cluster_labels = clustering.labels_
        # min2max = [-1, -1]
        # if cluster_center[0]<cluster_center[1]:
        #     min2max[0],min2max[1]=0,1
        # else:
        #     min2max[0],min2max[1]=1,0
        # for i in range(len(min2max)):
        #     cluster_labels[i] = min2max[i]

        if imgType== 'tif':
            ImageDiff=OneDtoTwoD(clustering.labels_,ImageShape[1:])
            return ImageDiff
        elif imgType in ['jpg', 'png']:
            ImageDiff = OneDtoTwoD(clustering.labels_, ImageShape[:2])
            return ImageDiff
        return None

    def run(self, filePath1, filePath2,savePath,n_components:int=2, imgType='png'):
        Image1,Image2=self.readImage(filePath1=filePath1,filePath2=filePath2)
        ImageDiff=self.calcu(Image1=Image1, Image2=Image2, n_components=n_components, imgType=imgType)
        print(ImageDiff)
        print(np.unique(ImageDiff))
        print(ImageDiff.shape)
        plt.imshow(ImageDiff,cmap='gray')
        plt.show()
        plt.imsave(savePath,ImageDiff,cmap='gray')


def MultiTemporalBandMix(image1,image2):
    image1[0]=image2[0]
    return image1


class indexChangeDetection:
    def __init__(self,image1Index:RSIndex,image2Index:RSIndex):
        self.image1Index=image1Index
        self.image2Index=image2Index
    def getDifferMask(self, Image1: np.array, Image2: np.array):
        differ = Image2 - Image1
        if len(differ.shape)==2:
            MIN=np.min(differ)
            MAX=np.max(differ)
            differ=(differ-MIN)/(MAX-MIN)
        elif len(differ.shape)==2:
            for i in range(differ.shape[0]):
                MIN = np.min(differ[i])
                MAX = np.max(differ[i])
                differ[i] = (differ[i] - MIN) / (MAX - MIN)
        return differ
    def cluster(self,image:np.array):
        pixList=TwoDtoOneD(image,imgType='oneBand')
        pixListShape=pixList.shape
        clustering =MiniBatchKMeans(n_clusters=3).fit(np.reshape(pixList,(pixListShape[0],1)))
        #固定类别颜色（减少黑、不变灰、增多白）
        cluster_center=clustering.cluster_centers_.reshape(clustering.cluster_centers_.shape[0])
        cluster_labels=clustering.labels_
        min2max=[-1,-1,-1]
        for i in range(len(cluster_center)):
            order = 0
            for j in range(len(cluster_center)):
                if cluster_center[i]<cluster_center[j]:
                    order+=1
            min2max[i]=order
        for i in range(len(min2max)):
            cluster_labels[i]=min2max[i]
        class_image = OneDtoTwoD(cluster_labels, image.shape)
        return class_image
    def NDMIchange(self):
        DifferMask = self.getDifferMask(self.image1Index.NDMI(),self.image2Index.NDMI())
        return self.cluster(DifferMask)
    def NDWIchange(self):
        DifferMask= self.getDifferMask(self.image1Index.NDWI(), self.image2Index.NDWI())
        return self.cluster(DifferMask)

    def MNDWIchange(self):
        DifferMask = self.getDifferMask(self.image1Index.MNDWI(), self.image2Index.MNDWI())
        return self.cluster(DifferMask)
    def RVIchange(self):
        DifferMask =self.getDifferMask(self.image1Index.RVI(), self.image2Index.RVI())
        return self.cluster(DifferMask)
    def NDVIchange(self):
        DifferMask= self.getDifferMask(self.image1Index.NDVI(), self.image2Index.NDVI())
        return self.cluster(DifferMask)
    def SAVIchange(self):
        DifferMask=self.getDifferMask(self.image1Index.SAVI(), self.image2Index.SAVI())
        return self.cluster(DifferMask)
    def NDBIchange(self):
        DifferMask= self.getDifferMask(self.image1Index.NDBI(), self.image2Index.NDBI())
        return self.cluster(DifferMask)
    def NDISIchange(self):
        DifferMask= self.getDifferMask(self.image1Index.NDISI(), self.image2Index.NDISI())
        return self.cluster(DifferMask)



if __name__=='__main__':
    from utils import readImage
    import matplotlib.pyplot as plt


    '''全部地物变化检测'''
    # filePath1 = './testData/train_1.png'
    # filePath2 = './testData/train_1(1).png'

    filePath1 = './testData/image0_0.tif'
    filePath2 = './testData/image_10_0.tif'
    # pcaDif=PCADiffer()
    # pcaDif.run(filePath1=filePath1, filePath2=filePath2,savePath='./testData/test1PCAbrich_2c.jpg', n_components=1, imgType='tif')



    '''单类地物变化检测 只针对tif影像'''

    image1 = readImage(filePath1)
    image2 = readImage(filePath2)
    # image=MultiTemporalBandMix(image1=image1,image2=image2)
    # plt.imshow(image1)
    # plt.show()
    # plt.imshow(image2)
    # plt.show()
    # plt.imshow(image)
    # plt.show()
    # plt.imsave('./testData/mix.png',image)


    image1_index=RSIndex(image=image1,R=4,G=3,B=2,NIR=8)
    image2_index=RSIndex(image=image2,R=4,G=3,B=2,NIR=8)
    changeImage=indexChangeDetection(image1Index=image1_index,image2Index=image2_index)
    NDMIchangeImage=changeImage.NDMIchange()
    plt.imshow(NDMIchangeImage,cmap='gray')
    plt.show()
    plt.imsave('./testData/NDMIchange.jpg',NDMIchangeImage,cmap='gray')

    NDVIchangeImage=changeImage.NDVIchange()
    plt.imshow(NDVIchangeImage,cmap='gray')
    plt.show()
    plt.imsave('./testData/NDVIchange.jpg',NDVIchangeImage,cmap='gray')