import ee
import geemap
import json
import numpy as np
from tkinter import EXCEPTION
from utils import *

class index_differ:
    # def __init__(self,date_start_befor,date_end_befor,date_start_now,date_end_now,border_path):
    #     self.date_start_befor=date_start_befor
    #     self.date_end_befor=date_end_befor
    #     self.date_start_now=date_start_now
    #     self.date_end_now=date_end_now
    #     self.border_path=border_path

    def getGeoJsonGeometry(self, GeoJson: json):
        '''
        :param GeoJson:
        :return: geometry or geometries
        '''
        if GeoJson['type'] == 'FeatureCollection':
            return [feature['geometry'] for feature in GeoJson['features']]
        elif GeoJson['type'] == 'Feature':
            return GeoJson['geometry']
        elif GeoJson['type'] in ["Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"]:
            return GeoJson
        elif GeoJson['type'] == "GeometryCollection":
            return GeoJson['geometries']
        else:
            return False

    def get_border_json(self, border_path):
        '''
        根据搜索条件下载遥感影像
        '''
        # open(border_path, encoding="utf-8")
        # vector_border = json.load(border_path)  # 得到矢量边界的json文件
        temp = self.getGeoJsonGeometry(border_path)
        border = list(temp[0].values())[1][0]
        return border

    def otsu(self,histogram):
        '''
            计算阈值
        '''
        counts = ee.Array(histogram.getInfo().get('pc1').get('histogram'))
        means = ee.Array(histogram.getInfo().get('pc1').get('bucketMeans'))
        # print('means',means)
        size = means.length().get([0])
        # print('size',size)
        total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        mean = sum.divide(total)
        indices = ee.List.sequence(1, size)
        def fun(i):
            aCounts = counts.slice(0, 0, i)
            aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
            aMeans = means.slice(0, 0, i)
            aMean = aMeans.multiply(aCounts)\
                            .reduce(ee.Reducer.sum(), [0]).get([0])\
                            .divide(aCount)
            bCount = total.subtract(aCount)
            bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
            return aCount.multiply(aMean.subtract(mean).pow(2)).add(bCount.multiply(bMean.subtract(mean).pow(2)))
        bss = indices.map(lambda i:fun(i))
        return means.sort(bss).get([-1])

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

