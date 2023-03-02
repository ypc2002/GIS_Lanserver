import ee
import geemap
import json
import os
from ..fromUser.utils import readTif
import matplotlib as m
import matplotlib.pyplot as plt
import math
from .utils import mergeImage
import cv2

class index_differ():
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

    def mercatorTolonlat(self,mercator):
        j = 0
        for i in mercator[0]:
            # print(i)
            x = i[0]
            y = i[1]
            x = x / 20037508.34 * 180
            y = y / 20037508.34 * 180
            y = 180 / math.pi * (2 * math.atan(math.exp(y * math.pi / 180)) - math.pi / 2)
            mercator[0][j][0] = x
            mercator[0][j][1] = y
            j = j + 1
        return mercator

    def get_border_json(self, border_path):
        '''
        根据搜索条件下载遥感影像
        '''
        # open(border_path, encoding="utf-8")
        # vector_border = json.load(border_path)  # 得到矢量边界的json文件
        temp = self.getGeoJsonGeometry(border_path)
        print(('temp',temp))
        border = list(temp.values())[1][0]
        range=[border]

        return range
        # return self.mercatorTolonlat(range)

    def get_border_cordinates(self, border):
        length = len(border[0])
        # print(length)
        coord_x = sorted(border[0], key=lambda x: x[0])
        min_x = coord_x[0][0]
        max_x = coord_x[length - 1][0]
        # print(min_x,max_x)
        coord_y = sorted(border[0], key=lambda y: y[1])
        min_y = coord_y[0][1]
        max_y = coord_y[length - 1][1]
        # print(min_y,max_y)
        delt_x = max_x - min_x
        delt_y = max_y - min_y
        Outer_rectangle = [[[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]]
        return delt_x, delt_y, min_x, min_y, max_x, max_y



    def load_image(self, date_start, date_end, border_path,userId,time,landtype):
        delt = 0.12
        border = self.get_border_json(border_path)
        delt_x, delt_y, min_x, min_y, max_x, max_y = self.get_border_cordinates(border)
        temp_x = min_x
        temp_y = max_y

        if delt_x<delt and delt_y<delt:
            # 获取提取的影像
            image, color = self.get_img__collection(landtype, date_start, date_end, border_path)
            print(userId, time, landtype)
            path = os.getcwd() + '/static/tempImage/landUseImage/' + userId + '/' + time + '/' + landtype + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            rect = [
                [[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y]]]
            print(rect)
            region = ee.Geometry.Polygon(rect, None, False)
            geemap.ee_export_image(image,
                                   os.getcwd() + '/static/tempImage/landUseImage/' + userId + '/' + time + '/' + landtype + '/figue_0_0.tif', scale=10, region=region, file_per_band=False)
            langtype_image_name = '' + userId + '_' + landtype + '.tif'
            return langtype_image_name, color
        else:
            if (delt_x % delt) > 0:
                x_number = int(delt_x / delt) + 1
            if (delt_x % delt) == 0:
                x_number = int(delt_x / delt)

            if (delt_y % delt) > 0:
                y_number = int(delt_y / delt) + 1
            if (delt_y % delt) == 0:
                y_number = int(delt_y / delt)
            # 获取提取的影像
            image, color = self.get_img__collection(landtype, date_start, date_end, border_path)
            print(userId, time, landtype)
            path = os.getcwd() + '/static/tempImage/landUseImage/' + userId + '/' + time + '/' + landtype + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            for i in range(x_number):
                for j in range(y_number):
                    rect = [
                        [[min_x, max_y], [min_x + delt, max_y], [min_x + delt, max_y - delt], [min_x, max_y - delt]]]
                    print(rect)
                    region = ee.Geometry.Polygon(rect, None, False)
                    geemap.ee_export_image(image,
                                           os.getcwd() + '/static/tempImage/landUseImage/' + userId + '/' + time + '/' + landtype + '/figue_{}_{}.tif'.format(
                                               i, j), scale=10, region=region, file_per_band=False)
                    max_y -= delt
                max_y = temp_y
                min_x += delt
            langtype_image_name = '' + userId + '_' + landtype + '.tif'
            return langtype_image_name, color


    def merge_image(self, landtype,userId, name, time,color):
        tifDir = './static/tempImage/landUseImage/' + userId + '/' + time + '/'+landtype
        name = name[:-4] + '.png'
        a=mergeImage(tifDir, name, 4)
        # 将灰度图转换成np，然后显示颜色
        outtif_LUCC_path = './static/tempImage/landUseImage/' + userId + '/' + time + '/'+landtype+'/' + name + ''
        tifname = 'result_index.png'
        outtif_LUCC = './static/tempImage/landUseImage/' + userId + '/' + time + '/'+landtype+'/' + tifname + ''
        cmap = m.colors.ListedColormap(color)


        if landtype=="plant":
            plt.imsave(arr=a, fname=outtif_LUCC, vmin=0.15,vmax=1,cmap=cmap)
        if landtype=="water":
            plt.imsave(arr=a, fname=outtif_LUCC, vmin=-0.5,vmax=0.38, cmap=cmap)
        if landtype=="build":
            plt.imsave(arr=a, fname=outtif_LUCC, vmin=-0.56,vmax=0.5, cmap=cmap)
        url = './static/tempImage/landUseImage/' + userId + '/' + time + '/'+landtype+'/' + tifname + ''
        return url

    def get_img__collection(self,landtype,date_start, date_end, border_path):
        border = self.get_border_json(border_path)
        region = ee.Geometry.Polygon(border, None, False)
        image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterDate(date_start, date_end)
                 .filterBounds(region)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
                 .median()
                 ).select(['B4', 'B8', 'B3','B11'])

        if landtype=='water':
            ndwi=self.NDWI(image)
            color=['white','blue']
            return ndwi,color
        if landtype=='plant':
            ndvi=self.NDVI(image)
            color=['white','green']
            return ndvi,color
        if landtype=='build':
            ndbi=self.NDBI(image)
            color=['white','gray']
            return ndbi,color

    #归一化水体指数：NDMI的改进
    def NDWI(self,image):
        img = image.normalizedDifference(['B3', 'B8'])
        # ndwi = img.updateMask(img.gte(0.05))
        return img

    #归一化植被指数
    def NDVI(self,image):
        img = image.normalizedDifference(['B8', 'B4'])
        # ndvi = img.gte(0.2)
        return img

    #建筑指数（不透水面）
    def NDBI(self,image):
        img = image.normalizedDifference(['B8', 'B11'])
        # ndbi= img.updateMask(img.gte(0.5))
        return img

    # def NDISI(self,image,date_start, date_end, border_path ,userId,time):
    #     '''徐涵秋提出的NDISI来提取不透水面信息。该指数采用复合波段的方法构成，能有效区别不透水面和土壤，并且可以抑制沙土和水体信息的影响，因此不需要预先进行
    #   剔除，较好地提高了信息的纯度'''
    #
    #     MNDWI=self.MNDWI()
    #     return (self.TIR(MNDWI+self.NIR+self.MIR)/3)/(self.TIR+(MNDWI+self.NIR+self.MIR))

def figure_extract(date_start, date_end, border_path, userId, time,landtype):
    a =index_differ()
    tifName,color = a.load_image(date_start, date_end, border_path, userId, time,landtype)
    #tifName='source.tif'
    outpath = a.merge_image(landtype,userId, tifName, time, color)
    return outpath

