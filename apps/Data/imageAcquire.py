
import ee
import geemap
import json
import os
import  matplotlib as m
import matplotlib.pyplot as plt
from .utils import mercatorTolonlat,mergeImage
import cv2
import traceback
import numpy as np

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:12307'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:12307'

# ee.Authenticate()
ee.Initialize()
'''
数据获取：
1，原始影像获取；
2，土地利用分类图获取。
'''

class DataAcquire():
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
        根据搜索条件下载遥感影像边界
        '''
        # open(border_path, encoding="utf-8")
        # vector_border = json.load(border_path)  # 得到矢量边界的json文件
        temp = self.getGeoJsonGeometry(border_path)
        border = list(temp.values())[1][0]
        range = [border]

        return range


    def get_border_cordinates(self, border):
        '''
        获得边界的最小外包矩形
        '''
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

    def load_image(self, date_start, date_end, border_path, select_image, userId,time,landtype):
        '''
        根据搜索条件下载遥感影像
        '''
        if select_image==0:
            delt = 0.12
        else:
            delt=0.05
        border = self.get_border_json(border_path)

        delt_x, delt_y, min_x, min_y, max_x, max_y = self.get_border_cordinates(border)

        temp_x = min_x
        temp_y = max_y
        #对影像进行切片
        if (delt_x % delt)>0:
            x_number = int(delt_x / delt) + 1
        if (delt_x % delt)==0:
            x_number =int(delt_x / delt)

        if (delt_y % delt )> 0:
            y_number = int(delt_y / delt) + 1
        if (delt_y % delt)==0 :
            y_number =int(delt_y / delt)
        print(y_number, x_number)

        if select_image == 0:
            source_path=os.getcwd()+'/static/tempImage/sourceImage/'+userId+'/'+time+'/'
            if not os.path.exists(source_path):
                os.makedirs(source_path)
            if delt_x < delt and delt_y < delt:
                rect = [
                    [[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y]]]
                print(rect)
                region = ee.Geometry.Polygon(rect, None, False)
                image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                         .filterDate(date_start, date_end)
                         .filterBounds(region)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
                         .median()
                         ).select(['B4', 'B3', 'B2'])

                geemap.ee_export_image(image,
                                       './static/tempImage/sourceImage/' + userId + '/' + time + '/remote_image_0_0.tif'
                                           , scale=10, region=region, file_per_band=False)
            else:
                for i in range(x_number):
                    for j in range(y_number):
                        rect = [
                            [[min_x, max_y], [min_x + delt, max_y], [min_x + delt, max_y - delt], [min_x, max_y - delt]]]
                        print(rect)
                        region = ee.Geometry.Polygon(rect, None, False)
                        image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                                 .filterDate(date_start, date_end)
                                 .filterBounds(region)
                                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
                                 .median()
                                 ).select(['B4', 'B3', 'B2'])

                        geemap.ee_export_image(image,'./static/tempImage/sourceImage/' + userId + '/'+time+'/remote_image_{}_{}.tif'.format(i, j), scale=10, region=region, file_per_band=False)
                        max_y -= delt
                    max_y = temp_y
                    min_x += delt
            source_image_name='{}_sourceImage.tif'.format(userId)
            return source_image_name
        else:
            landuse_path=os.getcwd()+'/static/tempImage/landUseImage/'+userId+'/'+time+'/'
            if not os.path.exists(landuse_path):
                os.makedirs(landuse_path)
            if delt_x < delt and delt_y < delt:
                rect = [
                    [[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y]]]
                region = ee.Geometry.Polygon(rect, None, False)
                image = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                         .filterDate(date_start, date_end)
                         .filterBounds(region)
                         .median()
                         ).select(['label'])
                image1 = image.toUint8()
                geemap.ee_export_image(image1,
                                       os.getcwd() + '/static/tempImage/landUseImage/'
                                                     '' + userId + '/' + time + '/LUCC_image_0_0.tif', scale=10, region=region, file_per_band=False)
            else:
                for i in range(x_number):
                    for j in range(y_number):
                        rect = [
                            [[min_x, max_y], [min_x + delt, max_y], [min_x + delt, max_y - delt], [min_x, max_y - delt]]]
                        region = ee.Geometry.Polygon(rect, None, False)
                        image = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                                 .filterDate(date_start, date_end)
                                 .filterBounds(region)
                                 .median()
                                 ).select(['label'])
                        image1=image.toUint8()
                        geemap.ee_export_image(image1,
                                               os.getcwd()+'/static/tempImage/landUseImage/'
                                               '' + userId + '/'+time+'/LUCC_image_{}_{}.tif'.format(
                                                   i, j), scale=10, region=region, file_per_band=False)
                        max_y -= delt
                    max_y = temp_y
                    min_x += delt
            landuse_image_name = '{}_landuseImage.tif'.format(userId)
            return landuse_image_name

    def merge_image(self,select_img,userId,name,time,landtype):
        '''
        合并遥感影像
        '''
        if select_img==0:
            tifDir = './static/tempImage/sourceImage/'+userId+'/'+time+'/'
        else:
            tifDir = './static/tempImage/landUseImage/'+userId+'/'+time+'/'
        if select_img==0:
            try:
                name=name[:-4] + '.png'
                mergeImage(tifDir,name,3)
                a=url = './static/tempImage/sourceImage/'+userId+'/'+time+'/' + name + ''
                if a:
                    return url
                else:
                    return False
            except Exception as e:
                print(e.args)
                print(traceback.print_exc())
                return False
        else:
            name = name[:-4] + '.png'
            mergeImage(tifDir, name, 1)

            outtif_LUCC_path=tifDir+name
            a = cv2.imread(outtif_LUCC_path,0)
            print(a)
            temp=np.zeros(a.shape)
            print(temp)

            # cv2.imwrite('2.png', water)
            tifname='result.png'

            outtif_LUCC = './static/tempImage/landUseImage/' + userId + '/' + time + '/' + tifname + ''
            colors = [(191, 101, 33), (199, 131, 183), (120, 80, 173), (134, 121, 58), (28, 106, 203), (33, 61, 166),
                      (60, 216, 229), (91, 101, 113),
                      (77, 97, 31)]
            if landtype=='label':
                color=colors
                cmap = m.colors.ListedColormap(color)
                plt.imsave(arr=a, fname=outtif_LUCC, cmap=cmap, vmin=-0.5, vmax=8.5)
                url='./static/tempImage/landUseImage/'+userId+'/'+time+'/'+tifname+''
                return url
            elif landtype=='water':
                temp[a == 0] =1
                color=[(255,255,255),colors[0]]
            elif landtype=='trees':
                temp[a == 1] =1
                print("wa",temp)
                color=[(255,255,255),colors[1]]
            elif landtype=='grass':
                temp[a == 2] =1
                color=[(255,255,255),colors[2]]
            elif landtype=='flooded_vegetation':
                temp[a == 3] =1
                color=[(255,255,255),colors[3]]
            elif landtype=='crops':
                temp[a == 4] =1
                color=[(255,255,255),colors[4]]
            elif landtype=='shrub_and_scrub':
                temp[a == 5] =1
                color=[(255,255,255),colors[5]]
            elif landtype=='built':
                temp[a == 6] =1
                color=[(255,255,255),colors[6]]
            elif landtype=='bare':
                temp[a == 7] =1
                color=[(255,255,255),colors[7]]
            elif landtype=='snow_and_ice':
                temp[a == 8] =1
                color=[(255,255,255),colors[8]]
            cmap = m.colors.ListedColormap(color)
            plt.imsave(arr=temp, fname=outtif_LUCC, cmap=cmap,vmin=-0.5,vmax=1.5)
            url = './static/tempImage/landUseImage/' + userId + '/' + time + '/' + tifname + ''
            return url



def get_sourceimage(date_start, date_end, border_path, select_image, userId,time,landtype):
            a = DataAcquire()
            tifName=a.load_image(date_start, date_end, border_path, select_image, userId,time,landtype)
            # tifName='source.tif'
            outpath=a.merge_image(select_image,userId,tifName,time,landtype)
            return outpath
