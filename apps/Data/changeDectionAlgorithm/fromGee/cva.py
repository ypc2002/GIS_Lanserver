import ee
import geemap
import os
import json
from .utils import mercatorTolonlat
from .utils import mercatorTolonlat,mergeImage
import cv2
import matplotlib as m
import matplotlib.pyplot as plt

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:12307'
os.environ['HTTPS_PROXY'] ='http://127.0.0.1:12307'

# ee.Authenticate()
ee.Initialize()

def get_border_cordinates(border):
    length=len(border[0])
    # print(length)
    coord_x=sorted(border[0],key=lambda x:x[0])
    min_x=coord_x[0][0]
    max_x=coord_x[length-1][0]
    # print(min_x,max_x)
    coord_y=sorted(border[0],key=lambda y:y[1])
    min_y=coord_y[0][1]
    max_y=coord_y[length-1][1]
    # print(min_y,max_y)
    delt_x=max_x-min_x
    delt_y=max_y-min_y
    Outer_rectangle=[[[min_x,min_y],[min_x,max_y],[max_x,max_y],[max_x,min_y]]]
    return delt_x,delt_y,min_x,min_y,max_x,max_y

def getGeoJsonGeometry(GeoJson:json):
    '''
    :param GeoJson:
    :return: geometry or geometries
    '''
    if GeoJson['type']=='FeatureCollection':
        return [feature['geometry'] for feature in GeoJson['features']]
    elif GeoJson['type']=='Feature':
        return GeoJson['geometry']
    elif  GeoJson['type'] in ["Point","MultiPoint","LineString","MultiLineString","Polygon","MultiPolygon"]:
        return GeoJson
    elif GeoJson['type']=="GeometryCollection":
        return GeoJson['geometries']
    else:
        return False

class cva:
    # def __init__(self,date_start_befor,date_end_befor,date_start_now,date_end_now,border_path):
    #     self.date_start_befor=date_start_befor
    #     self.date_end_befor=date_end_befor
    #     self.date_start_now=date_start_now
    #     self.date_end_now=date_end_now
    #     self.border_path=border_path

    def get_mutibands_img(self,band_1, band_2, band_3,band_name1,band_name2,band_name3):
        img1 = ee.Image.constant(ee.Number(band_1)).rename(band_name1)
        img2 = ee.Image.constant(ee.Number(band_2)).rename(band_name2)
        img3 = ee.Image.constant(ee.Number(band_3)).rename(band_name3)
        return img1.addBands(img2).addBands(img3)

    def cva1(self,norm_A, norm_B,band_name1,band_name2,band_name3):
        img_diff = norm_A.subtract(norm_B)
        square_img_diff = img_diff.pow(2)
        sum_image = square_img_diff.expression("B1+B2+B3",
            {
                'B1':square_img_diff.select(band_name1),
                'B2':square_img_diff.select(band_name2),
                'B3':square_img_diff.select(band_name3)
            }
        )
        return sum_image.rename("sum_image").sqrt()

    def otsu(self,histogram):
        counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
        means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
        size = means.length().get([0])
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


    def cva_dectation(self,date_start_befor,date_end_befor,date_start_now,date_end_now,border_path):
        # vector_border=json.load(open(border_path,encoding="utf-8"))
        temp=getGeoJsonGeometry(border_path)
        border=list(temp.values())[1][0]
        range = [border]
        roi =ee.Geometry.Polygon(range,None, False)
        # roi = ee.Geometry.Polygon([[[113.316679, 31.059496], [113.316679, 31.099496], [113.356679, 31.099496], [113.356679, 31.059496]]],None,False)

        imageCollection = ee.ImageCollection("COPERNICUS/S2")

        # 设定波段组合
        band_list = ee.List(['B4', 'B3', 'B2']) # 432真彩色, 832假彩色
        band_name1 = ee.String(band_list.getInfo()[0])
        band_name2 = ee.String(band_list.getInfo()[1])
        band_name3 = ee.String(band_list.getInfo()[2])

        imgA = (imageCollection.filterDate(date_start_befor,date_end_befor)
                            .filterBounds(roi)
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                            .select(band_list)
                            .median()
                            .clip(roi)
        )

        imgB = (imageCollection.filterDate(date_start_now,date_end_now)
                            .filterBounds(roi)
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                            .select(band_list)
                            .median()
                            .clip(roi)
        )

        meanA = imgA.reduceRegion(ee.Reducer.mean(), roi, 10,maxPixels=1e9)
        meanB = imgB.reduceRegion(ee.Reducer.mean(), roi, 10,maxPixels=1e9)
        stdA = imgA.reduceRegion(ee.Reducer.stdDev(),roi ,10,maxPixels=1e9)
        stdB = imgB.reduceRegion(ee.Reducer.stdDev(),roi ,10,maxPixels=1e9)

        imgA_mean = self.get_mutibands_img(meanA.get(band_name1), meanA.get(band_name2), meanA.get(band_name3),band_name1,band_name2,band_name3)
        imgB_mean = self.get_mutibands_img(meanB.get(band_name1), meanB.get(band_name2), meanB.get(band_name3),band_name1,band_name2,band_name3)
        imgA_std = self.get_mutibands_img(stdA.get(band_name1), stdA.get(band_name2), stdA.get(band_name3),band_name1,band_name2,band_name3)
        imgB_std = self.get_mutibands_img(stdB.get(band_name1), stdB.get(band_name2), stdB.get(band_name3),band_name1,band_name2,band_name3)

        normA = imgA.subtract(imgA_mean).divide(imgA_std)
        normB = imgB.subtract(imgB_mean).divide(imgB_std)

        l2_norm = self.cva1(normA, normB,band_name1,band_name2,band_name3)
        histogram = l2_norm.reduceRegion(ee.Reducer.histogram(), roi, 10,maxPixels=1e9)
        threshold = self.otsu(histogram.get("sum_image"))

        mask = l2_norm.gte(threshold)
        return mask,range

    def load_image(self,img,border,userId,time):
        delt=0.1
        delt_x,delt_y,min_x,min_y,max_x,max_y=get_border_cordinates(border)
        temp_x=min_x
        temp_y=max_y
        if (delt_x % delt) > 0:
            x_number = int(delt_x / delt) + 1
        if (delt_x % delt) == 0:
            x_number = int(delt_x / delt)

        if (delt_y % delt) > 0:
            y_number = int(delt_y / delt) + 1
        if (delt_y % delt) == 0:
            y_number = int(delt_y / delt)
        img_path = os.getcwd()+'/static/tempImage/landChangeImage/'+userId+'/'+time+'/CVA/'
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if delt_x < delt and delt_y < delt:
            rect = [
                [[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y]]]
            region = ee.Geometry.Polygon(rect, None, False)
            geemap.ee_export_image(img,
                                   os.getcwd() + '/static/tempImage/landChangeImage/'
                                   + userId + '/' + time + '/CVA/image_cva_dect_0_0.tif', scale=10,
                                   region=region, file_per_band=False)
        else:
            for i in range(x_number):
                for j in range(y_number):
                    rect=[[[min_x,max_y],[min_x+delt,max_y],[min_x+delt,max_y-delt],[min_x,max_y-delt]]]
                    # print(rect)
                    region=ee.Geometry.Polygon(rect,None, False)

                    geemap.ee_export_image(img,
                                           os.getcwd()+'/static/tempImage/landChangeImage/'
                                           +userId+'/'+time+'/CVA/image_cva_dect_{}_{}.tif'.format(i,j),scale=10,region=region,file_per_band= False)
                    max_y-=delt
                max_y=temp_y
                min_x+=delt
        tifname = 'CVA_change_{}.tif'.format(userId)
        return tifname

    def merge_image(self,userId,tifname,time):
      tifDir = './static/tempImage/landChangeImage/'+userId+'/'+time+'/CVA/'
      name = tifname[:-4] + '.png'
      mergeImage(tifDir, name, 1)
      outtif_LUCC_path = tifDir + name
      name='result_cva.png'
      a = cv2.imread(outtif_LUCC_path, 0)
      outtif_LUCC = './static/tempImage/landChangeImage/' + userId + '/' + time + '/CVA/' + name + ''
      colors = ['white', 'black']
      cmap = m.colors.ListedColormap(colors)
      plt.imsave(arr=a, fname=outtif_LUCC, cmap=cmap)
      url='./static/tempImage/landChangeImage/'+userId+'/'+time+'/CVA/'+name+''
      return url


def get_cva_image(date_start_befor,date_end_befor,date_start_now,date_end_now,border_path,userId,time):
    # try:
        # date_start_befor="2019-01-01"
        # date_end_befor= "2019-12-31"
        # date_start_now="2020-01-01"
        # date_end_now= "2020-12-31"
        # border_path="./static/GeoJson/province/安陆市.json"
        # userId='ypc'
        cva_dect=cva()
        img,border=cva_dect.cva_dectation(date_start_befor,date_end_befor,date_start_now,date_end_now,border_path)
        tifname=cva_dect.load_image(img,border,userId,time)
        url=cva_dect.merge_image(userId,tifname,time)
        return  url
    # except EXCEPTION as e:
    #     print(e.args)
    #     print(traceback.format_exc)
    #     return 'downLoad error'
