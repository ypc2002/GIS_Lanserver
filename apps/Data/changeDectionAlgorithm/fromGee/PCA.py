import ee
import geemap
import os
import json
from .utils import mergeImage,mercatorTolonlat
import cv2
import matplotlib as m
import matplotlib.pyplot as plt
os.environ['PROJ_LIB'] = r'D:\Anaconda\envs\ee\Library\share\proj'
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

class pca:
  # def __init__(self,date_start_befor,date_end_befor,date_start_now,date_end_now,border_path):
  #    self.date_start_befor=date_start_befor
  #    self.date_end_befor=date_end_befor
  #    self.date_start_now=date_start_now
  #    self.date_end_now=date_end_now
  #    self.border_path=border_path

  def getNewBandNames(self,prefix,bandNames):
      seq = ee.List.sequence(1, bandNames.length())
      temp=prefix
      return seq.map(lambda b:ee.String(temp).cat(ee.Number(b).format('%d')))

  def getPrincipalComponents(self,centered, scale, region,bandNames):
      arrays = centered.toArray()
      covar = (arrays.reduceRegion(
        reducer=ee.Reducer.centeredCovariance(),
        geometry=region.geometry().getInfo(),
        scale=scale,
        maxPixels=1e9
      ))
      covarArray = ee.Array(covar.get('array'))
      eigens = covarArray.eigen()
      eigenValues = eigens.slice(1, 0, 1)
      eigenVectors = eigens.slice(1, 1)
      arrayImage = arrays.toArray(1)
      principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage)
      sdImage = (ee.Image(eigenValues.sqrt()) 
        .arrayProject([0]).arrayFlatten([self.getNewBandNames('sd',bandNames)]))
      principalComponents=(principalComponents
        .arrayProject([0]) 
        .arrayFlatten([self.getNewBandNames('pc',bandNames)]) 
        .divide(sdImage))
      return principalComponents

  def sub(self,img1,img2):
    return img2.subtract(img1)

  def Binarization(self,img,region_geo):
      minpix_img=ee.Dictionary(img.reduceRegion(ee.Reducer.min(), region_geo, 10,bestEffort=True,maxPixels=1e9).getInfo()).toImage()
      min=img.subtract(minpix_img)
      pc1=min.select(['pc1'])
      pc2=min.select(['pc2'])
      pc3=min.select(['pc3'])
      pc_temp=pc1.add(pc2)
      pc=pc_temp.add(pc3)
      histogram = pc.reduceRegion(ee.Reducer.histogram(), region_geo, 10,bestEffort=True,maxPixels=1e9)
      th=self.otsu(histogram)
      image=pc.gte(th)
      return image

  def otsu(self,histogram):
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

  def pca_dectation(self,date_start_befor,date_end_befor,date_start_now,date_end_now,border_path):
    # vector_border=json.load(open(border_path,encoding="utf-8"))
    temp=getGeoJsonGeometry(border_path)
    border = list(temp.values())[1][0]
    range = [border]
    print(range)
    region_geo =ee.Geometry.Polygon(range,None, False)
    region = ee.FeatureCollection(region_geo)
    sentImages = (ee.ImageCollection('COPERNICUS/S2_SR')
                      .filterDate(date_start_befor,date_end_befor)
                      .filterBounds(region)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
                      .median()
                      )

    bands=["B1","B2","B3","B4","B5","B6","B7","B8","B9"]            
    sentImage =sentImages.select(bands)
    image =  sentImage.select(bands)
    scale = 10
    bandNames = image.bandNames()
    meanDict = (image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region.geometry().getInfo(),
        scale=scale,
        maxPixels=1e9
      ))
    means = ee.Image.constant(meanDict.values(bandNames))
    centered = image.subtract(means)
    pcImage = self.getPrincipalComponents(centered, scale, region,bandNames)
    pcImage_output =pcImage.select(['pc1', 'pc2', 'pc3'])

    sentImages1 = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate(date_start_now,date_end_now)
                  .filterBounds(region)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
                  .median()
              )
    sentImage1 =sentImages1.select(bands)
    image1 =  sentImage1.select(bands)
    scale1 = 10
    bandNames1 = image1.bandNames()
    meanDict1 = (image1.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region.geometry().getInfo(),
        scale=scale1,
        maxPixels=1e9
      ))
    means1 = ee.Image.constant(meanDict1.values(bandNames1))
    centered1 = image1.subtract(means1)
    pcImage1 = self.getPrincipalComponents(centered1, scale1, region,bandNames1)
    pcImage_output1 =pcImage1.select(['pc1', 'pc2', 'pc3'])

    temp=self.sub(pcImage_output,pcImage_output1)
    pc_result=self.Binarization(temp,region_geo=region_geo)    
    return pc_result,range

  def load_image(self,img,border,userId,time):
    delt=0.04
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
    print(x_number,y_number)
    img_path = os.getcwd()+'/static/tempImage/landChangeImage/'+userId+'/'+time+'/PCA/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if delt_x < delt and delt_y < delt:
        rect = [
            [[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y]]]
        print(rect)
        region = ee.Geometry.Polygon(rect, None, False)
        geemap.ee_export_image(img, os.getcwd() + '/static/tempImage/landChangeImage/' +
                               userId + '/' + time + '/PCA/image_pca_dect_0_0.tif', scale=10,
                               region=region, file_per_band=False)
    else:
        for i in range(x_number):
              for j in range(y_number):
                rect=[[[min_x,max_y],[min_x+delt,max_y],[min_x+delt,max_y-delt],[min_x,max_y-delt]]]
                print(rect)
                region=ee.Geometry.Polygon(rect,None, False)

                geemap.ee_export_image(img,os.getcwd()+'/static/tempImage/landChangeImage/'+
                                       userId+'/'+time+'/PCA/image_pca_dect_{}_{}.tif'.format(i,j),scale=10,region=region,file_per_band= False)
                max_y-=delt
              max_y=temp_y
              min_x+=delt
    tifname='PCA_change_{}.tif'.format(userId)
    return tifname

  def merge_image(self,userId,tifname,time):

      tifDir = './static/tempImage/landChangeImage/'+userId+'/'+time+'/PCA/'
      name = tifname[:-4] + '.png'
      mergeImage(tifDir, name, 1)
      outtif_LUCC_path = tifDir + name
      name='result_pca.png'

      a = cv2.imread(outtif_LUCC_path, 0)
      outtif_LUCC = './static/tempImage/landChangeImage/' + userId + '/' + time + '/PCA/' + name + ''
      colors = ['white', 'black']
      cmap = m.colors.ListedColormap(colors)
      plt.imsave(arr=a, fname=outtif_LUCC, cmap=cmap)
      url = './static/tempImage/landChangeImage/'+userId+'/' + time + '/PCA/' + name + ''
      return url


def get_pca_change(date_start_befor,date_end_befor,date_start_now,date_end_now,border_path,userId,time):
  # try:
    # date_start_befor="2019-01-01"
    # date_end_befor= "2019-12-31"
    # date_start_now="2020-01-01"
    # date_end_now= "2020-12-31"
    # border_path="./static/GeoJson/province/安陆市.json"
    # userId='ypc'
    pc=pca()
    img,border=pc.pca_dectation(date_start_befor,date_end_befor,date_start_now,date_end_now,border_path)
    tifname=pc.load_image(img,border,userId,time)
    # tifname='pca131.tif'
    url=pc.merge_image(userId,tifname,time)
    return  url
  # except EXCEPTION as e:
  #   print(e.args)
  #   print(traceback.format_exc)
  #   return 'downLoad error'
  #
# date_start_befor="2019-01-01"
# date_end_befor= "2019-12-31"
# date_start_now="2020-01-01"
# date_end_now= "2020-12-31"
# border_path="./static/GeoJson/province/安陆市.json"
# userId='ypc'
# time='dsda'
# get_pca_change(date_start_befor,date_end_befor,date_start_now,date_end_now,border_path,userId,time)

