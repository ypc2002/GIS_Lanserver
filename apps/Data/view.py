'''系统相关库'''
import traceback
'''flask相关依赖'''
from flask_restful import Resource,reqparse
from flask import make_response,current_app,request
from werkzeug.datastructures import FileStorage
import jwt
from jwt import PyJWTError
'''数据库层'''
from .dataDBControl import illegalRegionManage
'''从GEE上下载数据的依赖'''
from .imageAcquire import *
from .changeDectionAlgorithm.fromGee.cva import *
from .changeDectionAlgorithm.fromGee.PCA import *
import time
from .LandUseAlgorithm.fromUser.DL import landUseExtract
from .LandUseAlgorithm.fromUser.DL import models as landUseModels
from .LandUseAlgorithm.fromGee.remoteIndex import figure_extract
'''处理用户上传的影像'''
from .LandUseAlgorithm.fromUser.remoteIndex import RSIndex
from .changeDectionAlgorithm.fromUser.DL import changeDection
from .changeDectionAlgorithm.fromUser.DL import models as changeDectionModels




illegalRegionM=illegalRegionManage(host='127.0.0.1',port='5432',database='test',username='postgres',password='0918')
changeDectionInstance=changeDection()#实例化变化检测对象
landUseInstance=landUseExtract()#实例化地物提取对象
'''下载影像访问队列'''

def verify_tokens(token_str):
    try:
        if isinstance(token_str, str):
            token_str = bytes(token_str, encoding="utf8")
            print(type(token_str))
        print(type(token_str))
        data=jwt.decode(token_str,key=current_app.config.get('SECRET_KEY'),algorithms=current_app.config.get('ALGORITHM'))
        current_app.logger.info(data)
        print(data.get('exp'))
        return data.get('role')
    except PyJWTError as e:
        current_app.logger.error(e)
        return False


class illegalRegion(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()  # 创建一个参数接收器
        self.parser.add_argument('role', type=str, help='身份权限', location='form')
        self.parser.add_argument('token', type=str, help='token认证', location='form')
        #插入违法区域参数:全部  更新记录信息参数:
        self.parser.add_argument('illegal_time', type=str, help='当前时间',location='form')
        self.parser.add_argument('illegal_location', type=str, help='地点', location=['form','args'])
        self.parser.add_argument('illegal_reason', type=str, help='原因', location=['form','args'])
        self.parser.add_argument('vector_boundary', type=str, help='矢量区域', location='form')
        self.parser.add_argument('illegal_description', type=str, help='描述', location=['form','args'])
        self.parser.add_argument('admin_id', type=str, help='管理员唯一编号', location='form')
        self.parser.add_argument('illegal_raster',type=FileStorage,help='违法区域栅格图',location='files')
        self.parser.add_argument('illegal_coordinate', type=str, help='违法区域坐标', location='form')
        # 查询参数全部   删除参数regionID
        self.parser.add_argument('startTime', type=str, help='起始时间',location='args')  # 用于时间段查询的起始时间
        self.parser.add_argument('endTime', type=str, help='终止时间', location='args')
        self.parser.add_argument('regionID', type=str, help='违法区域编号', location='args')
        self.parser.add_argument('range', type=str, help='矢量边界', location='args')
        self.parser.add_argument('adminID', type=str, help='管理员唯一编号', location='args')
        self.parser.add_argument('location', type=str, help='地点', location='args')
    def post(self):
        args = self.parser.parse_args()  # 解析参数
        print('LOGIN',args.get('Login'))
        # if verify_tokens(request.cookies.get('token'))!='管理员':
        if verify_tokens(args.get('token')) != '管理员':
            return make_response('请使用管理员身份登录后上传!', 400)

        illegal_time=args.get('illegal_time')
        illegal_location=args.get('illegal_location')
        illegal_reason=args.get('illegal_reason')

        vector_boundary=args.get('vector_boundary')
        vector_boundary=json.loads(vector_boundary)

        illegal_description=args.get('illegal_description')
        admin_id=args.get('admin_id')
        illegal_raster=args.get('illegal_raster')

        illegal_coordinate=args.get('illegal_coordinate')
        illegal_coordinate=json.loads(illegal_coordinate)

        illegal_raster_path=illegal_raster
        if illegal_raster!=None:
            illegal_location_= illegal_location.split('省')
            provice=illegal_location_[0]
            illegal_location__=illegal_location_[1].split('市')
            city=illegal_location__[0]
            filename=illegal_raster.filename
            illegal_raster_path='/static/illageRegionRaster/{}/{}/{}_{}'.format(provice,city,illegal_location__[1],filename)
            illegal_raster.save(illegal_raster_path)
        if illegalRegionM.insert(illegal_time=illegal_time, illegal_location=illegal_location, illegal_reason=illegal_reason,
                                 vector_boundary=vector_boundary, illegal_description=illegal_description, illegal_raster_path=illegal_raster_path,
                                 illegal_coordinate=illegal_coordinate, admin_id=admin_id):
            return make_response({'上传状态':'成功'},200)

        else:
            return make_response({'上传状态': '失败'}, 400)


    def get(self):
        args = self.parser.parse_args()  # 解析参数
        startTime=args.get('startTime')
        endTime=args.get('endTime')
        regionID = args.get('regionID')
        range = args.get('range')
        adminID = args.get('adminID')
        location = args.get('location')
        illegalRegions=None
        if startTime!=None and endTime!=None and location!=None:
            illegalRegions=illegalRegionM.findRegionByTimeAndLocation(startTime=startTime,endTime=endTime,location=location)
        elif startTime!=None and endTime!=None and range!=None:
            illegalRegions=illegalRegionM.findRegionByTimeAndRange(startTime=startTime,endTime=endTime,range=range)
        elif startTime!=None and endTime!=None:
            illegalRegions=illegalRegionM.findRegionByTime(startTime=startTime,endTime=endTime)
        elif regionID!=None:
            illegalRegions = illegalRegionM.findRegionByRegionID(RegionID=regionID)
        elif range!=None:#未测试
            range=json.loads(range)
            illegalRegions=illegalRegionM.findRegionByRange(range=range)
        elif location!=None:
            illegalRegions=illegalRegionM.findRegionByLocation(location=location)
        elif adminID!=None:
            illegalRegions=illegalRegionM.findRegionByAdminID(adminID=adminID)
        else:
            make_response({"状态":"请求错误","原因":"查询方式错误"}, 200)
        return make_response({"违法区域记录": illegalRegions}, 400)


    def delete(self):
        args = self.parser.parse_args()  # 解析参数
        # if verify_tokens(request.cookies.get('token'))!= '管理员':
        if verify_tokens(request.form.get('token')) != '管理员':
            return make_response('无权限,请使用管理员身份登录后进行该操作!', 400)
        regionID = args.get('regionID')
        if illegalRegionM.deleteByRegionID(illegalRegionID=regionID):
            return make_response({'状态':'删除成功!'},200)
        return make_response({'状态':'删除失败!'},400)

    def put(self):
        args = self.parser.parse_args()  # 解析参数
        # if verify_tokens(request.cookies.get('token'))!= '管理员':
        if verify_tokens(request.form.get('token')) != '管理员':
            return make_response('无权限,请使用管理员身份登录后进行该操作!', 400)
        illegalRegionID=args.get('regionID')
        # illegalRegionRecord=illegalRegionM.findRegionByRegionID(RegionID=illegalRegionID)
        illegal_location=args.get('illegal_location')
        illegal_reason=args.get('illegal_reason')
        illegal_description=args.get('illegal_description')
        print()
        if illegalRegionM.updata(illegalRegionID=illegalRegionID,illegal_location=illegal_location,
                                 illegal_reason=illegal_reason,illegal_description=illegal_description):
            return make_response({'状态':'更新成功'},200)
        return make_response({'状态':'更新失败'},400)



'''返回原始遥感影像API'''
class SourceImage(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()  # 创建一个参数接收器
        self.parser.add_argument('token', type=str, help='token认证', location=['form','args'])
        self.parser.add_argument('startTime', type=str, help='起始时间',location=['form','args'])  # 用于时间段查询的起始时间
        self.parser.add_argument('endTime', type=str, help='终止时间', location=['form','args'])#终止时间
        self.parser.add_argument('location', type=str, help='地点', location=['form','args'])#地点
        self.parser.add_argument('range', type=str, help='矢量边界', location=['form','args'])#矢量范围
        self.parser.add_argument('adminID', type=str, help='管理员唯一编号', location=['form','args'])#

    def get(self):
        # if verify_tokens(request.cookies.get('token'))!='管理员':

        args = self.parser.parse_args()  # 解析参数
        # print(args)
        # if verify_tokens(args.get('token')) != '管理员':
        #     return make_response('请使用管理员身份登录后上传!', 400)

        startTime = args.get('startTime')
        endTime = args.get('endTime')
        range = args.get('range')
        adminID = args.get('adminID')
        print(range)
        range = json.loads(range)
        print(range)
        time_request = str(time.time())
        url = get_sourceimage(startTime, endTime, range, 0, adminID, time_request,'none')
        return make_response({"SourceImage": url}, 200)

    def post(self):
        args = self.parser.parse_args()  # 解析参数
        return {"SourceImage":"post"}

    def delete(self):
        args = self.parser.parse_args()  # 解析参数
        return {"SourceImage":"deleta"}

    def put(self):
        args = self.parser.parse_args()  # 解析参数
        return {"SourceImage":"put"}



'''返回土地利用图像API'''
class LandUseImage(Resource):
    def __init__(self):
        self.saveBaseDir='./static/tempImage/landUseImage/'
        # 创建一个参数接收器
        self.parser = reqparse.RequestParser()
        #公共参数
        self.parser.add_argument('adminID', type=str, help='管理员唯一编号', location=['form','args'])  # # 用于时间段查询的起始时间
        self.parser.add_argument('token', type=str, help='token认证', location=['form','args'])
        self.parser.add_argument('landType', type=str, location=['form','args'],
                                 help='想要提取的土地利用类型,可取值:building、water、plant、road')
        # 用户上传影像解析
        self.parser.add_argument('image', type=FileStorage, location='files', help='需要提取的图像')  # 用于时间段查询的起始时间
        # 用户请求影像自动解析
        self.parser.add_argument('startTime', type=str, help='起始时间', location=['form','args'])  # 用于时间段查询的起始时间
        self.parser.add_argument('endTime', type=str, help='终止时间', location=['form','args'])  # 终止时间
        self.parser.add_argument('range', type=str, help='矢量边界', location=['form','args'])  # 矢量范围


    def get(self):
        # if verify_tokens(request.cookies.get('token'))!='管理员':
        args = self.parser.parse_args()  # 解析参数
        print(args)
        token=args.get('token')
        print(token)
        if verify_tokens(token) != '管理员':
            return make_response('请使用管理员身份登录后上传!',400)

        startTime = args.get('startTime')
        endTime = args.get('endTime')
        range = args.get('range')
        adminID = args.get('adminID')
        landType=args.get('landType')
        range = json.loads(range)
        print(range)
        url=''
        try:
            time_request = str(time.time())
            if landType=='label':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)
                return make_response({"LandUseImage": url,
                                      "BangName": ['water', 'trees', 'grass', 'flooded_vegetation', 'crops',
                                                   'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'],
                                      "color": ['419BDF', '397D49', '88B053', '7A87C6', 'E49635', 'DFC35A', 'C4281B',
                                                'A59B8F', 'B39FE1']}, 200)
            elif landType=='water':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='trees':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='grass':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='flooded_vegetation':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='crops':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='shrub_and_scrub':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='built':
                print('build')
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='bare':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)

            elif landType=='snow_and_ice':
                url = get_sourceimage(startTime, endTime, range, 1, adminID, time_request,landType)
            return make_response({"LandUseImage": url}, 200)
            # elif landType=='water':
            #     url = figure_extract(startTime, endTime, range, adminID, time_request, landType)
            # elif landType=='plant':
            #     url = figure_extract(startTime, endTime, range, adminID, time_request, landType)
            # elif landType=='build':
            #     url = figure_extract(startTime, endTime, range, adminID, time_request, landType)

            # return make_response({"LandUseImage": url}, 200)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return make_response({"状态":'失败'}, 400)


    '''解析上传的影像'''
    #需修改
    def post(self):
        # if verify_tokens(request.cookies.get('token'))!=
        args = self.parser.parse_args()  # 解析参数
        print(args)
        if verify_tokens(args.get('token')) != '管理员':
            return make_response('请使用管理员身份登录后上传!',400)

        adminID=args.get('adminID')
        landType = args.get('landType')
        image = args.get('image')
        try:
            '''创建文件夹保存图片'''
            path=self.saveBaseDir+adminID+'/' + str(time.time())
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)#递归创建文件夹
            saveSourceImagePath=path+'/'+image.filename#上传影像的保存地址
            print(saveSourceImagePath)
            image.save(saveSourceImagePath)#保存上传的影像
        except:
            return make_response({'状态':'失败','原因':'图片解析失败'},400)
        '''提取'''
        try:
            if landType=='water':
                resultPath = path + '/' + image.filename.split('.')[0] + '_' + 'waterMask.png'
            elif landType=='plant':
                resultPath = path + '/' + image.filename.split('.')[0] + '_' + 'plantMask.png'
            elif landType=='building':
                resultPath=path+'/'+image.filename.split('.')[0]+'_'+'buildingMask.png'
                landUseInstance.extract(modelKey=landUseModels.unetPP_building.value,
                                        input_path=saveSourceImagePath,
                                        resultPath=resultPath)
            return make_response({'提取结果url':resultPath},200)
        except:
            return make_response({'状态': '失败','原因':'提取过程失败'}, 400)




    def delete(self):
        args = self.parser.parse_args()  # 解析参数
        return {"LandUseImage":"deleta"}

    def put(self):
        args = self.parser.parse_args()  # 解析参数
        return {"LandUseImage":"put"}






'''返回地物变化影像API'''
class ChangeRegion(Resource):
    def __init__(self):
        #解析方法
        self.parser = reqparse.RequestParser()  # 创建一个参数接收器
        #changeType:water、plant、impervious、cover
        self.parser.add_argument('token', type=str, help='token认证', location=['args','form'])
        self.parser.add_argument('changeType', type=str, location=['form','args'], help='选择查看土地类型变化，可以取值water、plant、impervious、cover等')
        #用户上传影像解析
        self.parser.add_argument('Image1', type=FileStorage,location='files', help='较早时间段的影像')  # 用于时间段查询的起始时间
        self.parser.add_argument('Image2', type=FileStorage, location='files', help='较晚时间段的影像')
        #用户请求影像自动解析
        self.parser.add_argument('Image1startTime', type=str, location=['form','args'], help='起始时间')
        self.parser.add_argument('Image1endTime', type=str, location=['form','args'], help='终止时间')
        self.parser.add_argument('Image2startTime', type=str, location=['form','args'], help='起始时间')
        self.parser.add_argument('Image2endTime', type=str, location=['form','args'], help='终止时间')

        self.parser.add_argument('range', type=str, help='矢量边界', location=['form','args'])  # 矢量范围
        self.parser.add_argument('adminID', type=str, help='管理员唯一编号', location=['form','args']) # # 用于时间段查询的起始时间
        # self.parser.add_argument('dectation', type=str, help='检测方法', location=['form'])

        self.saveBaseDir='static/tempImage/landChangeImage/'


    def get(self):
        # if verify_tokens(request.cookies.get('token'))!='管理员':
        args = self.parser.parse_args()  # 解析参数
        print(args)
        if verify_tokens(args.get('token')) != '管理员':
            return make_response('请使用管理员身份登录后上传!', 400)

        # 自动获取影像解析参数
        Image1startTime = args.get('Image1startTime')
        Image1endTime = args.get('Image1endTime')
        Image2startTime = args.get('Image2startTime')
        Image2endTime = args.get('Image2endTime')
        range = args.get('range')
        adminID = args.get('adminID')

        range = json.loads(range)
        time_request = str(time.time())
        # print(range)
        changeType = args.get('changeType')
        try:
            if changeType == 'water':
                pass
            elif changeType == 'plant':
                pass
            elif changeType == 'impervious':
                pass
            elif changeType == 'cover':
                print(changeType)
                url = get_pca_change(Image1startTime, Image1endTime, Image2startTime, Image2endTime, range, adminID,
                                 time_request)
                # url=get_cva_image(Image1startTime, Image1endTime, Image2startTime, Image2endTime, range, adminID,
                #                  time_request)
            return make_response({"resultImageUrl": url}, 200)
        except:
            return make_response({"resultImageUrl": None}, 400)

#用户传来两张时序影像，解析后返回变化结果图
    def post(self):
        args = self.parser.parse_args()  # 解析参数
        # if verify_tokens(request.cookies.get('token'))!='管理员':
        if verify_tokens(args.get('token')) != '管理员':
            return make_response('请使用管理员身份登录后上传!',400)

        #上传影像解析所需参数
        Image1=args.get('Image1')
        Image2=args.get('Image2')
        adminID=args.get('adminID')
        changeType=args.get('changeType')

        '''创建文件夹保存图片'''
        path = self.saveBaseDir + adminID + '/' +  str(time.time())
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)  # 递归创建文件夹
        saveSourceImage1Path = path + '/' + Image1.filename  # 上传影像的保存地址
        saveSourceImage2Path = path + '/' + Image2.filename  # 上传影像的保存地址
        changeResultImagePath=path+'/'+Image1.filename.split('.')[0]+'_'+Image2.filename.split('.')[0]+'_changeResult.png'
        Image1.save(saveSourceImage1Path)  # 保存上传的影像
        Image2.save(saveSourceImage2Path)  # 保存上传的影像

        if Image1!=None and Image2!=None:
            #根据ID创建一个临时文件夹存放,将图片放入临时文件夹中
            #读取图片
            isSucceed=False
            if changeType=='water':
                pass
            elif changeType=='plant':
                pass
            elif changeType=='impervious':
                pass
            elif changeType=='cover':
                isSucceed=changeDectionInstance.predect(modelKey=changeDectionModels.unetPP.value,image1Path=saveSourceImage1Path,
                                              image2Path=saveSourceImage2Path,resultPath=changeResultImagePath)
            else:
                pass
        if isSucceed:
            return make_response({"变化检测结果url": changeResultImagePath},200)
        else:
            return make_response({'状态':'检测失败，图片不符合要求！'},300)

    def delete(self):
        args = self.parser.parse_args()  # 解析参数
        return {"SourceImage": "deleta"}

    def put(self):
        args = self.parser.parse_args()  # 解析参数
        return {"SourceImage": "put"}
