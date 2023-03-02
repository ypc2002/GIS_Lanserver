import json
from flask_restful import Resource,reqparse
from flask import make_response,request
from apps.ReportingInformation.RF_DBContral import reportInformationManage,reviewRecordManage
from werkzeug.datastructures import FileStorage
from flask import current_app
import jwt
from jwt import PyJWTError

from PIL import Image
import io
import numpy as np
import base64

#初始化不同资源的数据库
reportInformationM=reportInformationManage(host='127.0.0.1',port='5432',database='test',username='postgres',password='0918')
reviewRecordM=reviewRecordManage(host='127.0.0.1',port='5432',database='test',username='postgres',password='0918')
report_picture_path= '../../static/report_picture/'

'''允许上传的图片'''
allowImageType=['png','PNG','jpg','JPG','jpeg','tif']



def verify_tokens(token_str):
    try:
        if isinstance(token_str, str):
            token_str = bytes(token_str, encoding="utf8")
        data=jwt.decode(token_str,key=current_app.config.get('SECRET_KEY'),algorithms=current_app.config.get('ALGORITHM'))
        current_app.logger.info(data)
        print(data.get('exp'))
        return data.get('role')
    except PyJWTError as e:
        current_app.logger.error(e)
        return False

class reportInformation(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()  # 创建一个参数接收器
        self.parser.add_argument('reportInformationID', type=str, help='举报信息编号', location=['form'])
        #用于举报用户的ID查询
        self.parser.add_argument('userID', type=str, help='用户编号', location=['form','args'])
        # 用于时间段查询 location默认为values
        self.parser.add_argument('startTime', type=str, help='起始时间',location=['form'])
        self.parser.add_argument('endTime', type=str, help='结束时间', location=['form'])
        # 用于通过面区域查询
        self.parser.add_argument('range', type=str, location=['form'])
        self.parser.add_argument('inform_location', type=str, help='举报发生的行政区地点',location=['form'])  #举报地点
        self.parser.add_argument('inform_time', type=str,location=['form'])  #举报时间
        self.parser.add_argument('inform_reason', type=str, help='其输入举报原因',location=['form'])  #举报原因
        self.parser.add_argument('coordination', type=str,location=['form'])  #举报经纬度
        self.parser.add_argument('inform_description', type=str,location=['form'])  #详细描述
        self.parser.add_argument('inform_picture', type=FileStorage,location='files')  #举报图片路径 根据举报地点时间自己控制


    def get(self):
        args = self.parser.parse_args()  # 解析参数
        ID=args.get('reportInformationID')
        userID=args.get('userID')
        startTime=args.get('startTime')
        endTime=args.get('endTime')
        inform_location=args.get('inform_location')
        range=args.get('range')
        inform_reason=args.get('inform_reason')
        inform_description=args.get('inform_description')
        reportInformations=None
        if ID!=None:
            reportInformations=reportInformationM.findInformationByID(ID)
        elif userID!=None:#通过举报用户的ID查询
            reportInformations=reportInformationM.findInformationByUserID(userID)
        elif startTime!=None and endTime!=None:#通过举报时间段查询
            reportInformations=reportInformationM.findInformationByTime(startTime=startTime,endTime=endTime)
        elif inform_location!=None:#通过举报位置查询
            reportInformations=reportInformationM.findInformationByLocation(location=inform_location)
        elif range!= None:
            range=json.loads(range)
            reportInformations=reportInformationM.findInformationByRange(range=range)
        elif inform_reason!=None:#通过举报原因查询
                pass
        elif inform_description!=None:#通过举报描述查询
                pass
        else:
            return make_response({"错误": "无该查询方式"}, 400)

        # try:
        #     # 传输图片
        #     for reportInformation in reportInformations:
        #         print(reportInformation)
        #         with open(reportInformation['照片路径'], 'rb') as img_f:
        #             print('---------------------------------------')
        #             img_stream = img_f.read()
        #             img_stream = base64.b64encode(img_stream)
        #             print(type(img_stream))
        #             print(img_stream)
        #             del reportInformation['照片路径']
        #             reportInformation['举报图片流'] = img_stream
        #     return make_response({"reprotInformations": reportInformations}, 200)
        # except:
        #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #     return make_response({"reprotInformations": reportInformations}, 200)
        return make_response({"reprotInformations": reportInformations}, 200)

    def post(self):
        '''
        插入举报信息
        :return:
        '''

        args = self.parser.parse_args()  # 解析参数
        dome_con_id=args.get('userID')
        inform_location=args.get('inform_location')
        inform_time=args.get('inform_time')
        inform_reason=args.get('inform_reason')
        coordination=args.get('coordination').split(',')
        inform_description=args.get('inform_description')
        inform_picture=args.get('inform_picture')
        print()
        inform_location_=inform_location.split('省')
        inform_picture_path='/static/report_picture/{}/{}'.format(inform_location_[0],inform_picture.filename)
        coordination=[float(coordination[0][1:]),float(coordination[1][:-1])]
        inform_picture.save(inform_picture_path)
        # print(inform_picture)
        # inform_picture_bin=inform_picture.read()
        # print("二进制:",inform_picture_bin)
        # image=Image.open(io.BytesIO(inform_picture_bin))
        # print("PIL的打开",image)
        # print("矩阵:",np.array(image))

        if reportInformationM.insertInformation(dome_con_id,inform_location,inform_time,inform_reason,coordination,inform_description,inform_picture_path)==True:
            return make_response({"举报状态":"成功，等待管理员审核！"},200)
        else:
            return make_response({"举报状态": "失败"}, 400)



    def delete(self):
        args=self.parser.parse_args()
        ID = args.get('reportInformationID')  # 解析参数
        # if verify_tokens(request.cookies.get('token'))!= '管理员':
        if verify_tokens(request.form.get('token')) != '管理员':
            return make_response('无权限,请使用管理员身份登录后进行该操作!', 400)
        if ID==None:
            return make_response({"状态": "删除失败!！","原因":"缺少举报信息编号"}, 400)
        elif reportInformationM.findInformationByID(ID)!=[] and reportInformationM.deleteOneByID(ID)!=None:
            return make_response({"状态":"删除成功！"},200)
        else:
            return make_response({"状态":"该记录不存在或已被删除！"},400)

    def put(self):
        args = self.parser.parse_args()  # 解析参数
        ID=args.get('reportInformationID')
        # dome_con_id = args.get('userID')
        inform_location = args.get('inform_location')
        inform_time = args.get('inform_time')
        inform_reason = args.get('inform_reason')
        coordination = args.get('coordination').split(',')
        inform_description = args.get('inform_description')
        inform_picture = args.get('inform_picture')
        #构建照片存储路径
        inform_location_=inform_location.split('省')
        inform_picture_path = '/static/report_picture/{}/{}'.format(inform_location_[0],inform_picture.filename)
        #保存照片
        inform_picture.save(inform_picture_path)
        coordination = [float(coordination[0][1:]), float(coordination[1][:-1])]
        if reportInformationM.updata(ID,inform_location,inform_time,inform_reason,coordination,inform_description,inform_picture_path)==True:
            return make_response({"状态":"更新成功！"},200)
        else:
            return make_response({"状态": "更新失败！"},400)





class reviewRecord(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()  # 创建一个参数接收器
        #时间段查询
        self.parser.add_argument('startTime', type=str, help='起始时间',location='form')
        self.parser.add_argument('endTime', type=str, help='结束时间',location='form')
        # 管理员ID查询
        self.parser.add_argument('adminID', type=str,location='form')
        #位置查询
        self.parser.add_argument('location',type=str,location='form')
        #矢量区域查询
        self.parser.add_argument('range',type=str,location='form')
        #插入举报信息的属性
        self.parser.add_argument('adminID', type=str, location='form')
        self.parser.add_argument('informID', type=str, location='form')
        self.parser.add_argument('examine_time', type=str, location='form')
        self.parser.add_argument('inform_fact', type=str, location='form')
        #cookies中的认证信息
        self.parser.add_argument('role',type=str,location='cookies')
        self.parser.add_argument('Login', type=str, location='cookies')


    def get(self):
        args = self.parser.parse_args()  # 解析参数
        adminID=args.get('adminID')
        informID=args.get('informID')
        location=args.get('location')
        range=args.get('range')
        startTime=args.get('startTime')
        endTime=args.get('endTime')
        print(adminID)
        if adminID != None:
            return make_response({"审查记录":reviewRecordM.findRecordByAdminID(adminID)},200)
        elif informID!=None:
            return make_response({"审查记录": reviewRecordM.findRecordByInformID(informID=informID)}, 200)
        elif location!= None:
            return make_response({"审查记录": reviewRecordM.findRecordByLocation(location=location)}, 200)
        elif range!=None:
            range = json.loads(range)
            return make_response({"审查记录": reviewRecordM.findRecordByRange(range=range)}, 200)
        elif startTime!=None and endTime!=None:
            return make_response({"审查记录": reviewRecordM.findRecordByTime(startTime=startTime,endTime=endTime)}, 200)
        else:
            return make_response({"错误":"无该查询方式"},400)


    def post(self):
        args = self.parser.parse_args()  # 解析参数
        print(args)
        # if verify_tokens(request.cookies.get('token'))!='管理员':
        if verify_tokens(request.form.get('token')) != '管理员':
            return make_response('无权限,请使用管理员身份登录后进行该操作!', 400)
        admin_id=args.get('adminID')
        inform_id=args.get('informID')
        examine_time=args.get('examine_time')
        inform_fact=args.get('inform_fact')
        if admin_id!=None and  inform_id!=None and inform_fact!=None and examine_time!=None:
             if reviewRecordM.insertRecord(admin_id=admin_id,inform_id=inform_id,inform_fact=inform_fact, examine_time=examine_time)==True:
                 return make_response({"状态":"审查记录上传成功"},200)
             else:
                 return make_response({"状态":"审查记录上传失败"},400)
        else:
            return make_response({"状态":"审查记录上传错误","原因":"确实必要信息"},400)

    def delete(self):
        args = self.parser.parse_args()  # 解析参数
        # if verify_tokens(request.cookies.get('token'))!= '管理员':
        if verify_tokens(request.form.get('token')) != '管理员':
            return make_response('无权限,请使用管理员身份登录后进行该操作!', 400)
        adminID=args.get('adminID')
        informID=args.get('informID')
        if adminID!=None and informID!=None:
            if reviewRecordM.delete(admin_id=adminID,inform_id=informID)==True:
                return make_response({"是否删除":"是"},200)
            else:
                return make_response({"是否删除": "否","原因":"不存在该记录或该记录已被删除"}, 400)
        else:
            return make_response({"是否删除": "否", "原因": "缺少必要信息"}, 400)

    def put(self):
        args = self.parser.parse_args()  # 解析参数
        # if verify_tokens(request.cookies.get('token')) != '管理员':
        if verify_tokens(request.form.get('token')) != '管理员':
            return make_response('无权限,请使用管理员身份登录后进行该操作!', 400)
        admin_id = args.get('adminID')
        inform_id = args.get('informID')
        examine_time = args.get('examine_time')
        inform_fact = args.get('inform_fact')
        if reviewRecordM.updata(admin_id=admin_id,inform_id=inform_id,examine_time=examine_time,inform_fact=inform_fact)==True:
            return make_response({"更新状态":"成功"},200)
        else:
            return make_response({"更新状态": "失败"}, 400)


