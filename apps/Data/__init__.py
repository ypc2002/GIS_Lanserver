from flask import Blueprint
from flask_restful import Api
from .view import LandUseImage,illegalRegion,SourceImage,ChangeRegion
#蓝图
data_blue=Blueprint("data", __name__, url_prefix="/Data")
#蓝图的restful API
api=Api(data_blue)
#添加类视图url
api.add_resource(LandUseImage, '/LandUseImage')
api.add_resource(illegalRegion, '/illegalRegion')

api.add_resource(SourceImage, '/SourceImage')#即下载即返回（在服务器临时存储，若服务器足够大则可以永久存储）
api.add_resource(ChangeRegion,'/ChangeRegion')