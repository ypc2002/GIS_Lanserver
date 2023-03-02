from flask import Blueprint
from flask_restful import Api

from apps.ReportingInformation.view import reportInformation,reviewRecord

#蓝图
reInformation_blue=Blueprint('reInformation',__name__,url_prefix="/reportInformation")
#蓝图的restful API
api=Api(reInformation_blue)
#添加路径
api.add_resource(reportInformation, '/reportRecords')
api.add_resource(reviewRecord, '/reviewRecord')

