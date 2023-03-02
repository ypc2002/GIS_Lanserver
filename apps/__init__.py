import sys
import os
folderPath=os.getcwd()
childrenFolderPath=os.listdir(folderPath)
sys.path.extend([folderPath+'\\'+childrenFolderPath[i] for i in range(len(childrenFolderPath)-1)])


from flask import Flask
#扩展
from flask_login import LoginManager
from flask_mail import Mail
from flask_cors import CORS

#配置
import setting
#蓝图
from User.view import user_blue
from Data import data_blue
from ReportingInformation import reInformation_blue
#RESTful_API
# from Data.view import ImageData,GeoJson

#初始化扩展对象
login_manager = LoginManager()  # 实例化登录管理对象
mail = Mail()
cors=CORS(supports_credentials=True,resources=r'/*')#解决跨域




def create_app(config_name=None):
   #初始化app
   app=Flask(__name__,static_folder="../static",template_folder="../templates")
   #添加扩展
   login_manager.init_app(app)
   login_manager.login_view = 'login'  # 设置用户登录视图函数 endpoint
   cors.init_app(app)

   # CORS(app,supports_credentials=True,resources=r'/*')
   mail.init_app(app)
   #设置配置参数
   app.config.from_object(setting.DevelopmentConfig)
   #合并蓝图
   app.register_blueprint(user_blue)
   app.register_blueprint(data_blue)
   app.register_blueprint(reInformation_blue)


   return app



app=create_app()



@app.route('/')
def index():
    return "hellePage!"


