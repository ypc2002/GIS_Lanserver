from flask import Blueprint,current_app
from  .userDBContral  import userManage



user_blue = Blueprint('user', __name__, url_prefix='/user')
userm=userManage(host='127.0.0.1',port='5432',database='test',username='postgres',password='0918')

# userm=userManage(**current_app.config['DATABASE'])
