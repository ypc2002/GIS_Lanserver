import werkzeug
from flask import make_response,request,jsonify
from . import user_blue,userm


import jwt
# from comment.utils import const
from datetime import datetime,timedelta
from jwt import PyJWTError
from flask import current_app



def generate_tokens(userID,role):
    '''
    一个用户在一次会话生成一个token
    :param uid: 用户id
    :return:
    '''
    # params：是生成token的参数
    # exp：代表token的有效时间,datetime.utcnow():代表当前时间
    # timedelta:表示转化为毫秒
    params = {
        'userID': userID,
        'role': role,
        'exp': datetime.utcnow() + timedelta(seconds=current_app.config.get('JWT_EXPIRY_SECOND'))
    }
    # key:密钥,
    # algorithm:算法，算法是SHA-256
    # SHA-256:密码散列函数算法.256字节长的哈希值（32个长度的数组）---》16进制字符串表示，长度为64。信息摘要，不可以逆
    return jwt.encode(payload=params, key=current_app.config.get('SECRET_KEY'),algorithm=current_app.config.get('ALGORITHM'))


def verify_tokens(token_str):
    try:
        if isinstance(token_str,str):
            token_str=bytes(token_str,encoding = "utf8")
            print(type(token_str))
        print(type(token_str))
        data=jwt.decode(token_str,key=current_app.config.get('SECRET_KEY'),algorithms=current_app.config.get('ALGORITHM'))
        current_app.logger.info(data)
        print(data.get('exp'))
        return data.get('role')
    except PyJWTError as e:
        current_app.logger.error(e)
        return False


from itsdangerous import TimedSerializer as Serializer
from itsdangerous import BadSignature, SignatureExpired



# def generate_token(user, operation, **kwargs):
#     """生成用于邮箱验证的JWT（json web token）"""
#     s = Serializer(config['SECRET_KEY'], expire_in)
#     # 待签名的数据负载
#     data = {'id': user.id, 'operation': operation}
#     data.update(**kwargs)
#     return s.dumps(data)

# def validate_token(user, token, operation):
#     """用于验证用户注册的token, 并完成相应的确认操作"""
#     s = Serializer(config['SECRET_KEY'])
#     try:
#         data = s.loads(token)
#     except (SignatureExpired, BadSignature):
#         return False
#     ... # 相关字段确认
#     return True


@user_blue.route('/')
def user_center():
    print("用户接口中心！")
    return "aaaaaaaaaaaaaaaaaaaaaa"

#优化：注册完之后返回登录页面
@user_blue.route('/register',methods=['POST'])
def register():
    if request.method == 'POST':
        print("here")
        name=request.form.get('username')
        account = request.form.get('account')
        password = request.form.get('password')
        email=request.form.get('email')
        role=request.form.get('role')
        precinct = request.form.get('precinct')
        print(role)
        print("login……")
        if len(password) < 20:
            if role=='管理员':
                if userm.findManagerByAccount(admin_account=account)==None:
                    password = werkzeug.security.generate_password_hash(password)
                    if userm.insertManager(admin_name=name,admin_account=account,admin_password=password,admin_email=email,precinct=precinct):
                        return make_response({"注册状态":"成功注册！"},200)
                    else:
                        make_response({"注册状态": "注册失败"}, 300)
                else:
                    return make_response({"注册状态":"已存在该用户！"},301)
            elif role=='普通用户':
                if userm.findOrdinaryUserByAccount(dome_con_account=account)==None:
                    password = werkzeug.security.generate_password_hash(password)
                    if userm.insertOrdinaryUser(dome_con_name=name,dome_con_account=account,dome_con_password=password,dome_con_email=email):
                        return make_response({"注册状态":"成功注册！"},200)
                    else:
                        make_response({"注册状态": "注册失败！"}, 300)
                else:
                    return make_response({"注册状态":"已存在该用户！"},301)
            else:
                return make_response({"注册状态":"请正选择身份"},302)
        else:
            return make_response({"注册状态": "失败！", "失败原因:": "密码过长！"}, 303)

    else:
        return make_response({"状态":"错误请求方式"},304)



@user_blue.route('/login',methods=['POST'])
def login():
    if request.method=='POST':

        account = request.form.get('account')
        password = request.form.get('password')
        role=request.form.get('role')
        print(account,password,role)
        user=None
        if role=='管理员':
            user=userm.findManagerByAccount(admin_account=account)
        elif role=='普通用户':
            user=userm.findOrdinaryUserByAccount(dome_con_account=account)
        if user!=None:
            # session.permanent = True #session生命周期生效
            userID=user[0]
            userName=user[1]
            userAccount=user[2]
            pwhash=user[3]
            email=user[4]

            #将密码加密存入数据库
            if werkzeug.security.check_password_hash(pwhash,password):
                if not verify_tokens(request.cookies.get('token')):  # 浏览器中token过期或者不含token,则用户可以再次登录
                    # response = make_response({'状态':'登陆成功'},200)
                    # response.set_cookie(key='userID',value=str(userID))
                    # response.set_cookie(key='userName',value=userName)
                    # response.set_cookie(key='account',value=userAccount)
                    # response.set_cookie(key='role',value=role)
                    # response.set_cookie(key='email', value=email)


                    if role=='管理员':
                        # response.set_cookie(key='precinct', value=precinct)
                        # response.set_cookie(key='token',value=generate_tokens(userID=userID,role=role))
                        precinct = user[5]
                        return make_response({'状态': '登陆成功',
                                              'userID':userID,
                                              'userName':userName,
                                              'account':account,
                                              'role':role,
                                              'email':email,
                                              'precinct':precinct,
                                              'token': str(generate_tokens(userID=userID, role=role))}, 200)
                    # , encoding = "utf-8"
                    elif role=='普通用户':
                        # response.set_cookie(key='precinct', value=None)
                        # response.set_cookie(key='token', value=generate_tokens(userID=userID,role=role))
                        return make_response({'状态': '登陆成功',
                                              'userID': userID,
                                              'userName':userName,
                                              'account':account,
                                              'role':role,
                                              'email':email,
                                              'token': str(generate_tokens(userID=userID, role=role),encoding = "utf-8")},200)
                else:
                    return make_response({'状态': '用户已经登录'}, 300)
            else:
                return make_response({'状态': '登陆失败','原因':'用户账号或密码错误'},303)
        else:
            print("用户不存在！")
            return make_response({'状态': '登陆失败','原因':'用户不存在'},301)

    return make_response({"状态":"错误请求方式"},302)


@user_blue.route('/logout',methods=['POST'])
def logout():
    if request.method=='POST':
        # print(request)
        if verify_tokens(request.form.get('token')):
            # print(request.cookies.get('token'))
            response=make_response({"状态":"退出登录"},200)
            # response.set_cookie(key='token',value='invalid token')
            return response
        else:
            return make_response({"状态": "错误，没有登录，不存在退出"}, 300)



@user_blue.route('/delete',methods=['POST'])
def delete():
    pass


@user_blue.route('/modifyInformation',methods=['POST'])
def modifyInformation():

    return



