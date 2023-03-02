class baseConfig:
    #配置邮箱验证
    MAIL_SERVER='smtp.gmail.com'
    MAIL_PORT=465
    MAIL_USERNAME='1787983095@qq.com'
    MAIL_PASSWORD='xiangbudao123.'
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True#启用传输层TSL协议

    '''JWT配置'''
    SECRET_KEY='some secret words'
    JWT_EXPIRY_SECOND=3600*2
    ALGORITHM='HS256'

'''开发环境'''
class DevelopmentConfig(baseConfig):
    DEBUG = True
    # SERVER_NAME='GISGAME.TEST'
    DATABASE={
        'host':'127.0.0.1',
        'port':'5432',
        'database':'test',
        'username':'postgres',
        'password':'alwaysbxd2711.'
    }
'''测试环境'''
class TestingConfig(baseConfig):
    DEBUG = True
'''生产环境'''
class productionConfig(baseConfig):
    DEBUG = False


config={

}


 