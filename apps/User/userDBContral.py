import psycopg2
import traceback
'''操作administrator和domestic_consumer两张表'''
class userManage:
    def __init__(self, host, port, database, username, password) -> None:
        self.conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
        self.conn.autocommit=True #自动提交事务
        self.db = self.conn.cursor()

    def findManagerByAccount(self, admin_account: str):
        try:
            sql = """SELECT * FROM administrator where admin_account = %s;"""
            params = (admin_account,)
            self.db.execute(sql, params)
            user = self.db.fetchone()
            print(user)
            if user == None:
                print("数据库中无该用户！")
                return None
            print("查找成功，该用户存在数据库中！")
            return user
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False

    def insertManager(self,admin_name, admin_account, admin_password, admin_email, precinct):
        try:
            sql = """INSERT INTO administrator (admin_name,admin_account,admin_password,admin_email,precinct) VALUES 
            (%s,%s, %s,%s,%s);"""
            params = (admin_name, admin_account, admin_password, admin_email, precinct)
            self.db.execute(sql, params)
            print("插入成功，该用户注册成功！")
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False

    def deleteManager(self, admin_account):
        try:
            sql = """delete from administrator where admin_account = %s  """
            params = (admin_account,)
            # 执行语句
            self.db.execute(sql, params)
            print("删除成功，该用户注已注销！")
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False

    def updateManager(self,admin_account,admin_name, admin_password, admin_email):
        try:
            sql = """update administrator set admin_name = %s,admin_password=%s,admin_email=%s where admin_account = %s  """
            params = (admin_name,admin_password,admin_email,admin_account)
            self.db.execute(sql, params)
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False

    def findOrdinaryUserByAccount(self,dome_con_account):
        try:
            sql = """SELECT * FROM domestic_consumer where dome_con_account = %s;"""
            params = (dome_con_account,)
            self.db.execute(sql, params)
            user = self.db.fetchone()
            if user == None:
                print("数据库中无该用户！")
                return None
            print("查找成功，该用户存在数据库中！")
            return user
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False

    def insertOrdinaryUser(self,dome_con_name, dome_con_account, dome_con_password, dome_con_email):
        try:
            sql = """INSERT INTO domestic_consumer(dome_con_name, dome_con_account, dome_con_password, dome_con_email) VALUES 
                   (%s,%s, %s,%s);"""
            params = (dome_con_name, dome_con_account, dome_con_password, dome_con_email)
            self.db.execute(sql, params)
            print("插入成功，该用户注册成功！")
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False


    def deleteOrdinaryUser(self,dome_con_account):
        try:
            sql = """delete from domestic_consumer where dome_con_account = %s  """
            params = (dome_con_account,)
            # 执行语句
            self.db.execute(sql, params)
            print("删除成功，该用户注已注销！")
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False


    def updateOrdinaryUser(self, dome_con_name, dome_con_account, dome_con_password, dome_con_email):
        try:
            sql = """update administrator set dome_con_name = %s,dome_con_password=%s,dome_con_email=%s where dome_con_account = %s  """
            params = (dome_con_name , dome_con_password, dome_con_email, dome_con_account)
            self.db.execute(sql, params)
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False

