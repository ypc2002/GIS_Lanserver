import psycopg2
import json
from .utils import getGeoJsonGeometry
import traceback

'''操作report_information和dome_con_inform 这两张表'''
class reportInformationManage():
    def __init__(self,host,port,database,username,password) -> None:
        self.conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
        self.conn.autocommit =True  # 自动提交事务
        self.db=self.conn.cursor()

    def insertInformation(self,dome_con_id,inform_location,inform_time,inform_reason,coordination:list,inform_description,inform_picture_path):
        try:
            if None in [dome_con_id,inform_location,inform_time,inform_reason,coordination,inform_description,inform_picture_path]:
                return '信息不完整'
            sql1 = """INSERT INTO report_information (inform_location,inform_time,inform_reason,coordination,inform_description,inform_picture_path) VALUES 
                   (%s,%s,%s,%s,%s,%s) RETURNING inform_id;"""
            wkt_point='SRID=3857;POINT({} {})'.format(coordination[0],coordination[1])
            self.conn.autocommit=False
            params1 = (inform_location,inform_time,inform_reason,wkt_point,inform_description,inform_picture_path)
            self.db.execute(sql1, params1)
            inform_id=self.db.fetchone()
            print("返回inform_id:",inform_id)
            sql2="""INSERT INTO dome_con_inform(dome_con_id,inform_id) VALUES 
                   (%s,%s);"""
            params2=(dome_con_id,inform_id)
            self.db.execute(sql2, params2)
            self.conn.commit()
            self.conn.autocommit = True
            return True
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False

    def reportInformation2dict(self,all_reportInformation):
        try:
            if all_reportInformation == None:
                print("无该用户段的举报信息！")
                return None
            print("查找到相关用户的举报信息")
            reportInformation_dic_list=[]
            keys=["举报信息编号", "地点","举报时间", "举报原因","经纬度", "描述","照片路径"]
            for reportInformation in all_reportInformation:
                reportInformation_dict={}
                for key,value in zip(keys,reportInformation):
                    reportInformation_dict[key]=value
                reportInformation_dic_list.append(reportInformation_dict)
            return reportInformation_dic_list
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False
    def findInformationByID(self,ID):
        try:
            sql = """select inform_id,inform_location,inform_time,inform_reason,ST_AsGeoJson(coordination),
                    inform_description,inform_picture_path from report_information 
                   where inform_id=%s;"""
            params = (ID,)
            self.db.execute(sql, params)
            all_reportInformation = self.db.fetchall()
            return self.reportInformation2dict(all_reportInformation)
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False
    def findInformationByUserID(self, userID):
        try:
            sql = """select report_information.inform_id,report_information.inform_location,report_information.inform_time,
                  report_information.inform_reason,ST_AsGeoJson(report_information.coordination),
                  report_information.inform_description,report_information.inform_picture_path
              from report_information,dome_con_inform 
              where dome_con_inform.dome_con_id=%s and report_information.inform_id=dome_con_inform.inform_id;"""
            params = (userID,)
            self.db.execute(sql, params)
            all_reportInformation = self.db.fetchall()
            return self.reportInformation2dict(all_reportInformation)
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False

    def findInformationByLocation(self,location:str):
        try:
            sql="""select inform_id,inform_location,inform_time,inform_reason,ST_AsGeoJson(coordination),
                    inform_description,inform_picture_path
                    from report_information where inform_location like %s"""
            location='%{}%'.format(location)
            params=(location,)
            self.db.execute(sql, params)
            all_reportInformation = self.db.fetchall()
            return self.reportInformation2dict(all_reportInformation)
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False

    def findInformationByRange(self,range:json):
        try:
            sql="""SELECT inform_id,inform_location,inform_time,inform_reason,ST_AsGeoJson(coordination),
                    inform_description,inform_picture_path FROM report_information where 
                        ST_Contains(ST_GeomFromGeoJSON(%s),coordination);"""
            Range=getGeoJsonGeometry(range)
            params=(Range,)
            self.db.execute(sql, params)
            all_reportInformation = self.db.fetchall()
            return self.reportInformation2dict(all_reportInformation)
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False

    def findInformationByTime(self, startTime: str, endTime: str):
        try:
            sql = """select inform_id,inform_location,inform_time,inform_reason,ST_AsGeoJson(coordination),
                    inform_description,inform_picture_path from report_information where inform_time
                    between {} and {};""".format('\''+startTime+'\'','\''+endTime+'\'')
            print(sql)
            self.db.execute(sql)
            all_reportInformation = self.db.fetchall()
            return self.reportInformation2dict(all_reportInformation)
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False


    def findInformationByReason(self,Description):
        pass

    def findInformationByDescription(self, Description):
        pass


    def deleteInformationByUserID(self,userID):
        return True

    # def deleteInformationByLocation(self,location):
    #     sql = """delete from  student where inform_location like %s;"""
    #     location = '%{}%'.format(location)
    #     params = (location,)
    #     self.db.execute(sql, params)
    #     return True

    #仅通过ID删除（客户端一定存在该信息的ID才能删除）
    def deleteOneByID(self, ID):
        try:
            sql = """delete from report_information where inform_id = %s returning inform_id """
            params = (ID,)
            self.db.execute(sql, params)
            id=self.db.fetchone()
            print('id',id)
            return id
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False

    #通过ID更新
    def updata(self,inform_id,inform_location,inform_time,inform_reason,coordination:list,inform_description,inform_picture_path):
        try:
            sql="""UPDATE report_information
                    SET inform_location=%s,inform_time=%s,inform_reason=%s,coordination=%s,inform_description=%s,inform_picture_path=%s
                    WHERE inform_id=%s;"""
            wkt_point='SRID=3857;POINT({} {})'.format(coordination[0], coordination[1])
            params = (inform_location,inform_time,inform_reason,wkt_point,inform_description,inform_picture_path,inform_id,)
            self.db.execute(sql,params)
            return True
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False



'''操作review_record表'''
class reviewRecordManage():
    def __init__(self, host, port, database, username, password) -> None:
        self.conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
        self.conn.autocommit = True  # 自动提交事务
        self.db = self.conn.cursor()

    def insertRecord(self,admin_id,inform_id,examine_time,inform_fact):
        try:
            sql="""insert into review_record (admin_id,inform_id,examine_time,inform_fact)
             values(%s,%s,%s,%s) returning inform_id"""
            params=(admin_id,inform_id,examine_time,inform_fact)
            self.db.execute(sql,params)
            re_inform_id=self.db.fetchone()
            if re_inform_id!=None:
                return True
        except Exception as e:
            print(traceback.print_exc())
            print(e.args)
            return False


    def records2dict(self, records):
        if records == None:
            print("无该用户段的举报信息！")
            return None
        print("查找到相关用户的举报信息")
        records_dic_list = []
        keys = ["检测管理员编号","举报信息编号","是否属实","审核时间", "地点", "举报时间", "举报原因", "经纬度", "描述", "照片路径"]
        for record in records:
            record_dict = {}
            for key, value in zip(keys, record ):
                record_dict[key] = value
            records_dic_list.append(record_dict)
        return records_dic_list

    def findRecordByAdminID(self, adminID):
        sql="""select 
        review_record.admin_id,review_record.inform_id,review_record.inform_fact,review_record.examine_time,
        report_information.inform_location,report_information.inform_time,report_information.inform_reason,
        ST_AsGeoJson(report_information.coordination),report_information.inform_description,report_information.inform_picture_path
        from report_information,review_record 
        where 
        review_record.admin_id={} and report_information.inform_id=review_record.inform_id""".format(adminID)
        self.db.execute(sql)
        records=self.db.fetchall()
        return self.records2dict(records=records)
    def findRecordByInformID(self, informID):
        sql="""select 
        review_record.admin_id,review_record.inform_id,review_record.inform_fact,review_record.examine_time,
        report_information.inform_location,report_information.inform_time,report_information.inform_reason,
        ST_AsGeoJson(report_information.coordination),report_information.inform_description,report_information.inform_picture_path
        from report_information,review_record 
        where 
        review_record.inform_id={} and report_information.inform_id=review_record.inform_id""".format(informID)
        self.db.execute(sql)
        records=self.db.fetchall()
        return self.records2dict(records=records)

    def findRecordByTime(self,startTime,endTime):
        sql="""select 
        review_record.admin_id,review_record.inform_id,review_record.inform_fact,review_record.examine_time,
        report_information.inform_location,report_information.inform_time,report_information.inform_reason,
        ST_AsGeoJson(report_information.coordination),report_information.inform_description,report_information.inform_picture_path
        from review_record,report_information where report_information.inform_id=review_record.inform_id and examine_time 
        between {} and {}""".format('\''+startTime+'\'','\''+endTime+'\'')
        self.db.execute(sql)
        records=self.db.fetchall()
        return self.records2dict(records=records)

    def findRecordByLocation(self,location):
        sql="""select 
        review_record.admin_id,review_record.inform_id,review_record.inform_fact,review_record.examine_time,
        report_information.inform_location,report_information.inform_time,report_information.inform_reason,
        ST_AsGeoJson(report_information.coordination),report_information.inform_description,report_information.inform_picture_path 
        from report_information,review_record 
        where 
        report_information.inform_location like {} and report_information.inform_id=review_record.inform_id""".format('\'%'+location+'%\'')
        self.db.execute(sql)
        records=self.db.fetchall()
        return self.records2dict(records=records)

    def findRecordByRange(self,range):
        sql="""select 
        review_record.admin_id,review_record.inform_id,review_record.inform_fact,review_record.examine_time,
        report_information.inform_location,report_information.inform_time,report_information.inform_reason,
        ST_AsGeoJson(report_information.coordination),report_information.inform_description,report_information.inform_picture_path
        from report_information,review_record 
        where 
        ST_Contains(ST_GeomFromGeoJSON(%s),report_information.coordination) and report_information.inform_id=review_record.inform_id"""
        Range=getGeoJsonGeometry(range)
        params=(Range,)
        self.db.execute(sql,params)
        records = self.db.fetchall()
        return self.records2dict(records=records)
    #仅通过ID删除
    def delete(self,admin_id,inform_id):
        sql="""delete from review_record where admin_id={} and inform_id={} returning admin_id""".format(admin_id,inform_id)
        self.db.execute(sql)
        del_admin_id=self.db.fetchone()
        if del_admin_id!=None:
            return True
        else:
            return False
    #仅通过ID更新
    def updata(self,admin_id,inform_id,examine_time,inform_fact):
        sql = """update review_record
        set examine_time={},inform_fact={}
        where admin_id={} and inform_id={} returning inform_id""".format('\''+examine_time+'\'', inform_fact,admin_id, inform_id )
        self.db.execute(sql)
        re_inform_id = self.db.fetchone()
        if re_inform_id != None:
            return True
        else:
            return False


