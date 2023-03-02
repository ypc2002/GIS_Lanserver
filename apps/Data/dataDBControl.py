import psycopg2
from .utils import getGeoJsonGeometry
import traceback



'''操作illegal_region和admin_illegal_region表   illegal_region的region（矢量）使用矢量切面返回'''
class illegalRegionManage:
    def __init__(self,host,port,database,username,password) -> None:
        self.conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
        self.conn.autocommit = True
        self.db=self.conn.cursor()

    def insert(self, illegal_time, illegal_location, illegal_reason, vector_boundary, illegal_description, illegal_raster_path, illegal_coordinate, admin_id):
        try:
            sql1="""insert into illegal_region(illegal_time,illegal_location,illegal_reason,vector_boundary,area,illegal_description,illegal_raster_path,illegal_coordinate)
        values(%s,%s,%s,ST_GeomFromGeoJSON(%s,3857),ST_Area(ST_GeomFromGeoJSON(%s,3857)),%s,%s,ST_GeomFromGeoJSON(%s,3857)) returning illegal_id;"""
            vector_boundary_geom=getGeoJsonGeometry(vector_boundary)
            illegal_coordinate_geom=getGeoJsonGeometry(illegal_coordinate)
            # area = 'ST_Area(ST_GeomFromGeoJSON({}))'.format(vector_boundary_geom)
            # vector_boundary_geom='ST_GeomFromGeoJSON({})'.format(vector_boundary_geom)
            illegal_time='\'{}\''.format(illegal_time)
            params1=(illegal_time,illegal_location,illegal_reason,vector_boundary_geom,vector_boundary_geom,illegal_description,illegal_raster_path,illegal_coordinate_geom)
            self.db.execute(sql1,params1)
            illegal_id=self.db.fetchone()[0]
            print('illegal_id',illegal_id)
            sql2="""insert into admin_illegal_region(admin_id,illegal_id) values({},{})""".format(admin_id,illegal_id)
            self.db.execute(sql2)
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False

    def regionInform2dict(self,illegalRegions):
        if illegalRegions == None:
            print("无违法用地信息！")
            return None
        print("查找到相关违法用地信息！")
        illegalRegions_dic_list = []
        keys = ["违法用地编号", "时间", "地点", "原因","矢量边界", "具体描述", "面积","照片路径","坐标"]
        for illegalRegion in illegalRegions:
            record_dict = {}
            for key, value in zip(keys, illegalRegion):
                record_dict[key] = value
            illegalRegions_dic_list.append(record_dict)
        return illegalRegions_dic_list
    def findRegionByAdminID(self, adminID):
        try:
            sql="""select admin_illegal_region.illegal_id,illegal_time,illegal_location,illegal_reason,ST_AsGeoJson(vector_boundary),illegal_description,area,illegal_raster_path,ST_AsGeoJson(illegal_coordinate)
            from illegal_region,admin_illegal_region 
            where admin_illegal_region.admin_id={} and admin_illegal_region.illegal_id=illegal_region.illegal_id """.format(adminID)
            self.db.execute(sql)
            allRegion=self.db.fetchall()
            return self.regionInform2dict(allRegion)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果":"查询失败"}
    def findRegionByTime(self,startTime,endTime):
        try:
            sql="""select illegal_id,illegal_time,illegal_location,illegal_reason,ST_AsGeoJson(vector_boundary),illegal_description,area,illegal_raster_path,ST_AsGeoJson(illegal_coordinate)
                from illegal_region
                where illegal_region.illegal_time between {} and {}""".format('\''+startTime+'\'','\''+endTime+'\'')
            self.db.execute(sql)
            illegalRegions = self.db.fetchall()
            return self.regionInform2dict(illegalRegions=illegalRegions)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果": "查询失败"}
    def findRegionByLocation(self,location):
        try:
            sql = """select illegal_id,illegal_time,illegal_location,illegal_reason,ST_AsGeoJson(vector_boundary),illegal_description,area,illegal_raster_path,ST_AsGeoJson(illegal_coordinate)
                        from illegal_region
                        where illegal_region.illegal_location like {}""".format('\'%'+location+'%\'')
            self.db.execute(sql)
            illegalRegions = self.db.fetchall()
            return self.regionInform2dict(illegalRegions=illegalRegions)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果": "查询失败"}

    def findRegionByRange(self,range):
        try:
            sql = """select 
                illegal_id,illegal_time,illegal_location,illegal_reason,ST_AsGeoJson(vector_boundary),illegal_description,area,illegal_raster_path,ST_AsGeoJson(illegal_coordinate)
               from illegal_region
                where 
                ST_Contains(ST_GeomFromGeoJSON(%s,3857),vector_boundary) or ST_Contains(ST_GeomFromGeoJSON(%s,3857),illegal_coordinate)"""
            Range = getGeoJsonGeometry(range)
            params = (Range,Range)
            self.db.execute(sql, params)
            illegalRegions = self.db.fetchall()
            return self.regionInform2dict(illegalRegions=illegalRegions)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果": "查询失败"}

    '''未测试'''
    def findRegionByTimeAndLocation(self,startTime,endTime,location):
        try:
            sql = """select illegal_id,illegal_time,illegal_location,illegal_reason,ST_AsGeoJson(vector_boundary),illegal_description,area,illegal_raster_path,ST_AsGeoJson(illegal_coordinate)
                       from illegal_region
                       where illegal_region.illegal_time between {} and {} 
                       and illegal_region.illegal_location like {}""".format('\'' + startTime + '\'',
                                                                                     '\'' + endTime + '\'',
                                                                             '\'%'+location+'%\'')

            self.db.execute(sql)
            illegalRegions = self.db.fetchall()
            return self.regionInform2dict(illegalRegions=illegalRegions)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果": "查询失败"}

    '''未测试'''
    def findRegionByTimeAndRange(self, startTime, endTime, range):
        try:
            sql = """select illegal_id,illegal_time,illegal_location,illegal_reason,ST_AsGeoJson(vector_boundary),illegal_description,area,illegal_raster_path,ST_AsGeoJson(illegal_coordinate)
                       from illegal_region
                       where illegal_region.illegal_time between {} 
                       and {ST_Contains(ST_GeomFromGeoJSON(%s),vector_boundary) 
                       or 
                       ST_Contains(ST_GeomFromGeoJSON(%s),illegal_coordinate)""".format('\'' + startTime + '\'',
                                                                             '\'' + endTime + '\'')
            Range = getGeoJsonGeometry(range)
            params = (Range,Range)
            self.db.execute(sql,params)
            illegalRegions = self.db.fetchall()
            return self.regionInform2dict(illegalRegions=illegalRegions)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果": "查询失败"}



    def findRegionByRegionID(self, RegionID):
        try:
            sql = """select 
                   illegal_id,illegal_time,illegal_location,illegal_reason,ST_AsGeoJson(vector_boundary),illegal_description,area,illegal_raster_path,ST_AsGeoJson(illegal_coordinate)
                  from illegal_region
                   where illegal_id={}""".format(RegionID)
            self.db.execute(sql)
            illegalRegions = self.db.fetchall()
            return self.regionInform2dict(illegalRegions=illegalRegions)
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果": "查询失败"}

    '''可以增加查询方式'''


    def deleteByRegionID(self,illegalRegionID):
        try:
            sql="""delete from illegal_region where illegal_id={} returning illegal_id""".format(illegalRegionID)
            self.db.execute(sql)
            ID = self.db.fetchone()
            print('==========================================')
            return ID
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return {"结果": "删除失败",'原因':'无该记录！'}

    def updata(self,illegalRegionID,illegal_location,illegal_reason,illegal_description):
        try:
            sql="""update illegal_region set illegal_location=%s,illegal_reason=%s,illegal_description=%s
            where illegal_id=%s"""
            params = (illegal_location, illegal_reason,illegal_description,illegalRegionID)
            self.db.execute(sql,params)
            return True
        except Exception as e:
            print(e.args)
            print("************************************")
            print(traceback.format_exc())
            return False








'''操作land_use和admin_land_use表'''
class landUseManage:
    def __init__(self,host,port,database,username,password) -> None:
        self.conn = psycopg2.connect(host=host, port=port, database=database, user=username, password=password)
        self.conn.autocommit = True
        self.db=self.conn.cursor()

    def insert(self,land_use_time,land_use_location,land_use_description,land_use_path):
        '''插入land_use和admin_land_use表'''
        pass
    def findlandUseByTime(self):
        return
    def findByadmin(self):
        pass
    def findByTime(self):
        pass
    def deleteBylandUseByTime(self):
        pass

