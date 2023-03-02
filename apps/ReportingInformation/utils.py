import json


def getGeoJsonGeometry(GeoJson:json)->str:
    '''
    :param GeoJson:
    :return: geometry or geometries
    '''
    if GeoJson['type']=='FeatureCollection':
        # return str([feature['geometry'] for feature in GeoJson['features']])
        return str(GeoJson['features'][0]['geometry'])
    elif GeoJson['type']=='Feature':
        return str(GeoJson['geometry'])
    elif  GeoJson['type'] in ["Point","MultiPoint","LineString","MultiLineString","Polygon","MultiPolygon"]:
        return str(GeoJson)
    elif GeoJson['type']=="GeometryCollection":
        return str(GeoJson['geometries'])
    else:
        return False






if __name__=="__main__":
    with open('../../static/GeoJson/province-city-county/周至县.json', 'r', encoding='utf-8') as f:
        data=json.load(f)
        print(data)
        features=data['features']
        print(len(features))
        feature=features[0]
        print(feature)
        geometry=feature['geometry']
        print(geometry)
        coordinates=geometry['coordinates']
        print(coordinates)



