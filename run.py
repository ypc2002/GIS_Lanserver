from apps import app
from gevent import pywsgi



if __name__ == '__main__':
   # server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
   # server.serve_forever()
   print(app.url_map)
   app.run(host='0.0.0.0',port=5000,threaded=True)#开启多线程，默认为10