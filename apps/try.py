from datetime import datetime
from datetime import timedelta
import cv2



if __name__=='__main__':
    # print(datetime.datetime.now())
    # im=cv2.imread('../static/tempImage/sourceImage/test/test.png',cv2.COLOR_BGR2RGB)
    # print(type(im))
    # cv2.imshow('demo', im)
    # cv2.waitKey(5000)
    now=datetime.utcnow()
    afterNow=datetime.utcnow()+timedelta(seconds=3)
    print(now<afterNow)