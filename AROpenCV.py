import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('TargetImage.jpg')
myVid = cv2.VideoCapture('Chicago_360p.mp4')

hT , wT , cT = imgTarget.shape
success , imgVideo = myVid.read()
imgVideo = cv2.resize(imgVideo , (wT , hT)) 


orb = cv2.ORB_create(nfeatures = 1000)
kp1 , des1 = orb.detectAndCompute(imgTarget , None)
# imgTarget = cv2.drawKeypoints(imgTarget , kp1 , None)



while True:
    success , imgWebcam = cap.read()
    imgWebcam = cv2.flip(imgWebcam , 1)
    
    kp2 , des2 = orb.detectAndCompute(imgWebcam , None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam , kp2 , None)
    
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1 , des2 , k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget , kp1 , imgWebcam , kp2 , good , None , flags = 2)
    

    cv2.imshow('imgFeatures' , imgFeatures)
    cv2.imshow('imgWebcam' , imgWebcam)
    cv2.imshow('Target Image' , imgTarget)
    cv2.imshow('Image Video' , imgVideo)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()