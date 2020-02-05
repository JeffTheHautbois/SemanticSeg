import cv2
import numpy as np
MIN_MATCH_COUNT = 8
# Initiate SIFT detector
detector = cv2.ORB_create()


imgList = []
descList = []
kpList = []
trainImg = cv2.imread("gum_f.png", 0)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
imgList.append(trainImg)
descList.append(trainDesc)
kpList.append(trainKP)
trainImg2 = cv2.imread("bar_f.png", 0)
trainKP2, trainDesc2 = detector.detectAndCompute(trainImg2, None)
imgList.append(trainImg2)
descList.append(trainDesc2)
kpList.append(trainKP2)

cam = cv2.VideoCapture(1)
while True:

    ret, QueryImgBGR = cam.read()
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    #QueryImg = QueryImg.astype('uint8')
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(0, len(imgList)):
        trainImg = imgList[i]
        trainDesc = descList[i]
        trainKP = kpList[i]
        # Match descriptors.
        #matches = bf.match(queryDesc, trainDesc)
        matches = flann.knnMatch(queryDesc, trainDesc, k=2)

        # matches = sorted(matches, key=lambda x: x.distance)

        # goodMatch = matches[:10]

        goodMatch = []
        for m, n in matches:
            if(m.distance < 0.75*n.distance):
                goodMatch.append(m)
        if(len(goodMatch) > MIN_MATCH_COUNT):
            tp = []
            qp = []
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp, qp))
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 5.0)
            h, w = trainImg.shape[:2]
            trainBorder = np.float32(
                [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            queryBorder = cv2.perspectiveTransform(trainBorder, H)
            cv2.polylines(QueryImgBGR, [np.int32(
                queryBorder)], True, (0, 255, 0), 2, cv2.LINE_AA)
        '''
        if (isOne):
            isOne = False
        else:
            isOne = True
        '''
    # else:
    #     print("Not Enough match found- %d/%d", len(goodMatch), MIN_MATCH_COUNT)
    cv2.imshow('result', QueryImgBGR)
    if cv2.waitKey(0) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
