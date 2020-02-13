import cv2
import numpy as np
MIN_MATCH_COUNT = 15

detector = cv2.ORB_create()

FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=12)
flann = cv2.FlannBasedMatcher(flannParam, {})
#flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

imgList = []
descList = []
kpList = []
labelList = []
trainImg = cv2.imread("soymilk.png", 0)
#trainImg = cv2.resize(trainImg, (360, 480))
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
imgList.append(trainImg)
descList.append(trainDesc)
kpList.append(trainKP)
labelList.append("Soy Milk")
# trainImg2 = cv2.imread("bar_f.png", 0)
# #trainImg2 = cv2.resize(trainImg2, (360, 480))
# trainKP2, trainDesc2 = detector.detectAndCompute(trainImg2, None)
# imgList.append(trainImg2)
# descList.append(trainDesc2)
# kpList.append(trainKP2)
# labelList.append("bar")

isOne = False
# print(len(trainKP))
# print(len(trainKP2))

cam = cv2.VideoCapture(1)
while True:

    ret, QueryImgBGR = cam.read()
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    #QueryImg = QueryImg.astype('uint8')
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)

    for i in range(0, len(imgList)):
        trainImg = imgList[i]
        trainDesc = descList[i]
        trainKP = kpList[i]
        matches = flann.knnMatch(queryDesc, trainDesc, k=2)

        goodMatch = []
        ratio_thresh = 0.7
        for m, n in matches:
            if(m.distance < ratio_thresh*n.distance):
                goodMatch.append(m)
        if(len(goodMatch) > MIN_MATCH_COUNT):
            tp = []
            qp = []
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp, qp))
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
            h, w = trainImg.shape
            trainBorder = np.float32(
                [[[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]])
            if H is not None:
                queryBorder = cv2.perspectiveTransform(trainBorder, H)
                # print(queryBorder)
                cv2.putText(QueryImgBGR, labelList[i], (queryBorder[0][0][0], queryBorder[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, [
                            255, 255, 255], thickness=3, lineType=cv2.LINE_AA)
                cv2.polylines(QueryImgBGR, [np.int32(
                    queryBorder)], True, (255, 255, 0), 5)
        else:
            print("Not Enough match found- %d/%d",
                  len(goodMatch), MIN_MATCH_COUNT)
    cv2.imshow('result', QueryImgBGR)
    if cv2.waitKey(30) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
