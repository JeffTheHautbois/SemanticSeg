import numpy as np
import cv2
import time
import imutils

MIN_MATCH_COUNT = 15

def img_erosion(image):
    kernel = np.ones((12,12),np.uint8)
    eroded = cv2.erode(image,kernel,iterations = 3)
    return eroded

def convert_1_to_3_channel(mask,image):
    mask_3_ch = np.zeros_like(image)
    mask_3_ch[:,:,0] = mask
    mask_3_ch[:,:,1] = mask
    mask_3_ch[:,:,2] = mask
    return mask_3_ch

def bitwise_segment(eroded_mask,image):
    eroded_3_ch = convert_1_to_3_channel(eroded_mask,image)
    segmented = cv2.bitwise_and(image, eroded_3_ch)
    return segmented

def canny_edge_find_contours(image):
    #canny edge detection
    edged = cv2.Canny(image, 30, 200)

    #find contours using simple approx
    contours, hierarchy=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #sort contours largest to smallest.
    contours=sorted(contours,key=cv2.contourArea,reverse=True)

    return contours[0]

def gb_cut_via_contour(cnt, image):
    x,y,w,h = cv2.boundingRect(cnt)
    rect = (x,y,w,h)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # running grabCut 10 times for good luck :)
    cv2.grabCut(image, mask, rect, bgdModel,
                fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    newimg = cv2.bitwise_and(image, image, mask=mask2)

    cv2.imshow("newimg", newimg)
    cv2.waitKey(0)

def extract_item(fg_mask, fg_frame):
    eroded = img_erosion(fg_mask)
    segmented = bitwise_segment(eroded,fg_frame)
    cnt = canny_edge_find_contours(segmented)
    gb_cut_via_contour(cnt,fg_frame)

def remove_shadow(fg_mask):
    ret,thresh = cv2.threshold(fg_mask,245,255,cv2.THRESH_BINARY)
    return thresh

def get_feature_list(detector):
    imgList = []
    descList = []
    kpList = []
    labelList = []

    trainImg = cv2.imread("soymilk.png", 0)
    trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
    imgList.append(trainImg)
    descList.append(trainDesc)
    kpList.append(trainKP)
    labelList.append("Organic Soy Beverage")

    trainImg = cv2.imread("apple.jpg",0)
    trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
    imgList.append(trainImg)
    descList.append(trainDesc)
    kpList.append(trainKP)
    labelList.append("Apple")

    # trainImg = cv2.imread("banana.jpg",0)
    # trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
    # imgList.append(trainImg)
    # descList.append(trainDesc)
    # kpList.append(trainKP)
    # labelList.append("Banana")
    return imgList, descList, kpList, labelList

def feature_match(capture,detector, imgList, descList, kpList, labelList):
    # FLANN_INDEX_KDITREE = 0
    # flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=12)
    # flann = cv2.FlannBasedMatcher(flannParam, {})
    while True:
        ret, QueryImgBGR = capture.read()
        QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
        #QueryImg = QueryImg.astype('uint8')
        queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)

        for i in range(0, len(imgList)):
            trainImg = imgList[i]
            trainDesc = descList[i]
            trainKP = kpList[i]
            # matches = flann.knnMatch(queryDesc, trainDesc, k=2)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(queryDesc, trainDesc)
            matches = sorted(matches, key = lambda x:x.distance)
            goodMatch = []
            ratio_thresh = 0.7
            for i, m in enumerate(matches):
                if i < len(matches) - 1 and m.distance < 0.7 * matches[i+1].distance:
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
        if cv2.waitKey(30) == 0x1b:
            break

def bitwise_extract(fg_mask, fg_frame):
    fg_mask = remove_shadow(fg_mask)
    kernel = np.ones((9,9),np.uint8)
    opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    # open_3_ch = convert_1_to_3_channel(opening, fg_frame)
    extractedImg = bitwise_segment(opening,fg_frame)
    name = input("enter item name")
    cv2.imwrite(name+".jpg",extractedImg)
    cv2.imshow("extracted", extractedImg)
    cv2.waitKey(0)

def main():


    detector = cv2.ORB_create()

    imgList, descList, kpList, labelList = get_feature_list(detector)

    capture = cv2.VideoCapture(1)
    capture.set(3, 1280)  # set the resolution
    capture.set(4, 720)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capture.set(cv2.CAP_PROP_EXPOSURE, -6)
    #capture.set(cv.CAP_PROP_CONTRAST, 0)
    back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    back_sub.setVarThreshold(100)
    update_bg_model = True
    bg_mask = 0
    while True:
        ret, frame = capture.read()
        fg_mask = back_sub.apply(
            frame, learningRate=-1 if update_bg_model else 0)
        cv2.imshow("fg_mask", fg_mask)
        k = cv2.waitKey(30)
        if (k == 0x20):
            update_bg_model = not update_bg_model
            print("updating", update_bg_model)
        elif (k == 0x73): #"key = 's'"

            print("segment")
            # extract_item(fg_mask,frame)
            bitwise_extract(fg_mask,frame)
        elif(k == 0x42): #"key = 'B'"
            feature_match(capture,detector,imgList, descList, kpList, labelList)

        elif (k == 0x1b): #"key = 'esc'"
            break

if __name__ == "__main__":
    main()