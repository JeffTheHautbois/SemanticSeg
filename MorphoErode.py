import cv2
import numpy as np

image = cv2.imread('Fg_mask.jpg',0)

img = cv2.imread('Fg_frame.jpg')
kernel = np.ones((12,12),np.uint8)
erosion = cv2.erode(image,kernel,iterations = 1)
cv2.imshow('eroded',erosion)
# image = cv2.resize(image, img.shape[1::-1])
img2 = np.zeros_like(img)
img2[:,:,0] = erosion
img2[:,:,1] = erosion
img2[:,:,2] = erosion
segmented = cv2.bitwise_and(img, img2)
cv2.imshow('segmented',segmented)
edged = cv2.Canny(segmented, 30, 200)
cv2.imshow('cannyedge',edged)

contours, hierarchy=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

n=len(contours)-1
# contours=sorted(contours,key=cv2.contourArea,reverse=False)[:n]
contours=sorted(contours,key=cv2.contourArea,reverse=True)
screenCnt = None
for c in contours:
    cv2.drawContours(img,[c],0,(0,255,0),3)
    cv2.imshow('contour',img)
    
    cv2.waitKey(0)
x,y,w,h = cv2.boundingRect(contours[0])
rect = (x,y,w,h)

# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# # running grabCut 15 times for good luck :)
# cv2.grabCut(img, mask, rect, bgdModel,
#             fgdModel, 15, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
# #newimg = newimg*mask2[:,:,np.newaxis]
# newimg = cv2.bitwise_and(img, img, mask=mask2)

# # save the segmented image to local
# cv2.imshow("newimg", newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()