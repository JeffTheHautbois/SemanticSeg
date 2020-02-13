import cv2
import numpy as np
image=cv2.imread('fg_mask.jpg')
image = cv2.bitwise_not(image)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('original image',image)
cv2.waitKey(0)
ret, thresh=cv2.threshold(gray,176,255,0)
contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
n=len(contours)-1
# contours=sorted(contours,key=cv2.contourArea,reverse=False)[:n]
contours=sorted(contours,key=cv2.contourArea,reverse=True)[:10]
screenCnt = None
for c in contours:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.015 * peri, True)
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

cv2.drawContours(image,[screenCnt],0,(0,255,0),3)
# for c in contours:
#     hull=cv2.convexHull(c)
#     cv2.drawContours(image,[hull],0,(0,255,0),3)
#     retval=cv2.minAreaRect(
#     cv2.imshow('convex hull',image)

cv2.imshow('convex hull',image)
cv2.waitKey(0)
cv2.destroyAllWindows()