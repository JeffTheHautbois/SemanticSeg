import numpy as np
import cv2 as cv

capture = cv.VideoCapture(0)
# capture.set(3, 1280)  # set the resolution
#capture.set(4, 720)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
update_bg_model = True

while (1):
    ret, frame = capture.read()
    fg_mask = fgbg.apply(frame, learningRate=-1 if update_bg_model else 0)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
    cv.imshow("frame", fg_mask)
    k = cv.waitKey(30)
    if (k == 0x20):
        update_bg_model = not update_bg_model
        print("updating", update_bg_model)
    elif (k == 0x1b):
        break
