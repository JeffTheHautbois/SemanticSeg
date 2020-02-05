import numpy as np
import cv2 as cv
import time
import imutils
import matplotlib.pyplot as plt


def mse(image_initial, image_final):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_initial.astype("float") -
                  image_final.astype("float")) ** 2)
    err /= float(image_initial.shape[0] * image_initial.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


capture = cv.VideoCapture(1)
back_sub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
# Number of frames to capture
num_frames = 125

# Start time
start = time.time()

# iterate through num_frames
for i in range(0, num_frames):
    ret, frame = capture.read()
    if frame is None:
        print("Error no frame")
        break
    fg_mask = back_sub.apply(frame)
    og_frame = frame
# End time
end = time.time()
# Time elapsed
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
# Calculate frames per second
fps = int(num_frames / seconds)
def_th = 128
while (True):
    ret, frame = capture.read()
    cv.imshow('cur_frame', frame)
    k = cv.waitKey(30)
    if k == 0x1b:
        # wait for auto focus
        cv.destroyAllWindows()
        for i in range(0, int(fps*3)):
            ret, frame = capture.read()
            cv.imshow('adjust', frame)
            new_frame = frame
        diff = cv.absdiff(og_frame, new_frame)
        mask = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

        th = def_th
        imask = mask > th
        markers = mask.astype(np.uint8)
        thresh = cv.threshold(
            markers, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        canvas = np.zeros_like(new_frame, np.uint8)
        canvas[imask] = new_frame[imask]
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(og_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.rectangle(new_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow("Original", og_frame)
        cv.imshow("result.png", canvas)
        og_frame = new_frame
        k = cv.waitKey(0)
        if k == 0x1b:
            def_th = int(input("change def_th"))
            print("new def_th", def_th)
        cv.destroyAllWindows()
