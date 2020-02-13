import numpy as np
import cv2 as cv
import time
import imutils
from skimage.metrics import structural_similarity


def initialize_bg_fg(capture, back_sub):
    width = capture.get(3)
    height = capture.get(4)
    # fps = capture.get(7) for video file only not for video stream from webcam

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

    return width, height, fps, fg_mask, og_frame


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


def extract_diff(image_initial, image_final):

    # convert to grey_scale
    gray_initial = cv.cvtColor(image_initial, cv.COLOR_BGR2GRAY)
    gray_final = cv.cvtColor(image_final, cv.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(gray_initial, gray_final, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv.threshold(
        diff/2, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    cnts = cv.findContours(
        thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #
    #
    # for c in cnts:
    #     # compute the bounding box of the contour and then draw the
    #     # bounding box on both input images to represent where the two
    #     # images differ
    #     (x, y, w, h) = cv.boundingRect(c)
    #     cv.rectangle(image_initial, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv.rectangle(image_final, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    cv.imshow("Original", image_initial)
    cv.imshow("Modified", image_final)
    cv.imshow("Diff", diff)
    cv.imshow("Thresh", thresh)
    cv.waitKey(0)


def main():
    # capture from built-in webcam - 0 or external webcam - 1
    capture = cv.VideoCapture(1)

    # initialize backgroundSubtractor
    back_sub = cv.createBackgroundSubtractorMOG2(detectShadows=False)

    width, height, fps, og_bg_mask, og_frame = initialize_bg_fg(
        capture, back_sub)
    cv.imshow('og_mask', og_bg_mask)
    cv.imshow('og_frame', og_frame)
    print("w", width, "h", height, "fps", fps)
    k = cv.waitKey(0)
    if k == 0x1b:
        cv.destroyAllWindows()
        while (True):
            # get current mask
            # for i in range(0, int(fps/2)):
            ret, frame = capture.read()
            cur_mask = back_sub.apply(frame)
            mask_diff = mse(og_bg_mask, cur_mask)
            print(mask_diff)
            #cv.imshow('cur_mask', cur_mask)
            if (mask_diff > 500):
                # wait for auto focus
                print("wait for camera to adjust")
                cv.destroyAllWindows()
                for i in range(0, int(fps*3)):
                    ret, frame = capture.read()
                    back_sub.apply(frame)
                    #cv.imshow('cur_frame', frame)
                    new_frame = frame
                cur_mask = back_sub.apply(frame)
                mask_diff = mse(og_bg_mask, cur_mask)
                # print("mask_diff", mask_diff)
                # cv.imshow('cur_mask', cur_mask)
                extract_diff(og_frame, new_frame)
                # cv.waitKey(0)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break
            #mask_diff = mse(og_bg_mask, cur_mask)
            # if mask_diff greater than 2000 starts updating background sub.

            # if mask_diff becomes less thant 200 set new og_background.

            # while (mask_diff > 200):
            #     ret, frame = capture.read()
            #     #mask_diff = mse(og_bg_mask, cur_mask)
            #     cv.imshow("cur_mask", cur_mask)
            #     cv.waitKey(0)
            #     counter += 1
            #     print("c", counter)
            #     if counter > 15:
            #         cur_mask = back_sub.apply(frame)
            #         break
            # mask_diff = mse(og_bg_mask, cur_mask)
            # while (mask_diff > 200 and counter < 30):
            #     ret, frame = capture.read()
            #     counter += 1
            #     back_sub.apply(frame)
            # if(mask_diff < 200 and counter > 30):
            #     print("subtract image", counter)
            #     counter = 0
            #     ret, new_frame = capture.read()
            #     #extract_diff(og_frame, new_frame)
            #     cv.imshow("new_frame", new_frame)
            #     cv.imshow("og_frame", og_frame)
            #     cv.imshow("cur_mask", cur_mask)
            #     # gray_initial = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
            #     # cv.imshow("gray_frame", gray_initial)
            #     cv.waitKey(0)
            #     break
            # print(mask_diff)

    # if k == 0x1b:
    #     print('capturing!')
    #     cv.imwrite("test.jpg", fgMask)


def refinedSegment(c_frame, bg_mask, o_frame):
    niter = 3


def main_2():
    capture = cv.VideoCapture(1)
    capture.set(3, 1280)  # set the resolution
    capture.set(4, 720)
    capture.set(cv.CAP_PROP_AUTOFOCUS, 0)
    capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
    capture.set(cv.CAP_PROP_EXPOSURE, 3)
    #capture.set(cv.CAP_PROP_CONTRAST, 0)
    back_sub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    back_sub.setVarThreshold(100)
    update_bg_model = True
    bg_mask = 0
    while True:
        ret, frame = capture.read()
        fg_mask = back_sub.apply(
            frame, learningRate=-1 if update_bg_model else 0)
        cv.imshow("fg_mask", fg_mask)
        k = cv.waitKey(30)
        if (k == 0x20):
            update_bg_model = not update_bg_model
            print("updating", update_bg_model)
        elif (k == 0x73):
            cv.imwrite("D:\PyProj\SemanticSeg\Fg_mask.jpg", fg_mask) 
            cv.imwrite("D:\PyProj\SemanticSeg\Fg_frame.jpg", frame) 
            print("saved fg_frame & fg_mask")
        elif(k == 0x42):
            cv.imwrite("D:\PyProj\SemanticSeg\Bg_mask.jpg", fg_mask)
            cv.imwrite("D:\PyProj\SemanticSeg\Bg_frame.jpg", frame) 
            print("saved bg_frame")
        elif (k == 0x1b):
            break


if __name__ == "__main__":
    main_2()

# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break
#
#     fgMask = backSub.apply(frame)
#     if (initialfirstBg):
#         firstBg = fgMask
#         initialfirstBg = False
#
#
#     cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
#     cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Mask', fgMask)
#
#     k = cv.waitKey(10)
#     if k == 0x1b:
#         print('capturing!')
#         cv.imwrite("test.jpg", fgMask)

#
# mixtureGaussian
# binaryMask,
# connectedComponents algorithm
# BagofVisualWords
