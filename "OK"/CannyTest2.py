from __future__ import print_function

import cv2 as cv
import numpy as np
from tqdm import tqdm
import sys

def sketch_transform(image, thrs1, thrs2):
    image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_canny = cv.Canny(image_grayscale, thrs1, thrs2, apertureSize = 7)
    _, mask = cv.threshold(image_canny, 100, 255, cv.THRESH_BINARY_INV)
    return mask

def main():

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cap = cv.VideoCapture("/home/max/Videos/testCam/WoodTest1.avi")
    cv.namedWindow('edge',cv.WINDOW_KEEPRATIO)
    cv.createTrackbar('thrs1', 'edge', 8000, 20000, nothing)
    cv.createTrackbar('thrs2', 'edge', 10000, 20000, nothing)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, unit='frame', desc='Processing')

    while True:
        _flag, img = cap.read()
        thrs1 = cv.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv.getTrackbarPos('thrs2', 'edge')

        
        edge = sketch_transform(img, thrs1, thrs2)
        # vis = img.copy()
        # vis[edge != 0] = (0, 255, 0)
        cv.imshow('edge', edge)
        progress_bar.update(1)
        ch = cv.waitKey(5) & 0xff
        if ch == 27:
            break

    print('Done')
    progress_bar.close()
    cap.release()

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
    