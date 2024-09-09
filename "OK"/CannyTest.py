from __future__ import print_function

import cv2 as cv
import numpy as np
from tqdm import tqdm

# built-in module
import sys


def main():
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cap = cv.VideoCapture("/home/max/Videos/testCam/CameraTest1.avi")
    cv.namedWindow('edge',cv.WINDOW_FREERATIO)
    cv.createTrackbar('thrs1', 'edge', 1000, 5000, nothing)
    cv.createTrackbar('thrs2', 'edge', 2000, 5000, nothing)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, unit='frame', desc='Processing')

    while True:
        _flag, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thrs1 = cv.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv.getTrackbarPos('thrs2', 'edge')
        edge = cv.Canny(gray, thrs1, thrs2, apertureSize=5)
        vis = img.copy()
        vis[edge != 0] = (0, 255, 0)
        cv.imshow('edge', vis)
        progress_bar.update(1)
        ch = cv.waitKey(5)
        if ch == 27:
            break

    print('Done')
    progress_bar.close()

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
    