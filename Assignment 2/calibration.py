# calibrate camera
import cv2
import numpy as np

class Calibration(object):
    def __init__(self, checkerboard_size = (9,6), criterion = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        self.checkerboard_size = checkerboard_size
        self.criterion = criterion
        self.three_d_points = []
        self.two_d_points = []
        self.objectp3d = np.zeros((1,checkerboard_size[0]*checkerboard_size[1],3),np.float32)
        self.objectp3d[0,:,:2] = np.mgrid[0:checkerboard_size[0],0:checkerboard_size[1]].T.reshape(-1,2)
        self.prev_img_shape = None
        self.images = []

    def calibrate_camera(self):

        # take 10 pictures and add to the images array
        capture = cv2.VideoCapture(0)
        how_many_good_images = 0

        while(how_many_good_images < 100):
            ret, frame = capture.read()
            if ret == False:
                break
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            if ret == True:
                self.images.append(frame)
                how_many_good_images += 1
        capture.release()


        # for each image, find the corners
        for image in self.images:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            print("ret")
            print(ret)
            if ret == True:
                self.three_d_points.append(self.objectp3d)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criterion)
                self.two_d_points.append(corners2)
                cv2.drawChessboardCorners(image, self.checkerboard_size, corners2, ret)
                cv2.imshow('img',image)
                cv2.waitKey(500)


        cv2.destroyAllWindows()
        height, width = gray.shape[:2]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.three_d_points, self.two_d_points, gray.shape[::-1],None,None)

        np.savez('calibration_params.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
