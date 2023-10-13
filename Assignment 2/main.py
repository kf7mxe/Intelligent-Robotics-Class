import cv2
import numpy as np
from calibration import Calibration
from monovideoodometery_camera import MonoVideoOdometery


# look for calibration file
# if not found, calibrate camera
# if found, load calibration file 

# check for calibration_params.npz

try:
    config = np.load('calibration_params.npz')
    print('calibration file found')
    # read in the calibration file
    mtx = config['mtx']
    dist = config['dist']
    rvecs = config['rvecs']
    tvecs = config['tvecs']

except:
    print('calibration file not found')
    Calibration().calibrate_camera()
    config = np.load('calibration_params.npz')
    print('calibration file found')
    # read in the calibration file
    mtx = config['mtx']
    dist = config['dist']
    rvecs = config['rvecs']
    tvecs = config['tvecs']



lk_params = dict( winSize  = (21,21),
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))



capture = cv2.VideoCapture(0)

i = 1

visualOdemetry = MonoVideoOdometery(pose=None,cameraMatrix=mtx, distortionCoefficients=dist)
ret, frame = capture.read()

visualOdemetry.old_frame = frame

traj = np.zeros(shape=(600, 800, 3))

every = 100

# while(capture.isOpened()):
while(i<1000):
    ret, frame = capture.read()
    if ret == False:
        break
    visualOdemetry.process_frame(frame,i)
    mono_coord = visualOdemetry.get_mono_coordinates()
    # print mono_coord everym 100 frames
    if i % every == 0:
        print(mono_coord)

    draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
    traj = cv2.circle(traj, (draw_x + 40, draw_z + 10), 1, list((0, 255, 0)), 4)
    cv2.putText(traj, 'Estimated Odometry Position:', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv2.putText(traj, 'Green', (270, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
    cv2.imshow('trajectory', traj)

    i +=1
cv2.imwrite('map.png', traj)
capture.release()
cv2.destroyAllWindows()