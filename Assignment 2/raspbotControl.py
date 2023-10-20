from KeyboardInputDriver import KeyboardInput
# from ControllerInputDriver import ControllerInput
from Car import Car
import time
from create_map import stop_odemetry, start_odemetry
import cv2
import os

global isCircling
isCircling = False

def drive_forward():
    car.control_car(car.power, car.power)
def turn_left():
    car.control_car(-car.power, car.power)
def turn_right():
    car.control_car(car.power, -car.power)
def drive_backward():
    car.control_car(-car.power, -car.power)

def rotate_90(counterclockwise = False):
    TURN_TIME = 1
    TURN_POWER = 128

    if counterclockwise:
        TURN_POWER *= -1

    car.control_car(TURN_POWER, -TURN_POWER)
    time.sleep(TURN_TIME)
    car.control_car(0, 0)

def rotate_90_ccw():
    rotate_90(True)

def rotate_90_cw():
    rotate_90(False)

def move_circle():
    global isCircling
    if isCircling:
        isCircling = False
        car.control_car(0, 0)
    else:
        V = 50
        W = 38 # .14 meters
        R = 50 # .5 meters should be 135 "power seconds", but 50 ends up somewhere between .5 and .75 meters

        # 100 p = .37 m/s
        V_LEFT = V - (V*W)/(2 * R)
        V_RIGHT = 2 * V - V_LEFT
        car.control_car(int(V_LEFT), int(V_RIGHT))
        isCircling = True
        takePictures()

def takePictures():
    capture = cv2.VideoCapture(0)
    cwd = os.getcwd()
    i = 0
    while isCircling:
        i += 1
        ret, frame = capture.read()
        cv2.imwrite(cwd + f'/video/image{i}.png', frame)
        time.sleep(.05)

def move_square(down = True):
    if not down:
        return

    STRAIGHT_TIME = 2
    STRAIGHT_POWER = 60

    for i in range(0, 4):
        car.control_car(STRAIGHT_POWER, STRAIGHT_POWER) #Forward
        time.sleep(STRAIGHT_TIME)
        rotate_90()
        time.sleep(.2)

car = Car()
keyboardInput = KeyboardInput()
# controllerInput = ControllerInput()

def left_stick_drive(x_in, y_in):
    pass

def two_stick_drive(right_stick, y_in):
    if right_stick:
        car.control_car(car.last_left, y_in)
    else:
        car.control_car(y_in, car.last_right)



def stop_on_release(isDown, function):
    if not isDown:
        car.control_car(0, 0)
        return
    function()

def run_on_release(isDown, function):
    if isDown:
        function()

if __name__ == '__main__':

    keyboardInput.add_listener('w', lambda isDown: stop_on_release(isDown, drive_forward))
    keyboardInput.add_listener('a', lambda isDown: stop_on_release(isDown, turn_left))
    keyboardInput.add_listener('d', lambda isDown: stop_on_release(isDown, turn_right))
    keyboardInput.add_listener('s', lambda isDown: stop_on_release(isDown, drive_backward))
    keyboardInput.add_listener('q', lambda isDown: run_on_release(isDown, rotate_90_ccw))
    keyboardInput.add_listener('e', lambda isDown: run_on_release(isDown, rotate_90_cw))
    keyboardInput.add_listener('c', lambda isDown: run_on_release(isDown, move_circle))

    keyboardInput.enable()

