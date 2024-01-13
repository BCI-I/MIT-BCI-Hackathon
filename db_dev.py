#!/usr/bin/env python3
import time
import keyboard

from dt_duckiematrix_protocols import Matrix
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import cv2

SPEED = 10



# vehicle calibration
# - camera
camera_info: Dict[str, int] = {
    "width": 640,
    "height": 480
}

window = plt.imshow(np.zeros((camera_info["height"], camera_info["width"], 3)))
plt.axis("off")
fig = plt.figure(1)
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
plt.pause(0.01)


# new frame callback
def on_new_frame(cframe):
    # get frame as uint8 array
    jpeg = cframe.as_uint8()
    # uint8 array to bgr8
    rgb = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
    return rgb



class WheelsIO:
    def __init__(self):
        # create connection to the matrix engine
        self.matrix = Matrix("localhost", auto_commit=True,)
        print(self.matrix)

        # create connection to the vehicle
        self.robot = self.matrix.robots.DB21M("map_0/vehicle_0")

    def get_input(self):
        key = keyboard.read_key()
        if key == "up":
            return SPEED, SPEED
        elif key == "down":
            return -SPEED, -SPEED
        elif key == "left":
            return -SPEED/2, SPEED/2
        elif key == "right":
            return SPEED/2, -SPEED/2
        else:
            return 0.0, 0.0

    def run(self):
        while not self.is_shutdown:
            # vl, vr = self.get_input()
            # self.robot.drive(left=vl, right=vr)
            
            tof_reading = self.robot._time_of_flight.capture(block=False)
            print(f"TOF reading: {tof_reading}")


            try:
                cframe = self.robot.camera.capture(block=True)
                img = on_new_frame(cframe)
            except:
                pass
            else:
                window.set_data(img)
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.00001)
            
            
            time.sleep(0.001)

    @property
    def is_shutdown(self):
        # TODO: link this to SIGINT
        return False


if __name__ == "__main__":
    node = WheelsIO()
    node.run()