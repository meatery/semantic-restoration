import time

import cv2
import numpy as np
from PIL import Image

from model import CYCLEGAN

if __name__ == "__main__":
    ourmodel = OURMODEL()
    mode = "predict"

    #video_path      = 0
    #video_save_path = ""
    #video_fps       = 25.0
    #test_interval   = 100
    #fps_image_path  = "img/1.png"

    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = ourmodel.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict'.")
