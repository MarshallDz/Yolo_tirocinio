from ultralytics import YOLO
import cv2
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np 
import os
from os import listdir

import warnings 
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')  # Suppress Matplotlib warnings

img = cv2.imread("composystemSuperioreV3_flip_agumentation/test/images/28_png.rf.51b6cf2f50a79fcf3db10175319f9221.jpg")
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255

model = YOLO("/home/user/yolo/yolo_tune_s/tune/weights/best.pt")
model = model.cpu()
target_layers =[model.model.model[-2]]
cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
cv2.imshow("windo2", cam_image)
cv2.waitKey(0)
filename = 'result.png'
if os.path.exists(f'{os.getcwd()}/result.png'):
    files = [f for f in listdir(os.getcwd()) if f[0:6] == 'result']
    filename = f'result{len(files)}.png'

cv2.imwrite(filename, cam_image)