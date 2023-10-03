import cv2
from IPython.display import clear_output
import numpy as np

clear_output()

image = cv2.imread("peppers.png")
show = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 15)