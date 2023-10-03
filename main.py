import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
  
# assign and open image
url = 'https://media.geeksforgeeks.org/wp-content/cdn-uploads/20210401173418/Webp-compressed.jpg'
response = requests.get(url, stream=True)
  
with open('image.png', 'wb') as f:
    f.write(response.content)
  
img = cv2.imread('image.png')
  
# Converting the image into gray scale for faster
# computation.
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Calculating the SVD
u, s, v = np.linalg.svd(gray_image, full_matrices=False)
  
# inspect shapes of the matrices
print(f'u.shape:{u.shape},s.shape:{s.shape},v.shape:{v.shape}')