import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r"D:\Desktop\Roketsan Level Up\object_tracking\images\0000001_02999_d_0000005.jpg"
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Unable to read the image at {image_path}")

cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width, channels = img.shape
print(f"Image dimensions: Height={height}, Width={width}, Channels={channels}")

resized_img = cv2.resize(img, (width // 2, height // 2))
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

resized_height, resized_width, _ = resized_img.shape
x1 = int(resized_width * 0.25)
y1 = int(resized_height * 0.25)
x2 = int(resized_width * 0.75)
y2 = int(resized_height * 0.75)
cropped_img = resized_img[y1:y2, x1:x2]
cv2.imshow('Cropped Image', cropped_img)    
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.show()

gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img)
plt.imshow(gray_img, cmap='gray')
plt.show()


normalized_img = cropped_img / 255.0
plt.imshow(normalized_img)
plt.show()