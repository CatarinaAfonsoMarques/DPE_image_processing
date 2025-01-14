import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import active_contour
from skimage.draw import disk
from skimage import io, color

#(width, height, 3) -> (480,640,3)
#BGR (blue green red)
#calculating distance between edges?
#counting colored pixels?
#above x nmb pixels then rope in bad condition

#process each image 
# 1.filtering and noise removal 
# 2.segmentation (lines) -> if perfectly straight just add them at x pixel
# 3.binary mask? 
# 4.output image on screen and LED (0,1) show() & save image

os.makedirs('output', exist_ok=True)

def apply_filters(image):
    median_filtered = cv2.medianBlur(image, 5)
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)
    return gaussian_filtered

def fixed_threshold(image, threshold_value=200):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
	return binary

def otsu_threshold(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh_value = threshold_otsu(gray)
	_, binary = cv2.threshold(gray, int(thresh_value), 255, cv2.THRESH_BINARY)
	return binary

''' NOT WORKING
def active_contours_segmentation(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = sobel(gray)
	init_rect = np.array([
        [0, 0],                   # Top-left corner
        [0, 480],           # Top-right corner
        [640 , 480],    # Bottom-right corner
        [640 , 0],           # Bottom-left corner
        [0, 0]                   # Back to top-left to close the loop
    ])

	snake = active_contour(edges, init_rect, alpha=10, beta=1, max_num_iter=200)

	binary_mask = np.zeros_like(edges, dtype=np.uint8)
	snake_points = np.round(snake).astype(int)
	cv2.fillPoly(binary_mask, [snake_points], color=255)  # Fill the contour region

	return binary_mask'''

def process_image(image_path):

	image = cv2.imread(image_path)
	filtered_image = apply_filters(image)

	binary_fixed = fixed_threshold(filtered_image)
	binary_otsu = otsu_threshold(filtered_image)
	#binary_active = active_contours_segmentation(filtered_image)

	fig, ax = plt.subplots(2, 2)

	ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	ax[0, 0].set_title('Original Image')

	ax[1, 1].imshow(binary_fixed, cmap='gray')
	ax[1, 1].set_title('Fixed Threshold')

	ax[1, 0].imshow(binary_otsu, cmap='gray')
	ax[1, 0].set_title('OTSU Threshold')

	#ax[1, 1].imshow(binary_active, cmap='gray')
	#ax[1, 1].set_title('Active Contours')

	plt.tight_layout()
	plt.show()

	base_name = os.path.splitext(os.path.basename(image_path))[0]
	cv2.imwrite(f'output/{base_name}_fixed.jpg', binary_fixed)
	cv2.imwrite(f'output/{base_name}_otsu.jpg', binary_otsu)
	#cv2.imwrite(f'output/{base_name}_active.jpg', binary_active)

#process_image('images/00.jpg')			#SINGLE IMAGE PROCESSING

#PROCESSING ALL IMAGES IN FOLDER

for filename in os.listdir('images'):
	if filename.endswith('.jpg'):
		process_image(os.path.join('images', filename))
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
print("All images processed :)")
