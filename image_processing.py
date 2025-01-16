import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import active_contour
from skimage.draw import disk
from skimage import io, color

os.makedirs('output', exist_ok=True)

def crop_image(image):
    cropped = image[0:400, 100:450]
    return cropped

def apply_filters(image):
	median_filtered = cv2.medianBlur(image, 5)
	gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)
	gray = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
	enhanced_image = clahe.apply(gray)
	return enhanced_image

def fixed_threshold(image, threshold_value=150):
	_, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
	return binary

def otsu_threshold(image):
	thresh_value = threshold_otsu(image)
	_, binary = cv2.threshold(image, int(thresh_value), 255, cv2.THRESH_BINARY)
	return binary

def sobel_edge_detection(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    sobel = cv2.magnitude(grad_x, grad_y)  # Gradient magnitude
    sobel = cv2.convertScaleAbs(sobel)  # Convert to uint8
    return sobel

def laplacian_edge_detection(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

def prewitt_edge_detection(image):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    prewitt = cv2.magnitude(grad_x, grad_y)
    prewitt = cv2.convertScaleAbs(prewitt)
    return prewitt

def scharr_edge_detection(image):
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharr = cv2.magnitude(grad_x, grad_y)
    scharr = cv2.convertScaleAbs(scharr)
    return scharr

def canny_edge_detection(image):
    canny = cv2.Canny(image, 50, 100)
    return canny

def count_bright_pixels(image, roi_coords):
    x, y, w, h = roi_coords
    roi = image[y:y+h, x:x+w]

    # Create a binary mask for the ROI
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255  # Set ROI to white (masking area)

    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    bright_pixels_in_roi = cv2.bitwise_and(binary_image, mask)
    count_in_roi = np.sum(bright_pixels_in_roi == 255)

    mask_inverted = cv2.bitwise_not(mask)
    bright_pixels_outside_roi = cv2.bitwise_and(binary_image, mask_inverted)
    count_outside_roi = np.sum(bright_pixels_outside_roi == 255)

    return count_in_roi, count_outside_roi

def process_image(image_path):

	image = cv2.imread(image_path)
	cropped_image = crop_image(image)
	filtered_image = apply_filters(cropped_image)
	binary_fixed = fixed_threshold(filtered_image)
	binary_otsu = otsu_threshold(filtered_image)

	fig1, ax = plt.subplots(2, 2)

	ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	ax[0, 0].set_title('Original Image')
	ax[0, 1].imshow(filtered_image, cmap='gray')
	ax[0, 1].set_title('Filtered Image')
	ax[1, 0].imshow(binary_fixed, cmap='gray')
	ax[1, 0].set_title('Fixed Threshold')
	ax[1, 1].imshow(binary_otsu, cmap='gray')
	ax[1, 1].set_title('OTSU Threshold')

	plt.tight_layout()
	#plt.show()
	base_name = os.path.splitext(os.path.basename(image_path))[0]
	fig1.savefig(f'output/{base_name}_mask.jpg')
	plt.close(fig1) 
    
	sobel = sobel_edge_detection(filtered_image)
	laplacian = laplacian_edge_detection(filtered_image)
	prewitt = prewitt_edge_detection(filtered_image)
	scharr = scharr_edge_detection(filtered_image)
	canny = canny_edge_detection(filtered_image)

	fig2, ax = plt.subplots(2, 3)

	ax[0, 0].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
	ax[0, 0].set_title('Filtered Image')
	ax[0, 1].imshow(sobel, cmap='gray')
	ax[0, 1].set_title('Sobel')
	ax[0, 2].imshow(laplacian, cmap='gray')
	ax[0, 2].set_title('Laplacian')
	ax[1, 0].imshow(prewitt, cmap='gray')
	ax[1, 0].set_title('Prewitt')
	ax[1, 1].imshow(scharr, cmap='gray')
	ax[1, 1].set_title('Scharr')
	ax[1, 2].imshow(canny, cmap='gray')
	ax[1, 2].set_title('Canny')

	plt.tight_layout()
	#plt.show()
	base_name = os.path.splitext(os.path.basename(image_path))[0]
	fig2.savefig(f'output/{base_name}_edges.jpg')
	plt.close(fig2)
	cv2.imwrite(f'output/{base_name}scharr.jpg', scharr)
	#final = fixed_threshold(scharr)
	roi_coords = (100, 0, 220, 350)
	count_in, count_out = count_bright_pixels(scharr, roi_coords)
	#print(f"Number of bright pixels inside the ROI: {count_in}")
	#print(f"Number of bright pixels outside the ROI: {count_out}")
    
	if (count_out >= 2200 or count_in <= 16000):
		print(base_name, "Rope surface is in bad condition!")
		return 0

#SINGLE IMAGE PROCESSING
#process_image('images/00.jpg')
#print("Done! :)")

#PROCESSING ALL IMAGES IN FOLDER
for filename in os.listdir('images'):
	if filename.endswith('.jpg'):
		process_image(os.path.join('images', filename))
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
print("All images processed :)")
