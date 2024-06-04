# from PIL import Image, ImageDraw
# import os

# def multi_image_viewer(dir: str, col: int, row: int):
#     # Get a list of all image files in the folder
#     image_files = [f for f in os.listdir(dir) if f.endswith('.jpg') or f.endswith('.png')]

#     # Determine the size of the grid image based on the individual image sizes
#     grid_width = max(300, min(col * 200, 800))  
#     grid_height = 200 * row
#     grid_image = Image.new('RGB', (grid_width, grid_height))

#     x = 0
#     y = 0

#     for img in image_files:
#         img_path = os.path.join(dir, img)
#         current_image = Image.open(img_path)
#         current_image.thumbnail((200, 200))  # Resize the image if needed
#         grid_image.paste(current_image, (x, y))
#         x += 200
#         if x >= grid_width:
#             x = 0
#             y += 200

#     return grid_image

# # multi_image_viewer("C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/easy/good", 4, 4).show()

# dir = "C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/easy/good"
# print(os.path.dirname(dir))

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 


def scale_contrast(mean_shift, contrast_scaling, img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray_img)
    std_dev = np.std(gray_img)
    normalized_img = (gray_img - mean_val) / std_dev
    # Modify the constants for contrast scaling and mean shift
    scaled_contrast_img = mean_shift + contrast_scaling * normalized_img
    scaled_contrast_img = np.clip(scaled_contrast_img, 0, 255).astype(np.uint8)
    return scaled_contrast_img

img = cv2.imread('C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/test/img1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
absolute_transformed = np.abs(gray_img)
cv2.imwrite('ok.jpg', absolute_transformed)



