from PIL import Image, ImageDraw
import os

def multi_image_viewer(dir: str, col: int, row: int):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Determine the size of the grid image based on the individual image sizes
    grid_width = max(300, min(col * 200, 800))  
    grid_height = 200 * row
    grid_image = Image.new('RGB', (grid_width, grid_height))

    x = 0
    y = 0

    for img in image_files:
        img_path = os.path.join(dir, img)
        current_image = Image.open(img_path)
        current_image.thumbnail((200, 200))  # Resize the image if needed
        grid_image.paste(current_image, (x, y))
        x += 200
        if x >= grid_width:
            x = 0
            y += 200

    return grid_image

multi_image_viewer("C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/easy/good", 4, 4).show()

# import numpy as np
# import cv2
# import os

# # Path to the folder containing images
# folder_path = "C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/easy/good"

# # Get a list of all image files in the folder
# image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# # Determine the grid layout dimensions based on the number of images
# num_images = len(image_files)
# num_cols = 3  # You can adjust the number of columns as per your requirement
# num_rows = (num_images + num_cols - 1) // num_cols

# # Load the first image to get its dimensions
# first_image_path = os.path.join(folder_path, image_files[0])
# first_image = cv2.imread(first_image_path)
# image_height, image_width, _ = first_image.shape

# # Create a blank grid image
# grid_width = num_cols * image_width
# grid_height = num_rows * image_height
# grid_image = 255 * np.ones(shape=(grid_height, grid_width, 3), dtype=np.uint8)

# # Paste each image onto the grid
# row = 0
# col = 0
# for img in image_files:
#     img_path = os.path.join(folder_path, img)
#     current_image = cv2.imread(img_path)
#     grid_image[row * image_height:(row + 1) * image_height, col * image_width:(col + 1) * image_width] = current_image
#     col += 1
#     if col == num_cols:
#         col = 0
#         row += 1

# # Save the final grid image
# cv2.imshow("path_to_save_final_image/grid_image.jpg", grid_image)
# cv2.waitKey(0)
