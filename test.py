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

# multi_image_viewer("C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/easy/good", 4, 4).show()

dir = "C:/Users/nguyenhuyhoa/Pictures/Saved Pictures/easy/good"
print(os.path.dirname(dir))