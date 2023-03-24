import numpy as np
from PIL import Image
import cv2
import os


def generate_black_images(water_line_root, image_id, image_width, image_height):
    size = (image_width, image_height)
    # 全黑.可以用在屏保
    img = Image.new('1', size, 0)
    img.save(os.path.join(water_line_root, image_id + '.png'))
    return Image.open(os.path.join(water_line_root, image_id + '.png'))
