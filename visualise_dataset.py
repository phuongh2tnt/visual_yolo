"""
This is the official version of "visualization.py" for the deliverables
"""
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import torchvision.transforms as T
import warnings

warnings.filterwarnings('ignore')


COLORS = [(0, 255, 0), (255, 0, 0)]  # RGB


def get_classes(file):
    """
    Get a list of class names in the dataset
    :param file: text file containing class names
    :return: a list
    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def draw_box(draw, label, color, left, top, right, bottom):
    """
    Drawing a bounding box with label at specified coordinates
    :param draw: a PIL ImageDraw object. Must be RGB (i.e., default).
    :param label: label
    :param color: color
    :param left: left coordinate (i.e.x1)
    :param top: top coordinate (i.e.y1)
    :param right: right coordinate (i.e.x2)
    :param bottom: bottom coordinate (i.e.y2)
    """
    font = ImageFont.truetype('FiraMono-Medium.otf', size=25)
    label_size = draw.textlength(label, font)

    # Adjust the bounding box if it is outside the input image (just for a better visualization)
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 3])

    # Draw a rectangle covering the detected object
    draw.rectangle([left, top, right, bottom], outline=color, width=3)  # box in red color

    # Draw an outside rectangle covering the text
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)

    # Draw the text in white color
    draw.text(tuple(text_origin), label, fill=(255, 255, 255), font=font)  # text in white color


def draw_boundary(draw, color, points, transparent=125):
    """
    Drawing the boundary with specified coordinates
    :param draw: a PIL ImageDraw object. Must be RGBA.
    :param color: color to use for the fill
    :param transparent: transparent level
    :param points: list of 4 points representing the oriented bounding box. Each item is a tuple (x, y).
    """
    # Draw a rectangle covering the detected object
    color = color + (transparent,)  # become a tuple of size 4
    draw.polygon(xy=points, fill=color, outline=(255, 255, 255, 0))


def visualize_detection(img, label_path, class_names):
    """
    Visualize detection ground-truth
    :param img: input PIL image
    :param label_path: path of the annotation file (i.e. text file)
    :param class_names: list of class names
    :return: PIL Image with annotation bounding boxes
    """
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # Read the annotation file
    file = open(label_path, 'r')
    lines = file.readlines()

    for k in range(len(lines)):
        coord = [float(i) for i in lines[k].split()]

        obj = int(coord[0])
        left = int(coord[1] * W - coord[3] * W / 2)
        top = int(coord[2] * H - coord[4] * H / 2)
        right = int(coord[1] * W + coord[3] * W / 2)
        bottom = int(coord[2] * H + coord[4] * H / 2)
        draw_box(draw, class_names[obj], COLORS[obj], left, top, right, bottom)

    return img


def visualize_segmentation(img, label_path):
    """
    Visualize segmentation ground-truth
    :param img: input PIL image
    :param label_path: path of the annotation file (i.e. text file)
    :return: PIL Image with annotation bounding boxes
    """
    W, H = img.size
    draw = ImageDraw.Draw(img, 'RGBA')

    # Read the annotation file
    file = open(label_path, 'r')
    lines = file.readlines()

    for k in range(len(lines)):
        coord = [float(i) for i in lines[k].split()]

        obj = int(coord[0])
        points = []
        for i in range(1, len(coord) - 1, 2):
            x = int(coord[i] * W)
            y = int(coord[i + 1] * H)
            points.append((x, y))

        draw_boundary(draw, COLORS[obj], points)

    return img


if __name__ == '__main__':

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Visualise the vessel dataset')
    args.add_argument('-i', '--input', default=None, type=str, help='Path of dataset folder')
    args.add_argument('--segment', action='store_true')
    cmd_args = args.parse_args()

    txt_folder = cmd_args.input + os.sep + 'labels/train'
    img_folder = cmd_args.input + os.sep + 'images/train'
    out_folder = cmd_args.input + os.sep + ''/content/drive/My Drive/AI/el' #'visualization'

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    txt_files = glob.glob(txt_folder + os.sep + '*.txt')
    if cmd_args.segment:
        for txt_file in tqdm(txt_files):
            if os.path.exists(img_folder + os.sep + os.path.basename(txt_file).replace('txt', 'bmp')):
                img_file = img_folder + os.sep + os.path.basename(txt_file).replace('txt', 'bmp')
            elif os.path.exists(img_folder + os.sep + os.path.basename(txt_file).replace('txt', 'jpg')):
                img_file = img_folder + os.sep + os.path.basename(txt_file).replace('txt', 'jpg')
            else:
                img_file = img_folder + os.sep + os.path.basename(txt_file).replace('txt', 'png')

            img = Image.open(img_file)
            img = img.convert('RGB')
            W, H = img.size

            combined_im = Image.new('RGB', (2 * W + 5, H), (255, 255, 255))  # white offset of 10 pixels
            color_bar = Image.open('color_bar_2.png').convert('RGB')
            _color_bar = T.Resize((int((2 * W + 5) * 0.03), 2 * W + 5),
                                  interpolation=T.InterpolationMode.BILINEAR)(color_bar)
            combined_im.paste(_color_bar, (0, 0))
            combined_im.paste(img, (0, int((2 * W + 5) * 0.03)))
            out_img = visualize_segmentation(img, txt_file)
            combined_im.paste(out_img, (W + 5, int((2 * W + 5) * 0.03)))
            combined_im.save(out_folder + os.sep + os.path.basename(img_file))

    else:
        class_names = get_classes(cmd_args.input + os.sep + 'classes.txt') #file class
        for txt_file in tqdm(txt_files):
            img_file = img_folder + os.sep + os.path.basename(txt_file).replace('txt', 'jpg')
            img = Image.open(img_file)
            img = img.convert('RGB')
            W, H = img.size
            combined_im = Image.new('RGB', (2 * W + 5, H), (255, 255, 255))  # white offset of 10 pixels
            combined_im.paste(img, (0, 0))
            out_img = visualize_detection(img, txt_file, class_names)
            combined_im.paste(out_img, (W + 5, 0))
            combined_im.save(out_folder + os.sep + os.path.basename(img_file))
