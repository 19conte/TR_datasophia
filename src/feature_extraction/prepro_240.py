import os
import numpy as np
from skimage import img_as_ubyte
from PIL import Image
from tqdm import tqdm

from preprocessing import prepro

"""This code is used to generate 240x240 images, used as input of the CNN"""

# def gen_path(txt_file_path):
#     with open(txt_file_path, "r") as f:
#         for line in f:
#             if line[-1] != "\n":
#                 path = line
#             else:
#                 path = line[:-1]
#             yield path
 
# path_data = gen_path("../Solais_Data/valid_img_paths.txt")
# PATHS = [path for path in path_data]

# for path in tqdm(PATHS):
#     date = path.split('/')[-3]
#     file = path.split("/")[-1]
#     filename = '../Solais_Data/mobotix1_prepro_240' + "/" + date + "/" + file
#     image_raw = np.array(Image.open(path))/255
#     r = 390 # observed as the best radius to get rid of non sky pixels
#     new_size = 256 #size of the squared dowscaled image
#     image = prepro(image_raw, r, new_size=new_size, outside_val=0.)
#     image = image/max(1., np.max(image))
#     try:
#         image = Image.fromarray(img_as_ubyte(image))
#     except ValueError:
#         print(file)
#         continue
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     image.save(filename)

txt_file_path = "../Solais_Data/valid_img_paths.txt"

with open(txt_file_path) as f:
    for path in f[:2]:
        date = path.split('/')[-3]
        file = path.split("/")[-1]
        filename = '../Solais_Data/mobotix1_prepro_240bis' + "/" + date + "/" + file
        image_raw = np.array(Image.open(path))/255
        r = 390 # observed as the best radius to get rid of non sky pixels
        new_size = 256 #size of the squared dowscaled image
        image = prepro(image_raw, r, new_size=new_size, outside_val=0.)
        image = image/max(1., np.max(image))
        try:
            image = Image.fromarray(img_as_ubyte(image))
        except ValueError:
            print(file)
            continue
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image.save(filename)