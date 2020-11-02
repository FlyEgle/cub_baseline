"""
-*-coding:utf-8-*-
Filter image 
"""
import PIL 
import json
import urllib.request as urt
import multiprocessing as mp 

from PIL import Image 
from io import BytesIO
from tqdm import tqdm
from PIL import ImageSequence


def read_image(image_file):
    try:
        image = urt.urlopen(image_file).read()
        image = Image.open(BytesIO(image))
        if type(image) == PIL.GifImagePlugin.GifImageFile:
            image_list = []
            for frame in ImageSequence.Iterator(image):
                image_list.append(image)

            if len(image_list) > 1:
                return {"image_path": image_file, "status": 1}
            else:
                return {"image_path": image_file, "status": 0}
        else:
            return {"image_path": image_file, "status": 0}

    except Exception as e:
        print("image_error", image_file)
    

if __name__ == "__main__":
    data_file = "/data/remote/yy_git_code/cub_baseline/dataset/train_accv_pingtai.txt"
    data_list = [x.strip().split(',')[0] for x in open(data_file).readlines()]
    pool = mp.Pool(64)
    result = pool.map(read_image, data_list)
    
    with open("/data/remote/yy_git_code/cub_baseline/dataset/train_accv_gif.txt", "w") as file:
        for data in result:
            file.write(json.dumps(data) + '\n')

# data_file = "/data/remote/yy_git_code/cub_baseline/dataset/train_accv_gif.txt"
# data_list = open(data_file).readlines()

# with open("/data/remote/yy_git_code/cub_baseline/dataset/train_accv_gif_clean.txt", "w") as file:
#     for data in data_list:
#         data_json = json.loads(data.strip())
#         image_file = data_json["image_path"]
#         image_gif = data_json["status"]
#         if image_gif == 1:
#             file.write(image_file + '\n')

