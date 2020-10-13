# merge logits
import os 
import json
import csv
import numpy as np

from scipy.special import softmax
from tqdm import tqdm

data_file1 = "/data/remote/yy_git_code/cub_baseline/test_predict_accv_logits.log"
data_file2 = "/data/remote/yy_git_code/cub_baseline/test_predict_accv_logits2.log"


def trainslate_dict(data_file):
    data_list = open(data_file).readlines()
    data_dict = {}
    for data in tqdm(data_list):
        data_json = json.loads(data.strip())
        image_path = data_json["image_path"]
        image_logits = data_json["image_logits"]
        data_dict[image_path] = image_logits
    return data_dict

data_dict1 = trainslate_dict(data_file1)
data_dict2 = trainslate_dict(data_file2)


PATH = "/data/remote/yy_git_code/cub_baseline/ensemble"
if not os.path.exists(PATH):
    os.mkdir(PATH)

with open(os.path.join(PATH, "resnet50_224_efnetb2_logits.log"), "w") as file:
    for key, value in tqdm(data_dict1.items()):
        if key in data_dict2.keys():
            array1 = np.array(value)
            array2 = np.array(data_dict2[key])
            # print(array1.shape)   
            m_array = (array1 + array2) / 2
            data_json = {"image_path":key, "image_logits":m_array.tolist()}
            file.write(json.dumps(data_json) + '\n')
# # merge 
# with open(os.path.join(PATH, "resnet50_efnetb3_pred.csv"), "w") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["image_name", "class"])
#     for key, value in tqdm(data_dict1.items()):
#         if key in data_dict2.keys():
#             array1 = np.array(value)
#             array2 = np.array(data_dict2[key])
#             # print(array1.shape)   
#             m_array = (array1 + array2) / 2
#             predlabel = np.argmax(softmax(m_array))
#             writer.writerow([key, str(predlabel)])
            


