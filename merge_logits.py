# merge logits
import os 
import json
import csv
import numpy as np

from scipy.special import softmax
from tqdm import tqdm

# data_file1 = "/data/remote/yy_git_code/cub_baseline/logits/regnet_54.195.log"
# data_file2 = "/data/remote/yy_git_code/cub_baseline/logits/regnet_54.195.log"
# data_file1 = "/data/remote/yy_git_code/cub_baseline/logits/r200_57228.log"
# data_file2 = "/data/remote/yy_git_code/cub_baseline/logits/r200_57105_clean.log"
# data_file3 = "/data/remote/yy_git_code/cub_baseline/logits/r200_56.145_raw.log"
# data_file1 = "/data/remote/yy_git_code/cub_baseline/ensemble/regnet12_resnest200_clean_mean.log"
# data_file2 = "/data/remote/yy_git_code/cub_baseline/ensemble/regnet12_resnest200_raw_mean.log"
data_file1 = "/data/remote/yy_git_code/cub_baseline/logits/20201108/r200_57828_clean.log"
data_file2 = "/data/remote/yy_git_code/cub_baseline/logits/20201108/r101_56405_clean.log"
data_file3 = "/data/remote/yy_git_code/cub_baseline/logits/r200_57105_clean.log"

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
data_dict3 = trainslate_dict(data_file3)


PATH = "/data/remote/yy_git_code/cub_baseline/ensemble"
if not os.path.exists(PATH):
    os.mkdir(PATH)

# with open(os.path.join(PATH, "regnet12_resnest200_raw_mean.log"), "w") as file:
#     for key, value in tqdm(data_dict1.items()):
#         if key in data_dict1.keys():
#             array1 = np.array(value)
#             array2 = np.array(data_dict2[key])
#             # array3 = np.array(data_dict3[key])
#             # print(array1.shape)   
#             # m_array = (array1 + array2 + array3) / 3
#             m_array = (array1 + array2) / 2
#             data_json = {"image_path":key, "image_logits":m_array.tolist()}
#             file.write(json.dumps(data_json) + '\n')
# ensemble method
ensemble_method = "mean"

# # merge 
with open(os.path.join(PATH, "r200_r101_repeat_200_clean.csv"), "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "class"])
    for key, value in tqdm(data_dict1.items()):
        if key in data_dict1.keys():
            array1 = np.array(value)
            array2 = np.array(data_dict2[key])
            array3 = np.array(data_dict3[key])
            # print(array1.shape)   
            if ensemble_method == "mean":
                # m_array = (array1 + array2) / 2
                m_array = (array1 + array2 + array3) / 3
            # TODO: vote method
            predlabel = np.argmax(softmax(m_array))
            writer.writerow([key, str(predlabel)])
            


