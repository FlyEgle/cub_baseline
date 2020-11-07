import os 
import csv
import json
import numpy as np 

from tqdm import tqdm
from scipy.special import softmax


npy_path = "/data/remote/output_ckpt_with_logs/accv/logits/r200_train"

def load_file(data_npy):
    logits_list = []
    label_list = []
    data = np.load(data_npy, allow_pickle=True)
    for d in data:
        logits = {}
        label = {}
        image_url = d["image_path"].split('/')[-1]
        image_logits = d["image_logits"]

        logits["image_path"] = image_url
        logits["image_logits"] = image_logits

        label["image_path"] = image_url 
        # print(image_logits.shape)
        label["image_lbl"] = np.argmax(softmax(np.array(image_logits)))
        logits_list.append(logits)
        label_list.append(label)
    return logits_list, label_list


for i in tqdm(range(8)):
    data_npy = os.path.join(npy_path, "r200_{}.npy".format(i))
    logits_result, label_result = load_file(data_npy)

    with open("/data/remote/yy_git_code/cub_baseline/logits/20201106/r200_train_file.log", "a") as file:
        for data in tqdm(label_result):
            file.write(data["image_path"] +','+str(data["image_lbl"]) + '\n')