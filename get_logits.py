"""
Get the model logits
"""
import os 
import csv
import json
import numpy as np 

from tqdm import tqdm
from scipy.special import softmax


npy_path = "/data/remote/output_ckpt_with_logs/accv/logits/efnetb5-456/"

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


for i in tqdm(range(4)):
    data_npy = os.path.join(npy_path, "efnetb5_{}.npy".format(i))
    logits_result, label_result = load_file(data_npy)
    with open("/data/remote/yy_git_code/cub_baseline/logits/20201016/efnetb5_456_cosinelr_logits.log", "a") as file:
        for data in tqdm(logits_result):
            file.write(json.dumps(data) + '\n')
        
    # with open("/data/remote/yy_git_code/cub_baseline/logits/20201016/r50_448_cosinelr_lbl.csv", "a") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for data in tqdm(label_result):
    #         writer.writerow([data["image_path"], data["image_lbl"]])

        

