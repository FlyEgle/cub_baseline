"""
-*- coding:utf-8 -*-
use the ensemble model for feature filter
query image is find by manused 
"""
import os 
import json
import numpy as np 
from tqdm import tqdm 


def get_cosine_distance(x, y):
    eps = 1e-7
    distance = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + eps)
    return distance

def merge_npy(data_folder):
    data_dict = {}
    for i in tqdm(range(8)):
        data_file = os.path.join(data_folder, "resnest200_feat_{}.npy".format(i))
        result = np.load(data_file, allow_pickle=True)
        for data in tqdm(result):
            image_path = data["image_path"]
            # print(image_path)
            image_feat = data["image_feat"]
            data_dict[image_path] = image_feat
    np.save("./train_feat.npy", np.array(data_dict))

def load_npy(data_file):
    data_dict = np.load(data_file, allow_pickle=True).tolist()
    # print(type(data_dict))
    # new_data_dict = {}
    # for data in tqdm(data_dict):
    #     print(data)
    #     image_path = data["image_path"]
    #     # print(image_path)
    #     image_feat = data["image_feat"]
    #     new_data_dict[image_path] = image_feat
    return data_dict

def calculate_distance(query, data_dict):
    distance_dict = {}
    for key, value in tqdm(data_dict.items()):
        dis = get_cosine_distance(query, np.array(value))
        distance_dict[key] = dis
    return distance_dict

def topk_dict(distance_dict, topk=800, threshold=0.55):
    sorted_dict = sorted(distance_dict.items(), key=lambda item: item[1], reverse=True)
    count = 0
    result_image = []
    # print(sorted_dict)
    for data in sorted_dict:
        if count <= topk:
            result_image.append(data[0])
            count += 1
            if data[1] < threshold:
                break 
        else:
            break 
    return result_image

if __name__ == "__main__":
    query_list = open("/data/remote/yy_git_code/cub_baseline/dataset/filter_data/clean.txt").readlines()
    # query = query_list[0].strip() 
    # print(query)
    np_file = "/data/remote/output_ckpt_with_logs/accv/logits/r200_feat/train_feat.npy"
    data_dict = load_npy(np_file)
    # query_feature = None
    # if query in data_dict.keys():
    #     print(1)
    for i in range(len(query_list)):
        query_feature = np.array(data_dict[query_list[i].strip()])
        distance_dict = calculate_distance(query_feature, data_dict)
        result_image = topk_dict(distance_dict)
        data_head = "http://221.228.84.39:8901/dataset/jiangmingchao/dataset/accv_55w/train"
        with open("/data/remote/yy_git_code/cub_baseline/dataset/filter_data/query_{}_file.log".format(i), "w") as file:
            for data in result_image:
                data_folder = data.split('_')[0]
                image_path = os.path.join(data_head, data_folder + '/' + data)
                file.write(image_path + '\n')
    # merge_npy("/data/remote/output_ckpt_with_logs/accv/logits/r200_feat/")



