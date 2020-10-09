"""-*-coding:utf-8-*-
Test the cub validation result
"""
import os 
import torch
import numpy as np
import urllib.request as urt

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from scipy.special import softmax
from torchvision.transforms import transforms
from models.build_model import BuildModel
from torch.utils.data import Dataset, DataLoader


PATH = "/data/remote/yy_git_code/cub_baseline"

class TestDataSet(Dataset):
    def __init__(self):
        super(TestDataSet, self).__init__()
        self.test_file = "/data/remote/yy_git_code/cub_baseline/dataset/cub_test.txt"
        # self.test_file = "/data/remote/code/classification_trick_with_model/data/val_imagenet_128w.txt"
        self.test_list = [(x.strip().split(',')[0], int(float(x.strip().split(',')[1]))) for x in open(self.test_file).readlines()]
        self.Resize_size = 512
        self.input_size = 448
        self.imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        url, lbl = self.test_list[idx][0], self.test_list[idx][1]
        image = Image.open(BytesIO(urt.urlopen(url).read()))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transforms.Resize((self.Resize_size, self.Resize_size), Image.BILINEAR)(image)
        image = transforms.CenterCrop(self.input_size)(image)
        image = transforms.ToTensor()(image)
        image = self.imagenet_normalization_paramters(image)
        return image, url

    def __len__(self):
        return len(self.test_list)



class CUBModel:
    def __init__(self, model_name, num_classes, model_weights):
        self.Resize_size = 512
        self.input_size = 448
        self.model_name = model_name
        self.num_classes = num_classes
        self.imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        is_pretrained = False
        self.net = BuildModel(
            self.model_name, self.num_classes, is_pretrained)()
        model_state_dict = torch.load(model_weights, map_location="cpu")
        self.net.load_state_dict(model_state_dict["model"])
        # self.net.load_state_dict(model_state_dict)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def _process_image(self, url):
        image = Image.open(BytesIO(urt.urlopen(url).read()))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transforms.Resize(
            (self.Resize_size, self.Resize_size), Image.BILINEAR)(image)
        image = transforms.CenterCrop(self.input_size)(image)
        image = transforms.ToTensor()(image)
        image = self.imagenet_normalization_paramters(image)
        image = image.unsqueeze(0)
        return image

    def infer(self, url):
        self.net.eval()
        with torch.no_grad():
            image = self._process_image(url)
            image = image.to(self.device)
            logits = self.net(image)
            predict_label = np.argmax(softmax(logits.cpu().numpy(), axis=1))
        return url, predict_label


    def infer_batch(self, data):
        self.net.eval()
        data = data.to(self.device)
        with torch.no_grad():
            logits = self.net(data)
        return logits


def calculate_accuracy(test_gt, test_pd):
    gt_dict = {x.strip().split(',')[0]: x.strip().split(',')[1] for x in open(test_gt).readlines()}
    pd_dict = {x.strip().split(',')[0]: x.strip().split(',')[1] for x in open(test_pd).readlines()}
    count = 0
    for key, value in pd_dict.items():
        if key in gt_dict.keys():
            if value == gt_dict[key]:
                count += 1
    print("Accuracy: ", count / len(pd_dict))



if __name__ == "__main__":
    
    test_file = "/data/remote/yy_git_code/cub_baseline/dataset/cub_test.txt"
    # test_file = "/data/remote/code/classification_trick_with_model/data/val_imagenet_128w.txt"
    test_dict = {x.split(',')[0]: int(float(x.split(',')[1]))
                 for x in open(test_file).readlines()}
    model_ckpt = "/data/remote/output_ckpt_with_logs/cub/ckpt/best_acc_08412.pth.tar"
    # model_ckpt = "/data/remote/code/classification_trick_with_model/models/weights/resnet50.pth"
    model = CUBModel(model_name="resnet50", num_classes=200, model_weights=model_ckpt)
    batch_size = 128
    num_workers = 32
    test_dataset = TestDataSet()

    testLoader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    with open(os.path.join(PATH, "test_predict_cub.txt"), "w") as file:
        for idx, data in tqdm(enumerate(testLoader)):
            image_tensor, image_path = data[0], data[1]
            data_logits = model.infer_batch(image_tensor).cpu().numpy()
            data_pred = np.argmax(softmax(data_logits, axis=1), axis=1).tolist()
            for i in range(len(image_path)):
                file.write(image_path[i] + ',' + str(data_pred[i]) + '\n')

        
    calculate_accuracy(os.path.join(PATH, "test_predict_cub.txt"), test_file)