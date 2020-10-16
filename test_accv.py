"""-*-coding:utf-8-*-
Test the cub validation result
"""
import os 
import csv
import json
import torch
import numpy as np
import horovod.torch as hvd 
import urllib.request as urt
import torch.backends.cudnn as cudnn

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
        self.test_file = "/data/remote/yy_git_code/cub_baseline/dataset/test_accv.txt"
        # self.test_file = "/data/remote/code/classification_trick_with_model/data/val_imagenet_128w.txt"
        self.test_list = [(x.strip().split(',')[0], int(float(x.strip().split(',')[1]))) for x in open(self.test_file).readlines()]
        self.Resize_size = 380
        # self.input_size = 300
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
        # image = transforms.CenterCrop(self.input_size)(image)
        image = transforms.ToTensor()(image)
        image = self.imagenet_normalization_paramters(image)
        return image, url

    def __len__(self):
        return len(self.test_list)



class CUBModel:
    def __init__(self, model_name, num_classes, model_weights):
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 避免产生结果的随机性
    torch.backends.cudnn.deterministic = True

# predict_logits
def predict_logits(model, data_loader):
    model = model.cuda()
    model = model.eval()
    result_logits = []
    with tqdm(total=len(data_loader), desc="processing predict logits", disable=not verbose) as t:
        with torch.no_grad():
            for idx, data in tqdm(enumerate(data_loader)):
                image_tensor, image_path = data[0], data[1]
                data_logits = model.infer_batch(image_tensor).cpu().numpy()
                for i in range(len(image_path)):
                    result = {"image_path": image_path[i].split('/')[-1], "image_logits": data_logits[i].tolist()} 
                    result_logits.append(result)
    return result_logits

if __name__ == "__main__":
    
    setup_seed(42)
    hvd.init()
    cudnn.benchmark = True

    test_file = "/data/remote/yy_git_code/cub_baseline/dataset/test_accv.txt"
    # test_file = "/data/remote/code/classification_trick_with_model/data/val_imagenet_128w.txt"
    test_dict = {x.split(',')[0]: int(float(x.split(',')[1]))
                 for x in open(test_file).readlines()}

    batch_size = 96
    num_workers = 32
    
    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if torch.cuda.is_available() else {}

    test_dataset = TestDataSet()
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
    )
    testLoader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, **kwargs)
    verbose = 1 if hvd.rank() == 0 else 0

    model_ckpt = "/data/remote/output_ckpt_with_logs/accv/ckpt/efnet-b4_380_lr_01_50_epoch_cutmix/checkpoint-epoch-38.pth.tar"
    model = CUBModel(model_name="efficientnet-b4", num_classes=5000, model_weights=model_ckpt)
    # csv for predict label
    # with open(os.path.join(PATH, "test_predict_accv_efb4.csv"), "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["image_name", "class"])
    #     for idx, data in tqdm(enumerate(testLoader)):
    #         image_tensor, image_path = data[0], data[1]
    #         data_logits = model.infer_batch(image_tensor).cpu().numpy()
    #         data_pred = np.argmax(softmax(data_logits, axis=1), axis=1).tolist()
    #         for i in range(len(image_path)):
    #             writer.writerow([image_path[i].split('/')[-1], str(data_pred[i])])

    # predict logits
    # with open(os.path.join(PATH, "logits/efnetb4_logits.log"), "w") as file:
    #     for idx, data in tqdm(enumerate(testLoader)):
    #         image_tensor, image_path = data[0], data[1]
    #         data_logits = model.infer_batch(image_tensor).cpu().numpy()
    #         for i in range(len(image_path)):
    #             result = {"image_path": image_path[i].split('/')[-1], "image_logits": data_logits[i].tolist()}
    #             file.write(json.dumps(result) + '\n')

    # use hvd ddp
    logits_result = predict_logits(model, testLoader)
    for i in range(hvd.size()):
        if hvd.rank() == i:
            np.save("/data/remote/yy_git_code/cub_baseline/logits/20201014/efnetb4_logits/efnet4_{}.npy".format(i), np.array(logits_result))