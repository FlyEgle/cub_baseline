import os 
import time
import torch 
import numpy as np

import urllib.request as urt 
from io import BytesIO
from PIL import Image 
from torchvision.transforms import transforms
from models.build_model import BuildModel


def AccvModel(model_name, num_classes, model_weights):
    is_pretrained = False
    net = BuildModel(model_name, num_classes, is_pretrained)()
    if model_weights == "" or model_weights is None:
        return net 
    else:
        model_state_dict = torch.load(model_weights)
        net.load_state_dict(model_state_dict["model"])
        print("Load the accv dataset model")
        return net


def infer_batch(net, data):
    net.eval()
    data = data.cuda()
    with torch.no_grad():
        logits = net(data)
    return logits  


def infer_image(net, data):
    net.eval() 
    data = data.cuda()
    with torch.no_grad():
        logits = model(data).cpu().numpy()
        print(logits.shape)
        for i in range(logits.shape[0]):
            prob = softmax(logits[i])
            lbl = np.argmax(prob)
            print(lbl)

if __name__ == "__main__":
    
    test_file = "/data/remote/yy_git_code/cub_baseline/dataset/test_accv.txt"
    test_list = [x.split(',')[0] for x in open(test_file).readlines()]

    num_classes = 5000
    image_file = test_list[0]
    
    Resize_size = 456
    # self.input_size = 300
    imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    model_ckpt = "/data/remote/output_ckpt_with_logs/accv/ckpt/efnet-finetune/checkpoint-epoch-5.pth.tar"
    model = AccvModel(model_name="efficientnet-b5", num_classes=num_classes, model_weights=model_ckpt)
    start_time = time.time()
    for i in range(100):

        image = Image.open(BytesIO(urt.urlopen(test_list[i]).read()))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transforms.Resize((Resize_size, Resize_size), Image.BILINEAR)(image)
        image = transforms.ToTensor()(image)
        image = imagenet_normalization_paramters(image)

    
        model.cuda()
        model.eval()

        image = image.unsqueeze(0)
        image = image.cuda()
        
        # for i in range(1000):
        with torch.no_grad():
            output = model(image)
    print(time.time() - start_time)