import os 

image_head = "http://cache.filer.yy.com:9899/dataset/jiangmingchao/dataset/images/"
image_list = open("./images.txt").readlines()
image_dict = {}
for image in image_list:
    image_id = int(float(image.strip().split(' ')[0]))
    image_name = image.strip().split(' ')[1]
    image_dict[image_id] = os.path.join(image_head, image_name)

image_label_dict = {int(float(x.strip().split(" ")[0])): int(float(x.strip().split(" ")[1])) for x in open("./image_class_labels.txt").readlines()}
train_test_image_id_list = {int(float(x.strip().split(" ")[0])): int(float(x.strip().split(" ")[1])) for x in open("./train_test_split.txt").readlines()}

# train
train_image_dict = {}
test_image_dict = {}
for key, value in train_test_image_id_list.items():    
    image_label = image_label_dict[key]
    image_url = image_dict[key]
    if value == 1:
        train_image_dict[image_url] = image_label
    else:
        test_image_dict[image_url] = image_label

with open("./cub_train.txt", "w") as file:
    for key, value in train_image_dict.items():
        file.write(str(key) + ',' + str(value-1) + '\n')

with open("./cub_test.txt", "w") as file:
    for key, value in test_image_dict.items():
        file.write(str(key) + ',' + str(value-1) + '\n')