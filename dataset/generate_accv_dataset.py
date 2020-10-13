import os 

train_head = "http://cache.filer.yy.com:9899/dataset/jiangmingchao/dataset/accv_55w/train"
test_head = "http://cache.filer.yy.com:9899/dataset/jiangmingchao/dataset/accv_55w/test"
train_folder = "/data/local/accv_55w/train/"
test_folder = "/data/local/accv_55w/test/"

train_list = os.listdir(train_folder)
with open("/data/local/accv_55w/train_accv.txt", "w") as file:
    for data in train_list:
        data_classes = os.path.join(train_folder, data)
        image_list = os.listdir(data_classes)
        image_lbl = int(float(data))
        for image in image_list:

            image_url = os.path.join(train_head, os.path.join(data, image))
            file.write(str(image_url) + ',' + str(image_lbl) + '\n')

test_list = os.listdir(test_folder)
with open("/data/local/accv_55w/test_accv.txt", "w") as file:
    for data in test_list:
        image_url = os.path.join(test_head, data)
        file.write(image_url + ',' + '0' + '\n')
