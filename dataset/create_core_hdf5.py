from PIL import Image
import numpy as np
import os
import h5py
import json


def open_image(path):
    img = Image.open(path)
    img = np.asarray(img)

    if len(img.shape) != 3:  # gray-scale img
        print("greyscale")
        img = np.stack([img, img, img], axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))
    return img

def create_hdf5_train(source, run, experience, dest):
    filelist_path = os.path.join(source, "SSLAD-2D/labeled/batches_filelists/NI_inc_ub_cat/run{}/train_batch_0{}_filelist.txt"
                                 .format(run, experience))
    if not os.path.exists(filelist_path):
        FileNotFoundError("path {} not found".format(filelist_path))
    imgs = []
    labels = []
    filelist = open(filelist_path, 'r')
    lines = filelist.readlines()
    for l in lines:
        img_name, label = l.split(" ")
        img_path = os.path.join(source, "SSLAD-2D/labeled/core50_128x128", img_name)
        img = open_image(img_path)
        imgs.append(img)
        onehot_label = np.zeros(10, dtype=np.int64)
        onehot_label[int(label)] = 1
        labels.append(onehot_label.tolist())

    h5_file_path = os.path.join(dest, "train_task{}_imgs_core.hdf5".format(experience))
    hf = h5py.File(h5_file_path, 'w')
    hf.create_dataset('images', data=np.asarray(imgs))
    hf.close()

    json_labels_file_path = os.path.join(dest, "train_task{}_multi_hot_categories_core.json".format(experience))
    with open(json_labels_file_path, 'w') as json_file:
        json.dump(labels, json_file)

def create_hdf5_test(source, run, nr_exp, dest):
    filelist_path = os.path.join(source, "SSLAD-2D/labeled/batches_filelists/NI_inc_ub_cat/run{}/test_filelist.txt"
                                 .format(run))
    if not os.path.exists(filelist_path):
        FileNotFoundError("path {} not found".format(filelist_path))
    imgs = []
    labels = []
    filelist = open(filelist_path, 'r')
    lines = filelist.readlines()
    for l in lines:
        img_name, label = l.split(" ")
        img_path = os.path.join(source, "SSLAD-2D/labeled/core50_128x128", img_name)
        img = open_image(img_path)
        imgs.append(img)
        onehot_label = np.zeros(10, dtype=np.int64)
        onehot_label[int(label)] = 1
        labels.append(onehot_label.tolist())

    for experience in range(nr_exp):
        h5_file_path = os.path.join(dest, "test_task{}_imgs_core.hdf5".format(experience))
        hf = h5py.File(h5_file_path, 'w')
        hf.create_dataset('images', data=np.asarray(imgs))
        hf.close()

        json_labels_file_path = os.path.join(dest, "test_task{}_multi_hot_categories_core.json".format(experience))
        with open(json_labels_file_path, 'w') as json_file:
            json.dump(labels, json_file)

def create_dict(destination):
    categories = ["plug", "mobile phone", "scissors", "light bulb", "can", "glasses", "ball", "marker", "cup", "remote control"]
    dest_path = os.path.join(destination, "multi_hot_dict_core.json")
    with open(dest_path, 'w') as json_file:
        json.dump(categories, json_file)


if __name__ == "__main__":
    # for exp in range(8):
    #     print("create exp {}".format(exp))
    #     create_hdf5_train("/ssd1/datasets/core50_UB", 0, exp, "/home/ggraffieti/dev/PRS/dataset/data/core")
    # create_hdf5_test("/ssd1/datasets/core50_UB", 0, 8, "/home/ggraffieti/dev/PRS/dataset/data/core")
    create_dict("/home/ggraffieti/dev/PRS/dataset/data/core")
