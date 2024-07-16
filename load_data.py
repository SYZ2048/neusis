import os
import cv2
import pickle
import json
import math
from scipy.io import savemat
import random


def load_data(target, viewpoint_num=300):
    dirpath = "./data/{}".format(target)
    pickle_loc = "{}/Data".format(dirpath)
    output_loc = "{}/UnzipData".format(dirpath)
    cfg_path = "{}/Config.json".format(dirpath)

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    for agents in cfg["agents"][0]["sensors"]:
        if agents["sensor_type"] != "ImagingSonar": continue
        hfov = agents["configuration"]["Azimuth"]
        vfov = agents["configuration"]["Elevation"]
        min_range = agents["configuration"]["RangeMin"]
        max_range = agents["configuration"]["RangeMax"]
        hfov = math.radians(hfov)
        vfov = math.radians(vfov)

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    images = []
    sensor_poses = []

    # 获取文件列表
    files = os.listdir(pickle_loc)

    # 如果viewpoint_num大于文件数量，则使用全部文件
    if viewpoint_num > len(files):
        viewpoint_num = len(files)

    # 随机选择viewpoint_num个文件
    selected_files = random.sample(files, viewpoint_num)

    # 保存选中的文件名到一个文本文件
    selected_files_path = os.path.join(dirpath, "selected_files.txt")
    with open(selected_files_path, 'w') as f:
        for file in selected_files:
            f.write(f"{file}\n")

    # for pkls in os.listdir(pickle_loc):
    for pkls in selected_files:
        filename = "{}/{}".format(pickle_loc, pkls)
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            image = state["ImagingSonar"]
            s = image.shape
            image[image < 0.2] = 0
            image[s[0] - 200:, :] = 0
            pose = state["PoseSensor"]
            images.append(image)
            sensor_poses.append(pose)

    data = {
        "images": images,
        "images_no_noise": [],
        "sensor_poses": sensor_poses,
        "min_range": min_range,
        "max_range": max_range,
        "hfov": hfov,
        "vfov": vfov
    }

    savemat('{}/{}.mat'.format(dirpath, target), data, oned_as='row')
    return data
