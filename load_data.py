import os
import pickle
import json
import math
from scipy.io import savemat
import random
from models.ImageEncoder import ImageEncoder
import torch
import torch.nn.functional as F

def load_data(target, viewpoint_num, use_saved_selection=False):
    dirpath = "./data/{}".format(target)
    pickle_loc = "{}/Data".format(dirpath)
    # output_loc = "{}/UnzipData".format(dirpath)
    cfg_path = "{}/Config.json".format(dirpath)

    # Read Config
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

    # if not os.path.exists(output_loc):
    #     os.makedirs(output_loc)

    # Load data from Data/*.pkl
    images = []
    sensor_poses = []

    files = os.listdir(pickle_loc)  # 获取文件列表

    selected_files_path = os.path.join(dirpath, f"selected_files_{viewpoint_num}.txt")
    selected_files = []

    if use_saved_selection and os.path.exists(selected_files_path):
        # 从文件中读取选中的文件名
        with open(selected_files_path, 'r') as f:
            selected_files = f.read().splitlines()
    else:
        # 如果viewpoint_num大于文件数量，则使用全部文件
        if viewpoint_num > len(files):
            viewpoint_num = len(files)

        # 随机选择viewpoint_num个文件
        selected_files = random.sample(files, viewpoint_num)

        # 保存选中的文件名到一个文本文件
        with open(selected_files_path, 'w') as f:
            for file in selected_files:
                f.write(f"{file}\n")

    for pkls in selected_files:
        filename = "{}/{}".format(pickle_loc, pkls)
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            image = state["ImagingSonar"]
            s = image.shape
            image[image < 0.2] = 0          # threshold
            image[s[0] - 200:, :] = 0
            pose = state["PoseSensor"]
            images.append(image)
            sensor_poses.append(pose)

    # Convert images to tensor and permute to match the encoder input shape
    device = "cuda"
    images_tensor = torch.tensor(images, device=device).unsqueeze(1)    # images变为 (B, 1, H, W)

    print("images_tensor.shape:", images_tensor.shape)
    # Initialize the encoder and extract features
    encoder = ImageEncoder().to(device).eval()
    with torch.no_grad():
        images_reference_feature = encoder(images_tensor)
    # print("images_reference_feature.shape:", images_reference_feature.shape)  # (B, 512, H/2, W/2)

    data = {
        "images": images,   # B,C,H,W
        "images_no_noise": [],
        "sensor_poses": sensor_poses,
        "min_range": min_range,
        "max_range": max_range,
        "hfov": hfov,
        "vfov": vfov,
        "images_reference_feature": images_reference_feature
    }

    # savemat('{}/{}.mat'.format(dirpath, target), data, oned_as='row')   # too large
    return data


class SonarReferenceDataset:
    def __init__(self, reference, c2w, phi_min, phi_max, r_min, r_max, H, W):
        self.reference = reference
        self.c2w = torch.tensor(c2w, device=reference.device) if isinstance(c2w, list) else c2w
        self.n = self.c2w.shape[0]  # 假设 n 是相机数量
        self.R_t = self.c2w[:, :3, :3].permute(0, 2, 1)  # (n, 3, 3)
        self.camera_pos = self.c2w[:, :3, -1]  # (n, 3)
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.r_min = r_min
        self.r_max = r_max
        self.H, self.W = H, W

    @torch.no_grad()
    def feature_matching(self, pts_r_rand, n_selected_px, batch_size=10):
        # pts_r_rand shape: (n_selected_px * arc_n_samples * ray_n_samples, 3)
        n_points = pts_r_rand.shape[0]
        n_samples = n_points // n_selected_px
        pts_r_rand = pts_r_rand.reshape(n_selected_px, n_samples, 3)    # (n_selected_px, arc_n_samples * ray_n_samples, 3)


        # Expand pts_r_rand for each camera
        pos = pts_r_rand.unsqueeze(0).expand(self.n, n_selected_px, n_samples, 3)   # (n, n_selected_px, arc_n_samples * ray_n_samples, 3)

        # Calculate relative position to camera
        camera_pos = self.camera_pos[:, None, None, :].expand_as(pos)
        ref_pos = torch.einsum("kij,kbsj->kbsi", self.R_t, pos - camera_pos)  # (n, n_selected_px, arc_n_samples * ray_n_samples, 3)

        # Project to 2D image plane
        uv_pos = ref_pos[..., :-1] / ref_pos[..., -1:]  # (n, n_selected_px, arc_n_samples * ray_n_samples, 2)
        uv_pos[..., 1] *= -1.0  # Flip y-axis for image coordinates

        # Normalize coordinates to [-1, 1] range for grid_sample
        uv_pos[..., 0] = (uv_pos[..., 0] / (self.W / 2)) * 2 - 1  # Normalize width (W)
        uv_pos[..., 1] = (uv_pos[..., 1] / (self.H / 2)) * 2 - 1  # Normalize height (H)

        # Sample the feature maps
        sampled_features = F.grid_sample(self.reference, uv_pos, align_corners=True,
                                         padding_mode="border")
        print("sampled_features.shape: ", sampled_features.shape)   # (n, 512, n_selected_px, arc_n_samples * ray_n_samples)

        return sampled_features