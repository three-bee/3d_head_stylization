import torch
from torch import nn
import numpy as np
import shutil
import argparse
import os
import random
import torchvision
from tqdm import tqdm
from copy import deepcopy
import json

from torch_utils import misc
import dnnlib
import legacy 
from training.triplane import TriPlaneGenerator
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler

def fix_seeds(seed, device):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    g = torch.Generator(device=device)
    g.manual_seed(seed)

class Inference():
    def __init__(self, save_path, G_ckpt_path, stylized_G_ckpt_path, seed=0, device='cuda', **kwargs):
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.device = device

        fix_seeds(seed, device)

        self.G = self.set_generator(G_ckpt_path, device).requires_grad_(False).eval()
        self.G_frozen = deepcopy(self.G)
        self.G.load_state_dict(torch.load(stylized_G_ckpt_path), strict=True)

        self.cam_pivot = torch.tensor([0, 0, 0], device=device)
        self.cam_radius = self.G_frozen.rendering_kwargs.get("avg_camera_radius", 2.7)
        self.intrinsics = FOV_to_intrinsics(18.837, device=device)
        visualize_yaw_pitch_list = [(-180 + 30*i, 10) for i in range(13)]
        visualize_yaw_pitch_list_front = [visualize_yaw_pitch_list[i] for i in [3,5,6,7,9]]
        self.front_pose_list_visualize = [self.get_pose(intrinsics=self.intrinsics, cam_pivot=self.cam_pivot, yaw=y*np.pi/180, pitch=p*np.pi/180) for y,p in visualize_yaw_pitch_list_front]

    @staticmethod
    def set_generator(ckpt_path, device):
        with dnnlib.util.open_url(ckpt_path) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device)
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        del G
        return G_new

    @staticmethod
    def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=[-0.35,0.35], pitch_range=[-0.15,0.15], cam_radius=2.7, device='cuda'):
        if yaw is None:
            yaw = np.random.uniform(yaw_range[0], yaw_range[1])
        if pitch is None:
            pitch = np.random.uniform(pitch_range[0], pitch_range[1])
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
        return c

    @torch.no_grad()
    def infer(self, latent_list=[], synth_sample_num=10): 
        ws_list = []
        if latent_list == []:
            for _ in range(synth_sample_num):
                z = torch.from_numpy(np.random.randn(bs, self.G_frozen.z_dim)).to(self.device)
                ws = self.G_frozen.backbone.mapping(z, self.conditioning_camera_params.repeat(bs,1), truncation_psi=0.75, truncation_cutoff=14)
                ws_list.append(ws)
        else:
            ws_list = [torch.from_numpy(np.load(ws_np)['w']).to(self.device) for ws_np in latent_list]

        for idx, ws in tqdm(enumerate(ws_list)):
            p_list, p_list_frozen = [], []
            for pose in self.front_pose_list_visualize:
                img = self.G.synthesis(ws, pose, forward_full=True, generate_background=False, make_background_white=True)["image"]
                img_frozen = self.G_frozen.synthesis(ws, pose, forward_full=True, generate_background=False, make_background_white=True)["image"]
                p_list.append(img)
                p_list_frozen.append(img_frozen)
            p = torch.cat([torch.cat(p_list, dim=-1), torch.cat(p_list_frozen, dim=-1)], dim=-2)
            torchvision.utils.save_image(p, os.path.join(self.save_path, f'eval_mv_{str(idx).zfill(4)}.jpg'), normalize=True, value_range=(-1, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default="work_dirs/demo")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument('--latent_list_path', type=str, default="example")
    parser.add_argument('--synth_sample_num', type=int, default=10)
    parser.add_argument('--G_ckpt_path', type=str, default="")
    parser.add_argument('--stylized_G_ckpt_path', type=str, default="")
    
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))
    os.makedirs(args.save_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.save_path, os.path.basename(__file__)))
    with open(os.path.join(args.save_path, "args_log_eval.txt"), "w") as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

    try:
        latent_list = [os.path.join(args.latent_list_path,i) for i in os.listdir(args.latent_list_path) if i.endswith('npz')]
    except:
        latent_list = []
        print(f'There is no available folder to fetch W+ latents. Generating {args.synth_sample_num} synth samples.')
    
    inference = Inference(**vars(args))
    inference.infer(latent_list=latent_list, synth_sample_num=args.synth_sample_num)