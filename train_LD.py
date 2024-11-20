import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import shutil
import types
import argparse
import os
import random
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms import Resize as TResize
from torchvision.transforms import Normalize as TNormalize
from tqdm import tqdm
from copy import deepcopy
import json
import cv2
import PIL.Image
import lpips

from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torch_utils import misc
import dnnlib
import legacy 
from training.triplane import TriPlaneGenerator
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
   
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

class Coach():
    def __init__(self, diff_ckpt_path, G_ckpt_path, controlnet_edge_path, controlnet_depth_path, depth_path,
                 save_path='demo', lr=1e-4, seed=0, device='cuda',
                 **kwargs) -> None:
        self.save_path = save_path
        self.device = device
        self.lr = lr

        fix_seeds(seed, device)

        ## Networks
        self.G = self.set_generator(G_ckpt_path, device).requires_grad_(True).train()
        self.G_frozen = deepcopy(self.G).requires_grad_(False).eval()
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device).eval()

        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=self.lr)

        self.num_inference_steps = 50
        controlnet_edge = ControlNetModel.from_pretrained(controlnet_edge_path, torch_dtype=torch.float16).to(device)
        self.P_single = StableDiffusionControlNetPipeline.from_pretrained(diff_ckpt_path, controlnet=controlnet_edge, safety_checker=None, torch_dtype=torch.float16).to(device)
        self.P_single.scheduler = DDIMScheduler.from_config(self.P_single.scheduler.config)
        controlnet_depth = ControlNetModel.from_pretrained(controlnet_depth_path, torch_dtype=torch.float16).to(device)
        self.P_grid = StableDiffusionControlNetPipeline.from_pretrained(diff_ckpt_path, controlnet=controlnet_depth, safety_checker=None, torch_dtype=torch.float16).to(device)
        self.P_grid.scheduler = DDIMScheduler.from_config(self.P_grid.scheduler.config)

        ## Monkey-patch DepthAnythingV2
        self.D = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768]).to(self.device)
        self.D.load_state_dict(torch.load(depth_path, map_location='cpu'))
        self.D.eval()
        @torch.no_grad()
        def infer_image_torch(self, raw_image, input_size=518):
            transform = Compose(
                [
                    Resize(
                        width=input_size,
                        height=input_size,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )
            h, w = raw_image.shape[:2]
            image = raw_image
            image = transform({"image": image})["image"]
            image = torch.from_numpy(image).unsqueeze(0)
            DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            image = image.to(DEVICE)
            depth = self.forward(image)
            depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
            return depth
        self.D.infer_image_torch = types.MethodType(infer_image_torch, self.D)
        
        ## Rendering parameters
        self.cam_pivot = torch.tensor([0, 0, 0], device=device)
        self.cam_radius = self.G.rendering_kwargs.get("avg_camera_radius", 2.7)
        self.intrinsics = FOV_to_intrinsics(18.837, device=device)
        self.conditioning_camera_params = self.get_pose(self.cam_pivot, self.intrinsics, yaw=0, pitch=0.2, cam_radius=self.cam_radius, device=device)

        self.yaw_range_front = [-np.pi/3, np.pi/3]
        self.yaw_range_front_grid = [-np.pi, np.pi]
        self.pitch_range = [-np.pi/6, np.pi/6]

    def ldis_loss(self, Ti_prime, Tj_prime, Ti, Tj):
        """
        Loss proposed in https://arxiv.org/abs/2312.16837
        """
        transformed_distance = torch.norm(Ti_prime - Tj_prime, p=2, dim=1) ** 2 # ||Ti' - Tj'||
        original_distance = torch.norm(Ti - Tj, p=2, dim=1) ** 2 # ||Ti - Tj||
        original_distance = original_distance + 1e-8
        ldis = torch.abs((transformed_distance / original_distance) - 1)
        return ldis.mean()

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
    def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=[-0.35,0.35], pitch_range=[-0.15,0.15], cam_radius=2.7, device='cuda', return_yaw=False):
        if yaw is None:
            yaw = np.random.uniform(yaw_range[0], yaw_range[1])
        if pitch is None:
            pitch = np.random.uniform(pitch_range[0], pitch_range[1])
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
        if return_yaw:
            return c, yaw
        return c

    @staticmethod
    def resize_and_grid(tensors, resize=None):
        if resize is not None:
            resized = [F.interpolate(t, scale_factor=resize, mode='bilinear', align_corners=True) for t in tensors]
        else:
            resized = tensors
        return torch.cat([torch.cat(resized[:2], dim=-1), torch.cat(resized[2:], dim=-1)], dim=-2)

    def normalize(self, t):
        return (t - t.min()) / (t.max() - t.min() + 1e-6)

    def normalize_depth(self, depth, device):
        depth = torch.clip(depth, depth.min(), torch.tensor(2.85, device=device))
        normalized_depth = self.normalize(depth)
        normalized_depth = torch.abs(normalized_depth - 1)
        return normalized_depth

    @staticmethod
    def sample(curr_pipe, prompt,
               start_step=0, start_latents=None,
               guidance_scale=7.5, controlnet_scale=1.0, num_inference_steps=50,
               num_images_per_prompt=1, do_classifier_free_guidance=True,
               negative_prompt='', 
               controlnet_cond=None, 
               return_noise_pred=False, 
               max_num_inference_steps=None, 
               device='cuda'):
        text_embeddings = curr_pipe._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        curr_pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=device)
            start_latents *= curr_pipe.scheduler.init_noise_sigma

        latents = start_latents.clone()

        if max_num_inference_steps is None:
            max_num_inference_steps = num_inference_steps
            
        for i in (range(start_step, max_num_inference_steps)):
            t = curr_pipe.scheduler.timesteps[i]
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = curr_pipe.scheduler.scale_model_input(latent_model_input, t).to(torch.float16)

            if controlnet_cond is None:
                noise_pred = curr_pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            else:        
                controlnet_img = curr_pipe.prepare_image(
                    image=controlnet_cond,
                    width=curr_pipe.unet.config.sample_size * curr_pipe.vae_scale_factor,
                    height=curr_pipe.unet.config.sample_size * curr_pipe.vae_scale_factor,
                    batch_size=1,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=torch.float16,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=False,
                )

                down_block_res_samples, mid_block_res_sample = curr_pipe.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_img,
                    conditioning_scale=controlnet_scale,
                    guess_mode=False,
                    return_dict=False,
                )

                noise_pred = curr_pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
            alpha_t = curr_pipe.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = curr_pipe.scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
            direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
            latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

        if return_noise_pred:
            predicted_x0 = torch.from_numpy(curr_pipe.decode_latents(predicted_x0).transpose(0,3,1,2)).to(device).to(torch.float32)
            return guidance_scale * (noise_pred_text - noise_pred_uncond), predicted_x0, noise_pred
        
        images = curr_pipe.decode_latents(latents.to(torch.float16))
        images = curr_pipe.numpy_to_pil(images)
        return image_mask

    def low_rank_approximation(self, grad, k=4):
        """
        Apply low-rank approximation to the gradient tensor while keeping the batch dimension intact.
        Weigh the top-4 singular values with 100%, 75%, 50%, and 25%, respectively.
        """
        B, C, H, W = grad.shape
        grad_approx = torch.zeros_like(grad)

        for b in range(B):
            grad_flat = grad[b].view(C, H * W)

            grad_flat = grad_flat.to(torch.float32)
            try:
                U, S, V = torch.svd(grad_flat)
            except: # In case of ill-conditioned matrix 
                return grad

            k = min(4, S.size(0))
            weights = torch.tensor([1.0, 0.75, 0.5, 0.25], device=grad.device)[:k]
            S_k = S[:k] * weights
            U_k = U[:, :k]
            V_k = V[:, :k]

            grad_low_rank = torch.mm(U_k, torch.mm(torch.diag(S_k), V_k.t()))
            grad_approx[b] = grad_low_rank.view(C, H, W)

        grad_approx = grad_approx.to(grad.dtype)
        return grad_approx

    def score_distillation(self, pipe, img, img_frozen, img_mirror, in_prompt, start_step_range, guidance_scale=7.5, controlnet_scale=1.0,
                           controlnet_type='edge', external_controlnet_tensor=None, enable_mirror=True, 
                           mirror_weight=1.0, base_weight=1.0,
                           grad_mask=None, use_lowrank=False, lowrank_k=4,
                           grad_div_scale=1000,
                           use_SDS=False):
        """
        Performs score distillation and returns E[x_0|y]
        """
        latent = pipe.vae.encode(img.half())
        l = 0.18215 * latent.latent_dist.sample()
        if enable_mirror:
            latent_mirrored = pipe.vae.encode(img_mirror.half())
            l_mirrored = 0.18215 * latent_mirrored.latent_dist.sample()            

        if controlnet_type == 'edge':
            in_np = np.clip((img_frozen.detach().cpu().squeeze().numpy()+1)*127.5, 0, 255).astype(np.uint8)[0]
            in_np = cv2.Canny(in_np, 50, 250)[:, :, None]
            in_np = np.concatenate([in_np, in_np, in_np], axis=-1)
            controlnet_cond = PIL.Image.fromarray(in_np)
        elif controlnet_type == 'depth':
            in_np = external_controlnet_tensor.detach().cpu().squeeze().numpy()
            in_np = (in_np * 255).astype(np.uint8)
            controlnet_cond = PIL.Image.fromarray(in_np)
        else:
            controlnet_cond = None

        with torch.no_grad():
            start_step = random.randint(*start_step_range)
            pipe.scheduler.set_timesteps(self.num_inference_steps)
            noise = torch.randn_like(l, device=self.device)
            noisy_l = pipe.scheduler.add_noise(l, noise, pipe.scheduler.timesteps[start_step])

            score, x0hat, _ = self.sample(curr_pipe=pipe,
                    prompt=in_prompt, 
                    negative_prompt='',
                    start_latents=noisy_l, start_step=start_step, 
                    num_inference_steps=self.num_inference_steps,
                    max_num_inference_steps=start_step+1,
                    controlnet_cond=controlnet_cond,
                    return_noise_pred=True,
                    guidance_scale=guidance_scale,
                    controlnet_scale=controlnet_scale)

        if use_lowrank:
            score_rank1 = self.low_rank_approximation(score, k=random.choice([1,2,3,4]) if lowrank_k==-1 else lowrank_k)
            score = self.instance_norm(score, score_rank1)
        
        if use_SDS:
            grad = (score-noise) * torch.sqrt(1 - pipe.scheduler.alphas_cumprod[start_step])
        else: # likelihood distillation
            grad = score * torch.sqrt(pipe.scheduler.alphas_cumprod[start_step])
        
        if grad_mask is not None:
            grad *= grad_mask
        
        grad /= grad_div_scale
        grad = grad.clamp(-1,1)
        grad = torch.nan_to_num(grad, 0, 0, 0)
        
        loss_score = base_weight * SpecifyGradient.apply(l, grad)
        if mirror_weight>0.0 and enable_mirror:
            loss_score += mirror_weight * SpecifyGradient.apply(l_mirrored, torch.flip(grad,dims=[-1]))

        loss_score.backward(retain_graph=True)

        return x0hat
    
    @staticmethod
    def instance_norm(img_src, img_tgt):
        mean_a = img_src.mean(dim=(2, 3), keepdim=True)
        std_a = img_src.std(dim=(2, 3), keepdim=True) + 1e-5

        mean_b = img_tgt.mean(dim=(2, 3), keepdim=True)
        std_b = img_tgt.std(dim=(2, 3), keepdim=True) + 1e-5

        norm_tgt = (img_tgt - mean_b) / std_b
        result = norm_tgt * std_a + mean_a

        return result

    def determine_opt_layers(self, in_prompt, batch=1, iters=50, guidance_scale=7.5, topk_layers=5):
        """
        Performs distillation iters times to get the most affected W^+ layer.
        Returns layers to be updated, their indexes, and W^+ weights.
        """
        conditioning_camera_params = self.get_pose(self.cam_pivot, self.intrinsics, yaw_range=self.yaw_range_front, pitch_range=self.pitch_range, cam_radius=self.cam_radius, device=self.device)

        sample_z = torch.from_numpy(np.random.randn(batch, self.G.z_dim)).to(self.device)
        initial_w_codes = self.G_frozen.backbone.mapping(sample_z, conditioning_camera_params, truncation_psi=0.75, truncation_cutoff=14)
        w_codes = deepcopy(initial_w_codes).detach()
        
        w_codes.requires_grad = True
        w_optim = torch.optim.SGD([w_codes], lr=0.01)

        for _ in tqdm(range(iters)):
            camera_params = self.get_pose(self.cam_pivot, self.intrinsics, yaw_range=self.yaw_range_front, pitch_range=self.pitch_range, cam_radius=self.cam_radius, device=self.device)

            w_optim.zero_grad()

            generated_from_w = self.G.synthesis(w_codes, camera_params,
                forward_full=True,
                generate_background=False,
                return_triplanes=True)['image']
            generated_from_w_frozen = self.G_frozen.synthesis(w_codes, camera_params,
                forward_full=True,
                generate_background=False,
                return_triplanes=True)['image']
            
            self.score_distillation(pipe=self.P_single, 
                                    img=generated_from_w, img_frozen=generated_from_w_frozen, img_mirror=None,
                                    start_step_range=(35,49),
                                    in_prompt=in_prompt, controlnet_type=None, enable_mirror=False)
            w_optim.step()
        
        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        chosen_layer_idx = torch.topk(layer_weights, topk_layers)[1].cpu().numpy()
        idx_to_res_mapping = {0:4, 1:8, 2:8, 3:16, 4:16, 5:32, 6:32, 7:64, 8:64, 9:128, 10:128, 11:256, 12:256, 13:256}
        chosen_layers = [getattr(self.G.backbone.synthesis, f'b{idx_to_res_mapping[idx]}') for idx in chosen_layer_idx]

        return chosen_layers, chosen_layer_idx, layer_weights.detach().cpu().numpy()

    def prepare_4x4_grid(self, ws, backprop_grid_before_SR=False):
        l_img, l_img_GT, l_img_mirror, l_img_GT_depth, l_mask_GT = [], [], [], [], []

        for _ in range(4):
            c = self.get_pose(self.cam_pivot, self.intrinsics, yaw_range=self.yaw_range_front_grid, pitch_range=self.pitch_range, cam_radius=self.cam_radius, device=self.device)
            out_dict = self.G.synthesis(ws, c, forward_full=True, generate_background=False, return_triplanes=False)

            c_mirror = c.detach().clone()
            c_mirror[:, [1, 2, 3, 4, 8]] *= -1
            out_dict_mirror = self.G.synthesis(ws, c_mirror, forward_full=True, generate_background=False, return_triplanes=False)

            if backprop_grid_before_SR:
                l_img.append(out_dict['image_raw'])
                l_img_mirror.append(out_dict_mirror['image_raw'])
            else:
                l_img.append(out_dict['image'])
                l_img_mirror.append(out_dict_mirror['image'])

            with torch.no_grad():
                out_dict_frozen = self.G_frozen.synthesis(ws, c, forward_full=True, generate_background=False, return_triplanes=False)
        
                depth_frozen = self.D.infer_image_torch(raw_image=0.5*out_dict_frozen['image'].detach().cpu().numpy().squeeze().transpose(1,2,0) + 0.5)
                depth_frozen = self.normalize(depth_frozen).unsqueeze(0)                    

                l_img_GT.append(out_dict_frozen['image'])
                l_img_GT_depth.append(depth_frozen)
                l_mask_GT.append(out_dict_frozen['image_mask'])

        img_GT_prime = self.resize_and_grid(l_img_GT, resize=None)
        img_GT_prime_depth = self.resize_and_grid(l_img_GT_depth, resize=None)
        img_prime = self.resize_and_grid(l_img, resize=4 if backprop_grid_before_SR else 0.5)
        img_prime_mirror = self.resize_and_grid(l_img_mirror, resize=4 if backprop_grid_before_SR else 0.5)
        img_GT_mask = self.resize_and_grid(l_mask_GT, resize=0.5)

        return img_prime, img_GT_prime, img_prime_mirror, img_GT_prime_depth, img_GT_mask
    
    def get_batch(self, c, bs):
        c_mirror = c.detach().clone()
        c_mirror[:, [1, 2, 3, 4, 8]] *= -1

        z = torch.from_numpy(np.random.randn(bs, self.G.z_dim)).to(self.device)
        ws = self.G.backbone.mapping(z, self.conditioning_camera_params.repeat(bs,1), truncation_psi=0.75, truncation_cutoff=14)

        input_dict = self.G.synthesis(ws, c.repeat(bs,1), forward_full=True, generate_background=False, return_triplanes=True)
        input_image = input_dict['image']
        input_image_beforeSR = input_dict['image_raw']
        input_mask = input_dict['image_mask']
        input_depth = input_dict['image_depth']
        input_triplane = input_dict['triplanes']

        input_dict_mirror = self.G.synthesis(ws, c_mirror.repeat(bs,1), forward_full=True, generate_background=False, return_triplanes=False)
        input_image_mirror = input_dict_mirror['image']

        with torch.no_grad():
            input_dict_frozen = self.G_frozen.synthesis(ws, c.repeat(bs,1), forward_full=True, generate_background=False, return_triplanes=True)
            input_image_frozen = input_dict_frozen['image']
            input_image_beforeSR_frozen = input_dict_frozen['image_raw']
            input_triplane_frozen = input_dict_frozen['triplanes']
            input_depth_frozen = input_dict['image_depth']
            input_mask_frozen = input_dict['image_mask']

            input_dict_frozen_mirror = self.G_frozen.synthesis(ws, c_mirror.repeat(bs,1), forward_full=True, generate_background=False, return_triplanes=False)
            input_image_frozen_mirror = input_dict_frozen_mirror['image']

        return z, ws, input_image, input_triplane, input_depth, input_image_beforeSR, input_image_mirror, input_image_frozen, input_triplane_frozen, input_depth_frozen, input_image_beforeSR_frozen, input_image_frozen_mirror, input_mask_frozen

    def train(self, 
              prompt='Portrait of a werewolf',
              bs=1, total_it=10_000, 
              enable_ldis=True,
              enable_grad_masking=True,
              use_SDS=False,
              grad_div_scale=1000,
              guidance_scale=7.5,
              controlnet_scale=1.0,
                            
              it_enable_grid=0,
              backprop_grid_before_SR=True,
              enable_mirror_grid=False, 
              
              base_weight=0.75,
              mirror_weight=0.25,
              tweedie_base_weight=1.0,
              tweedie_mirror_weight=1.0,
              tweedie_grid_weight=0.0,
              lpips_base_weight=0.05,
              lpips_mirror_weight=0.05,

              use_lowrank=True,
              lowrank_k=4,

              base_start_step_range=(35,49),
              grid_start_step_range=(15,40),
              
              freq_log_ckpt=250,
              freq_log_imgs=100,
              **kwargs
              ):

        self.G.freeze_layers()
        requires_grad(self.G.backbone.synthesis, flag=True)
        requires_grad(self.G.superresolution, flag=True)
        self.G.freeze_bias()
        
        ## Get a single batch prior to training for LDIS loss
        c = self.get_pose(self.cam_pivot, self.intrinsics, yaw_range=self.yaw_range_front, pitch_range=self.pitch_range, cam_radius=self.cam_radius, device=self.device)
        z0, ws0, input_image0, input_triplane0, input_depth0, input_image_beforeSR0, input_image_mirror0, input_image_frozen0, input_triplane_frozen0, input_depth_frozen0, input_image_beforeSR_frozen0, input_image_frozen_mirror0, input_mask_frozen0 = self.get_batch(c, bs)

        for it in tqdm(range(total_it)):
            c = self.get_pose(self.cam_pivot, self.intrinsics, yaw_range=self.yaw_range_front, pitch_range=self.pitch_range, cam_radius=self.cam_radius, device=self.device)
            z, ws, input_image, input_triplane, input_depth, input_image_beforeSR, input_image_mirror, input_image_frozen, input_triplane_frozen, input_depth_frozen, input_image_beforeSR_frozen, input_image_frozen_mirror, input_mask_frozen = self.get_batch(c, bs)

            ## Single view distillation
            loss_single = {}
            x0hat = self.score_distillation(pipe=self.P_single, 
                                            img=input_image, img_frozen=input_image_frozen, img_mirror=input_image_mirror,
                                            start_step_range=base_start_step_range,
                                            in_prompt=prompt, 
                                            controlnet_type="edge", 
                                            base_weight=base_weight,
                                            mirror_weight=mirror_weight,
                                            grad_mask=input_mask_frozen if enable_grad_masking else None,
                                            use_lowrank=use_lowrank,
                                            lowrank_k=lowrank_k,
                                            use_SDS=use_SDS,
                                            grad_div_scale=grad_div_scale,
                                            guidance_scale=guidance_scale,
                                            controlnet_scale=controlnet_scale,
                                            )
            if tweedie_base_weight>0.0:
                loss_single['loss_E0hat'] = tweedie_base_weight * torch.square(x0hat - 0.5*(input_image+1)).mean()  
            if tweedie_mirror_weight>0.0:
                loss_single['loss_E0hat_mirror'] = tweedie_mirror_weight * torch.square(torch.flip(x0hat,dims=[-1]) - 0.5*(input_image_mirror+1)).mean() 
            if enable_ldis:
                loss_single['ldis_loss'] = self.ldis_loss(input_triplane0, input_triplane, input_triplane_frozen0, input_triplane_frozen)
            if lpips_base_weight>0.0:
                loss_single['loss_E0hat_lpips'] = lpips_base_weight * self.lpips_loss_fn(2*(x0hat-0.5), input_image).mean()
            if lpips_mirror_weight>0.0:
                loss_single['loss_E0hat_lpips_mirror'] = lpips_mirror_weight * self.lpips_loss_fn(2*(torch.flip(x0hat,dims=[-1])-0.5), input_image_mirror).mean()
            
            ## Multiview distillation
            loss_grid = {}
            if it >= it_enable_grid:
                input_image_prime, input_image_prime_frozen, input_image_prime_mirror, input_image_prime_frozen_depth, input_image_prime_frozen_mask = self.prepare_4x4_grid(ws, backprop_grid_before_SR=backprop_grid_before_SR)
                x0hat_grid = self.score_distillation(pipe=self.P_grid, 
                                                    img=input_image_prime, img_frozen=input_image_prime_frozen, img_mirror=input_image_prime_mirror,
                                                    start_step_range=grid_start_step_range,
                                                    in_prompt=prompt,
                                                    controlnet_type='depth', 
                                                    external_controlnet_tensor=input_image_prime_frozen_depth,
                                                    enable_mirror=False, #NOTE need to flip 4x4 gradient seperately, disabled for now
                                                    grad_mask=input_image_prime_frozen_mask if enable_grad_masking else None,
                                                    use_lowrank=False,
                                                    use_SDS=use_SDS,
                                                    grad_div_scale=grad_div_scale,
                                                    guidance_scale=guidance_scale,
                                                    controlnet_scale=controlnet_scale,
                                                    )
                if tweedie_grid_weight>0.0:
                    loss_grid['loss_E0hat_grid'] = tweedie_grid_weight * torch.square(x0hat_grid - 0.5*(input_image_prime+1)).mean()  

            total_loss = 0
            for loss_dict in [loss_single, loss_grid]:
                for loss_value in loss_dict.values():
                    total_loss += loss_value
            total_loss.backward()
            self.G_optim.step()
            self.G_optim.zero_grad()

            if it % freq_log_imgs == 0:
                with torch.no_grad():
                    torchvision.utils.save_image(torch.cat([input_image, input_image_frozen, input_image0, input_image_frozen0, 2*(x0hat-0.5)],dim=-1),
                                                f'{self.save_path}/sv_{str(it).zfill(4)}.jpg', normalize=True, value_range=(-1,1))
                    
                    if it >= it_enable_grid:
                        torchvision.utils.save_image(input_image_prime_frozen_depth, f"{self.save_path}/gridControl_{str(it).zfill(4)}.jpg", normalize=True, value_range=(0, 1))
                        torchvision.utils.save_image(x0hat_grid, f"{self.save_path}/gridTweedie_{str(it).zfill(4)}.jpg", normalize=True, value_range=(0, 1))

            if it % freq_log_ckpt == 0:
                torch.save(self.G.state_dict(), os.path.join(self.save_path, f'G_{str(it).zfill(4)}.pth'))

            ## Compute LDIS with previpous batch 
            input_image0 = input_image.clone().detach()
            input_image_frozen0 = input_image_frozen.clone().detach()
            input_triplane0 = input_triplane.clone().detach()
            input_triplane_frozen0 = input_triplane_frozen.clone().detach()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', type=str, default="")

    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--total_it", type=int, default=10_000)
    parser.add_argument("--enable_ldis", type=int, default=1)
    parser.add_argument("--enable_grad_masking", type=int, default=1)
    parser.add_argument("--use_SDS", type=int, default=0)
    parser.add_argument('--grad_div_scale', type=float, default=1000)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--controlnet_scale', type=float, default=1.0)

    parser.add_argument("--it_enable_grid", type=int, default=0)
    parser.add_argument("--backprop_grid_before_SR", type=int, default=1)

    parser.add_argument('--base_weight', type=float, default=0.75)
    parser.add_argument('--mirror_weight', type=float, default=0.25)
    parser.add_argument('--tweedie_base_weight', type=float, default=0.1)
    parser.add_argument('--tweedie_mirror_weight', type=float, default=0.0)
    parser.add_argument('--tweedie_grid_weight', type=float, default=0.0)
    parser.add_argument("--lpips_base_weight", type=float, default=0.05)
    parser.add_argument("--lpips_mirror_weight", type=float, default=0.0)

    parser.add_argument("--use_lowrank", type=int, default=1)
    parser.add_argument("--lowrank_k", type=int, default=4)

    parser.add_argument('--base_start_step_range', nargs='+', type=int, default=[35,49])
    parser.add_argument('--grid_start_step_range', nargs='+', type=int, default=[35,49])

    parser.add_argument("--freq_log_imgs", type=int, default=100)
    parser.add_argument("--freq_log_ckpt", type=int, default=250)

    parser.add_argument('--diff_ckpt_path', type=str, default="")
    parser.add_argument('--G_ckpt_path', type=str, default="")
    parser.add_argument('--controlnet_edge_path', type=str, default="")
    parser.add_argument('--controlnet_depth_path', type=str, default="")
    parser.add_argument('--depth_path', type=str, default="")

    parser.add_argument('--save_path', type=str, default="work_dirs/demo")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))
    os.makedirs(args.save_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(args.save_path, os.path.basename(__file__)))
    with open(os.path.join(args.save_path, "args_log_train.txt"), "w") as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

    coach = Coach(**vars(args))
    coach.train(**vars(args))