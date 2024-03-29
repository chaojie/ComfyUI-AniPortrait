import argparse
import os
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
node_path = folder_paths.get_folder_paths("custom_nodes")[0]

import ffmpeg
from datetime import datetime
from pathlib import Path
from typing import List
import subprocess
import av
import numpy as np
import cv2
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from scipy.signal import savgol_filter

from .configs.prompts.test_cases import TestCasesDict
from .src.models.pose_guider import PoseGuider
from .src.models.unet_2d_condition import UNet2DConditionModel
from .src.models.unet_3d import UNet3DConditionModel
from .src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from .src.utils.util import get_fps, read_frames, save_videos_grid

from .src.audio_models.model import Audio2MeshModel
from .src.utils.audio_util import prepare_audio_feature
from .src.utils.mp_utils  import LMKExtractor
from .src.utils.draw_util import FaceMeshVisualizer
from .src.utils.pose_util import project_points
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


ani_path=f'{node_path}/ComfyUI-AniPortrait'
config_path=f'{ani_path}/configs/prompts/animation_audio.yaml'
inference_config_path=f'{ani_path}/configs/inference/inference_v2.yaml'
audio_inference_config_path=f'{ani_path}/configs/inference/inference_audio.yaml'
audio_path=f'{ani_path}/configs/inference/audio/lyl.wav'
ref_video_path=f'{ani_path}/configs/inference/head_pose_temp/pose_ref_video.mp4'
#print(f'{ani_path}{config_path}{inference_config_path}{audio_inference_config_path}{audio_path}{ref_video_path}')
class AniPortraitLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/stable-diffusion-v1-5"}),
                "vae_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/sd-vae-ft-mse"}),
                "image_encoder_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/sd-image-variations-diffusers/image_encoder"}),
                "wav2vec2_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/facebook/wav2vec2-base-960h"}),
                "a2m_ckpt": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/ZJYang/AniPortrait/audio2mesh.pt"}),
                "motion_module_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/ZJYang/AniPortrait/motion_module.pth"}),
                "denoising_unet_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/ZJYang/AniPortrait/denoising_unet.pth"}),
                "reference_unet_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/ZJYang/AniPortrait/reference_unet.pth"}),
                "pose_guider_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/ZJYang/AniPortrait/pose_guider.pth"}),
                "weight_dtype": (["fp16","fp32"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("Pose2VideoPipeline","Audio2MeshModel",)
    RETURN_NAMES = ("pipe","a2m_model",)
    FUNCTION = "run"
    CATEGORY = "AniPortrait"

    def run(self,sd_path,vae_path,image_encoder_path,wav2vec2_path,a2m_ckpt,motion_module_path,denoising_unet_path,reference_unet_path,pose_guider_path,weight_dtype):
        #print(f'{ani_path}{config_path}{inference_config_path}{audio_inference_config_path}{audio_path}{ref_video_path}')
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=config_path)
        parser.add_argument("-W", type=int, default=512)
        parser.add_argument("-H", type=int, default=512)
        parser.add_argument("-L", type=int, default=16)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--cfg", type=float, default=3.5)
        parser.add_argument("--steps", type=int, default=25)
        parser.add_argument("--fps", type=int, default=30)
        #args = parser.parse_args()
        args, unknown = parser.parse_known_args()
        
        config = OmegaConf.load(args.config)

        OmegaConf.update(config, "pretrained_base_model_path", sd_path)
        OmegaConf.update(config, "pretrained_vae_path", vae_path)
        OmegaConf.update(config, "image_encoder_path", image_encoder_path)
        OmegaConf.update(config, "denoising_unet_path", denoising_unet_path)
        OmegaConf.update(config, "reference_unet_path", reference_unet_path)
        OmegaConf.update(config, "pose_guider_path", pose_guider_path)
        OmegaConf.update(config, "motion_module_path", motion_module_path)

        OmegaConf.update(config, "inference_config", inference_config_path)
        OmegaConf.update(config, "audio_inference_config", audio_inference_config_path)

        OmegaConf.update(config, "weight_dtype", weight_dtype)

        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
            
        audio_infer_config = OmegaConf.load(config.audio_inference_config)
        OmegaConf.update(audio_infer_config, "a2m_model.model_path", wav2vec2_path)
        OmegaConf.update(audio_infer_config, "a2p_model.model_path", wav2vec2_path)
        OmegaConf.update(audio_infer_config, "pretrained_model.a2m_ckpt", a2m_ckpt)

        # prepare model
        a2m_model = Audio2MeshModel(audio_infer_config['a2m_model'])
        a2m_model.load_state_dict(torch.load(audio_infer_config['pretrained_model']['a2m_ckpt']), strict=False)
        a2m_model.cuda().eval()

        vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to("cuda", dtype=weight_dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        #inference_config_path = config.inference_config
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")


        pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(device="cuda", dtype=weight_dtype) # not use cross attention

        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            config.image_encoder_path
        ).to(dtype=weight_dtype, device="cuda")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        # load pretrained weights
        denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )
        pose_guider.load_state_dict(
            torch.load(config.pose_guider_path, map_location="cpu"),
        )

        pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to("cuda", dtype=weight_dtype)

        return (pipe,a2m_model,)

def matrix_to_euler_and_translation(matrix):
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles, translation_vector


def smooth_pose_seq(pose_seq, window_size=5):
    smoothed_pose_seq = np.zeros_like(pose_seq)

    for i in range(len(pose_seq)):
        start = max(0, i - window_size // 2)
        end = min(len(pose_seq), i + window_size // 2 + 1)
        smoothed_pose_seq[i] = np.mean(pose_seq[start:end], axis=0)

    return smoothed_pose_seq

class AniPortraitRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("Pose2VideoPipeline",),
                "wav2vec2_path": ("STRING", {"default": "/home/admin/ComfyUI/models/diffusers/facebook/wav2vec2-base-960h"}),
                "a2m_model": ("Audio2MeshModel",),
                "image": ("IMAGE",),
                "pose": ("IMAGE",),
                "audio_path": ("STRING",{"default":audio_path}),
                "width": ("INT",{"default":512}),
                "height": ("INT",{"default":512}),
                "video_length": ("INT",{"default":16}),
                "steps": ("INT",{"default":25}),
                "cfg": ("FLOAT",{"default":3.5}),
                "seed": ("INT",{"default":1234}),
                "weight_dtype": (["fp16","fp32"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "AniPortrait"

    def run(self,pipe,wav2vec2_path,a2m_model,image,pose,audio_path,width,height,video_length,steps,cfg,seed,weight_dtype):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=config_path)
        parser.add_argument("-W", type=int, default=512)
        parser.add_argument("-H", type=int, default=512)
        parser.add_argument("-L", type=int, default=16)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--cfg", type=float, default=3.5)
        parser.add_argument("--steps", type=int, default=25)
        parser.add_argument("--fps", type=int, default=30)
        args, unknown = parser.parse_known_args()
        
        generator = torch.manual_seed(args.seed)
        args.W=width
        args.H=height
        args.L=video_length
        args.seed=seed
        args.cfg=cfg
        args.steps=steps

        config = OmegaConf.load(args.config)

        OmegaConf.update(config, "inference_config", inference_config_path)
        OmegaConf.update(config, "audio_inference_config", audio_inference_config_path)

        OmegaConf.update(config, "weight_dtype", weight_dtype)

        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
            
        audio_infer_config = OmegaConf.load(config.audio_inference_config)
        OmegaConf.update(audio_infer_config, "a2m_model.model_path", wav2vec2_path)
        OmegaConf.update(audio_infer_config, "a2p_model.model_path", wav2vec2_path)
        
        ref_image = 255.0 * image[0].cpu().numpy()
        ref_image_pil = Image.fromarray(np.clip(ref_image, 0, 255).astype(np.uint8))
        ref_image_w, ref_image_h = ref_image_pil.size
        
        config = OmegaConf.load(args.config)
        
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

        #save_dir = Path(f"output/{date_str}/{save_dir_name}")
        #save_dir.mkdir(exist_ok=True, parents=True)


        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer(forehead_edge=False)
        
        #ref_name = Path(ref_image_path).stem
        #audio_name = Path(audio_path).stem

        #ref_image_pil = Image.open(ref_image_path).convert("RGB")
        ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
        ref_image_np = cv2.resize(ref_image_np, (args.H, args.W))
        
        face_result = lmk_extractor(ref_image_np)
        assert face_result is not None, "No face detected."
        lmks = face_result['lmks'].astype(np.float32)


        ref_pose = vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)
        
        sample = prepare_audio_feature(audio_path, wav2vec_model_path=audio_infer_config['a2m_model']['model_path'])
        sample['audio_feature'] = torch.from_numpy(sample['audio_feature']).float().cuda()
        sample['audio_feature'] = sample['audio_feature'].unsqueeze(0)

        # inference
        pred = a2m_model.infer(sample['audio_feature'], sample['seq_len'])
        pred = pred.squeeze().detach().cpu().numpy()
        pred = pred.reshape(pred.shape[0], -1, 3)
        pred = pred + face_result['lmks3d']
        
        trans_mat_list = []
        for frame in pose:
            frame = 255.0 * frame.cpu().numpy()
            frame = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (args.H, args.W))
            result = lmk_extractor(frame)
            if result is not None and result['trans_mat'] is not None:
                trans_mat_list.append(result['trans_mat'].astype(np.float32))
            else:
                trans_mat_list.append(trans_mat_list[-1])
        trans_mat_arr = np.array(trans_mat_list)

        # compute delta pose
        trans_mat_inv_frame_0 = np.linalg.inv(trans_mat_arr[0])
        pose_arr = np.zeros([trans_mat_arr.shape[0], 6])

        for i in range(pose_arr.shape[0]):
            pose_mat = trans_mat_inv_frame_0 @ trans_mat_arr[i]
            euler_angles, translation_vector = matrix_to_euler_and_translation(pose_mat)
            pose_arr[i, :3] =  euler_angles
            pose_arr[i, 3:6] =  translation_vector

        total_frames=pose.shape[0]
        fps=30
        # interpolate to 30 fps
        new_fps = 30
        old_time = np.linspace(0, total_frames / fps, total_frames)
        new_time = np.linspace(0, total_frames / fps, int(total_frames * new_fps / fps))

        pose_arr_interp = np.zeros((len(new_time), 6))
        for i in range(6):
            interp_func = interp1d(old_time, pose_arr[:, i])
            pose_arr_interp[:, i] = interp_func(new_time)

        pose_seq = smooth_pose_seq(pose_arr_interp)
        #pose_seq = np.load(config['pose_temp'])
        mirrored_pose_seq = np.concatenate((pose_seq, pose_seq[-2:0:-1]), axis=0)
        cycled_pose_seq = np.tile(mirrored_pose_seq, (sample['seq_len'] // len(mirrored_pose_seq) + 1, 1))[:sample['seq_len']]

        # project 3D mesh to 2D landmark
        projected_vertices = project_points(pred, face_result['trans_mat'], cycled_pose_seq, [height, width])

        pose_images = []
        for i, verts in enumerate(projected_vertices):
            lmk_img = vis.draw_landmarks((width, height), verts, normed=False)
            pose_images.append(lmk_img)

        pose_list = []
        pose_tensor_list = []
        print(f"pose video has {len(pose_images)} frames, with {args.fps} fps")
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        for pose_image_np in pose_images[: args.L]:
            pose_image_pil = Image.fromarray(cv2.cvtColor(pose_image_np, cv2.COLOR_BGR2RGB))
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_image_np = cv2.resize(pose_image_np,  (width, height))
            pose_list.append(pose_image_np)
        
        pose_list = np.array(pose_list)
        
        video_length = len(pose_tensor_list)

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
            0
        )  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video_length
        )

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        video = pipe(
            ref_image_pil,
            pose_list,
            ref_pose,
            width,
            height,
            video_length,
            args.steps,
            args.cfg,
            generator=generator,
        ).videos

        '''
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
        save_path = f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_noaudio.mp4"
        save_videos_grid(
            video,
            save_path,
            n_rows=3,
            fps=args.fps,
        )
        
        stream = ffmpeg.input(save_path)
        audio = ffmpeg.input(audio_path)
        ffmpeg.output(stream.video, audio.audio, save_path.replace('_noaudio.mp4', '.mp4'), vcodec='copy', acodec='aac').run()
        os.remove(save_path)
        '''

        print(f'{video.shape}')
        video=video.permute(0, 2, 3, 4, 1)
        print(f'{video.shape}')
        
        return video

class MaskList2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT",{"defualt":10}),
            },
        }

    RETURN_TYPES = ("IMAGE","BOX",)
    FUNCTION = "run"
    CATEGORY = "AniPortrait"
    OUTPUT_NODE = True

    def run(self, image, mask, padding):
        from torchvision.ops import masks_to_boxes
        boxes = masks_to_boxes(mask)
        print(f'{boxes}')
        box=[int(torch.min(boxes,dim=0).values[0]),int(torch.max(boxes,dim=0).values[1]),int(torch.min(boxes,dim=0).values[2]),int(torch.max(boxes,dim=0).values[3])]
        
        if box[0]-padding>0:
            box[0]=box[0]-padding
        else:
            box[0]=0
        if box[1]-padding>0:
            box[1]=box[1]-padding
        else:
            box[1]=0
        if box[2]+padding<image.shape[2]:
            box[2]=box[2]+padding
        else:
            box[2]=image.shape[2]
        if box[3]+padding<image.shape[1]:
            box[3]=box[3]+padding
        else:
            box[3]=image.shape[1]
        return (image[:,box[1]:box[3],box[0]:box[2],:],box,)

class Box2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "box": ("BOX",),
            },
        }

    RETURN_TYPES = ("IMAGE","BOX",)
    FUNCTION = "run"
    CATEGORY = "AniPortrait"
    OUTPUT_NODE = True

    def run(self, image, box):
        return (image[:,box[1]:box[3],box[0]:box[2],:],box,)

class CoverVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bgimage": ("IMAGE",),
                "coverimage": ("IMAGE",),
                "box": ("BOX",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "AniPortrait"
    OUTPUT_NODE = True

    def run(self, bgimage, coverimage, box):
        bgimage[:,box[1]:box[1]+coverimage.shape[1],box[0]:+box[0]+coverimage.shape[2],:]=coverimage
        return (bgimage,)

NODE_CLASS_MAPPINGS = {
    "AniPortraitLoader":AniPortraitLoader,
    "AniPortraitRun":AniPortraitRun,
    "MaskList2Video":MaskList2Video,
    "Box2Video":Box2Video,
    "CoverVideo":CoverVideo,
}