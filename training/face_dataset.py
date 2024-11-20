import os
import numpy as np
import PIL.Image
from PIL import Image, ImageOps
import json
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import random

class OurImageFolderDataset(Dataset):
    def __init__(self,
                 img_folder_path,
                 image_list_json_path, # Enables rebalancing and to use several images multiple times in an epoch
                 cam_json_path,
                 img_ext:str) -> None:
        super().__init__()
        """
        If image_list_json_path is provided, get image names from _image_list_json_path and img_folder_path and take intersection.
        If not provided, only utilize img_folder_path. From these, create _image_fnames.
        Read camera params from cam_json_path, and only save the camera params of images in _image_fnames.
        """
        self._img_folder_path = img_folder_path
        self._image_list_json_path = image_list_json_path
        self._cam_json_path = cam_json_path
        self._img_ext = img_ext

        PIL.Image.init()
        
        ## Get image names
        if self._image_list_json_path:
            with open(self._image_list_json_path, "r") as f:
                self._image_fnames = json.load(f)
            self._image_fnames_json = sorted(self._remove_ext(fname) for fname in self._image_fnames if self._get_ext(fname) in PIL.Image.EXTENSION)
            self._image_fnames_dir = sorted(self._remove_ext(fname) for fname in os.listdir(self._img_folder_path) if self._get_ext(fname) in PIL.Image.EXTENSION)
            self._image_fnames = list(set(self._image_fnames_json).intersection(set(self._image_fnames_dir)))
        else:
            self._image_fnames = sorted(self._remove_ext(fname) for fname in os.listdir(self._img_folder_path) if self._get_ext(fname) in PIL.Image.EXTENSION)
        
        ## Get camera parameters
        with open(self._cam_json_path, "r") as f:
            self._raw_labels = dict(json.load(f)["labels"])
        self._raw_labels = {self._remove_ext(k): v for k,v in self._raw_labels.items()}
        self._raw_labels = [self._raw_labels.get(fname) for fname in self._image_fnames]
        self._raw_labels = np.array(self._raw_labels)

    @staticmethod
    def _get_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    @staticmethod
    def _remove_ext(fname):
        return os.path.splitext(fname)[0]
    
    def _open_file(self, fname):
        return open(os.path.join(self._img_folder_path, fname+f'.{self._img_ext}'), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = np.array(PIL.Image.open(f).resize((512,512)))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image, fname

    @staticmethod
    def flip_yaw(pose_matrix):
        flipped = pose_matrix.copy()
        flipped[0, 1] *= -1
        flipped[0, 2] *= -1
        flipped[1, 0] *= -1
        flipped[2, 0] *= -1
        flipped[0, 3] *= -1
        return flipped
    
    def get_label(self, idx):
        label = self._raw_labels[idx]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def __getitem__(self, idx):
        image, fname = self._load_raw_image(idx)
        mirror_image = np.array(ImageOps.mirror(Image.fromarray(image.transpose(1, 2, 0)))).transpose(2, 0, 1)
        label = self.get_label(idx)

        pose, intrinsics = np.array(label[:16]).reshape(4,4), np.array(label[16:]).reshape(3, 3)
        flipped_pose = self.flip_yaw(pose)
        mirror_label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)])

        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8

        image = torch.from_numpy(image / 255.0)
        mirror_image = torch.from_numpy(mirror_image / 255.0)
        image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        mirror_image = F.normalize(mirror_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        mirror_image_resized = F.resize(mirror_image, (256, 256))
        image_resized = F.resize(image, (256, 256))

        return image_resized, image, torch.from_numpy(label), mirror_image_resized, mirror_image, torch.from_numpy(mirror_label), fname
    
    def __len__(self):
        return len(self._image_fnames)

class CombinedImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, ds1, ds2, prob):
        super().__init__()
        self.ds1 = ds1
        self.ds2 = ds2
        self.prob = prob

    def __len__(self):
        return len(self.ds1) + len(self.ds2)

    def __getitem__(self, idx):
        if random.random() < self.prob:
            return self.ds1[idx % len(self.ds1)]
        else:
            return self.ds2[idx % len(self.ds2)]