import sys

sys.path.append(r"/home/dell/桌面/drh/CodeFormer/basicsr")
import torch
import numpy as np
import torch.utils.data as data
from utils import get_root_logger
from data.data_util import paths_from_folder
from utils import FileClient, imfrombytes, img2tensor
import os.path as osp
from data.transforms import augment
from torchvision.transforms.functional import normalize
from PIL import Image
import torchvision.transforms.functional as TF
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class HighQualityDataset(data.Dataset):
    def __init__(self, opt):
        super(HighQualityDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # io_backend

        self.gt_folder = opt['dataroot_gt']
        self.gt_size = opt.get('gt_size', 256)
        self.in_size = opt.get('in_size', 256)
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        if self.io_backend_opt['type'] == 'lmdb':  # load data from device
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.latent_gt_path = opt.get('latent_gt_path', None)  # for fine-tune
        if self.latent_gt_path is not None:
            self.load_latent_gt = True
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False

    def crop_image(self, image_tensor, crop_size=(256, 256)):
        C, H, W = image_tensor.shape
        crop_height, crop_width = crop_size

        if H < crop_height or W < crop_width:
            raise ValueError("Image is smaller than crop size")

        top = np.random.randint(0, H - crop_height)
        left = np.random.randint(0, W - crop_width)

        cropped_tensor = image_tensor[:, top:top + crop_height, left:left + crop_width]
        return cropped_tensor

    def __getitem__(self, index):
        # io_backend
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # load img and path
        gt_path = self.paths[index]
        name = osp.basename(gt_path)[:-4]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hfilp'], rotation=False, return_status=True)

        # pretrain codebook
        if self.load_latent_gt:
            if status[0]:
                latent_gt = self.latent_gt_dict['hflip'][name]
            else:
                latent_gt = self.latent_gt_dict['orig'][name]

        img_in = img_gt
        img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=True, float32=True)
        img_in, img_gt = self.crop_image(img_in), self.crop_image(img_gt)

        # round and clip
        img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_in, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # final return
        return_dict = {'in': img_in, 'gt': img_gt, 'gt_path': gt_path}
        if self.load_latent_gt:
            return_dict['latent_gt'] = latent_gt
        return return_dict

    def __len__(self):
        return len(self.paths)
