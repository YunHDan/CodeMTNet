import logging
import torch
from os import path as osp

from data import build_dataloader, build_dataset
from models import build_model
from train import parse_options
from utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from utils.options import dict2str


def main():
    # parse options, set distributed setting, set ramdom seed
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        model.validation(
            test_loader,

            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr,
            w=1
        ),

if __name__ == '__main__':
    main()
