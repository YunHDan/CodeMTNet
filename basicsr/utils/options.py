import yaml
import time
from collections import OrderedDict
from os import path as osp
from utils.misc import get_time_str

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, root_path, is_train=True, trial=None):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r', encoding='utf-8') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # -----------
    # third stage
    # opt['train']['pixel_opt']['loss_weight'] = trial.suggest_categorical("pix_weight",
    #                                                                      [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0])
    # opt['train']['ssim_opt']['loss_weight'] = trial.suggest_categorical("ssim_weight",
    #                                                                      [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0])
    # opt['train']['perceptual_opt']['loss_weight'] = trial.suggest_categorical("percep_weight",
    #                                                                      [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0])
    # opt['train']['optim_g']['lr'] = trial.suggest_categorical("lr", [1e-3, 5e-4, 8e-4, 2e-4, 1e-4, 8e-5, 5e-5, 2e-5, 1e-5])
    #
    # opt['train']['scheduler']['eta_mins'] = trial.suggest_categorical("eta_mins",
    #                                                           [[0.000001], [0.00001], [0.000005], [0.00002], [0.00005], [0.0000005], [0.0000001]])
    # -----------
    # second stage
    # opt['network_g']['d_state'] = trial.suggest_categorical("d_state",
    #                                                         [8, 16, 32, 64])
    # opt['network_g']['mlp_ratio'] = trial.suggest_categorical("mlp_ratio",
    #                                                           [1.0, 2.0, 4.0, 8.0])
    # opt['network_g']['L'] = trial.suggest_categorical("L",
    #                                                   [2, 4, 8, 16, 32, 64])
    # opt['network_g']['hidden_list'] = trial.suggest_categorical("hidden_list",
    #                                                             [[256, 256, 256], [128, 128, 128], [512, 512, 512]])
    # opt['train']['entropy_loss_weight'] = trial.suggest_categorical("entropy_loss_weight",
    #                                                                 [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0])
    # opt['train']['feat_loss_weight'] = trial.suggest_categorical("feat_loss_weight",
    #                                                              [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0])
    # --------------------------
    # third stage2
    # opt['path']['pretrain_network_g'] = trial.suggest_categorical("path", [
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-2-128-0_1-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-2-128-0_2-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-2-128-0_2-0_5_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-2-128-2_0-1_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-4-256-5_0-5_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-8-128-0_2-1_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-8-512-5_0-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-16-128-1_0-1_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-1-32-512-0_1-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-2-2-128-10_0-2_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-2-4-256-0_1-4_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-4-8-256-0_1-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-8-4-256-10_0-4_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/8-8-32-128-0_1-4_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/16-1-16-128-0_2-1_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/16-2-4-128-0_1-0_5_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/16-2-8-128-0_1-0_5_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/16-4-8-128-0_2-2_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/16-4-8-128-2_0-0_1_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-1-2-128-5_0-1_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-1-8-128-4_0-0_1_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-2-32-256-0_5-10_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-4-8-512-4_0-1_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-4-16-128-0_1-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-4-32-128-0_1-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-4-32-128-0_2-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-4-32-128-2_0-0_2_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-4-32-512-0_1-10_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/32-8-64-512-4_0-5_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/64-1-4-128-10_0-5_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/64-1-64-256-0_5-4_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/64-2-8-128-1_0-1_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/64-4-64-512-0_2-5_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/64-8-16-128-4_0-10_0_my_stage2/models/net_g_latest.pth",
    #     "/home/dell/桌面/drh/retry_no_norm/experiments/64-8-64-256-4_0-0_1_my_stage2/models/net_g_latest.pth"
    # ])
    # params_all = opt['path']['pretrain_network_g'].split("/")[-3]
    # opt['network_g']['d_state'] = int(params_all.split('-')[0])
    # opt['network_g']['mlp_ratio'] = float(params_all.split('-')[1])
    # opt['network_g']['L'] = int(params_all.split('-')[2])
    # opt['network_g']['hidden_list'] = [int(params_all.split('-')[3])] * 3
    # -------------------------------
    # # thrid stage3
    # opt['manual_seed'] = trial.suggest_categorical("seed", [13, 3407, 42, 100, 1, 14])

    if opt['path'].get('resume_state', None): # Shangchen added
        resume_state_path = opt['path'].get('resume_state')
        opt['name'] = resume_state_path.split("/")[-3]
    else:
        opt['name'] = f"{get_time_str()}_{opt['name']}"
    # -------------------------
    # if opt['name'].split("_")[-1] == 'stage3':
    #     opt['name'] = (
    #         f"{str(opt['train']['pixel_opt']['loss_weight']).replace('.', '_')}-{str(opt['train']['ssim_opt']['loss_weight']).replace('.', '_')}-"
    #         f"{str(opt['train']['perceptual_opt']['loss_weight']).replace('.', '_')}-{str(opt['train']['optim_g']['lr'])}-"
    #         f"{str(opt['train']['scheduler']['eta_mins'][0]).replace('.', '_')}_stage3"
    #     )
    # -------------------------
    # second stage
    # if opt['name'].split("_")[-1] == 'stage2':
    #     opt['name'] = (
    #         f"{opt['network_g']['d_state']}-{str(opt['network_g']['mlp_ratio']).split('.')[0]}-{opt['network_g']['L']}-{opt['network_g']['hidden_list'][0]}-"
    #         f"{str(opt['train']['entropy_loss_weight']).replace('.', '_')}-{str(opt['train']['feat_loss_weight']).replace('.', '_')}_my_stage2")
    # ---------------------------
    # third stage2
    # if opt['name'].split("_")[-1] == 'stage3':
    #     opt['name'] = (
    #         f"{opt['network_g']['d_state']}-{str(opt['network_g']['mlp_ratio']).split('.')[0]}-{opt['network_g']['L']}-{opt['network_g']['hidden_list'][0]}-"
    #         f"{str(opt['train']['entropy_loss_weight']).replace('.', '_')}-{str(opt['train']['feat_loss_weight']).replace('.', '_')}_my_stage3")
    # # ---------------------------

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

    else:  # test
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
