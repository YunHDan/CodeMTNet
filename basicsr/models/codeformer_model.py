import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .sr_model import SRModel
from losses import msssim


@MODEL_REGISTRY.register()
class CodeFormerModel(SRModel):
    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.lq = data['lq'].to(self.device)
        self.b = self.gt.shape[0]

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if self.opt['path'].get('pretrain_network_hq', None) is not None:  # todo
            load_path = self.opt['path'].get('pretrain_network_hq')
            self.net_hq = build_network(self.opt['network_vqgan']).to(self.device)
            self.net_hq.eval()
            self.generate_idx_gt = False

            self.load_network(self.net_hq, load_path, self.opt['path']['strict_load_hq'])
            for param in self.net_hq.parameters():
                param.requires_grad = False
        elif self.opt.get('network_vqgan', None) is not None:
            self.hq_vqgan_fix = build_network(self.opt['network_vqgan']).to(self.device)
            self.hq_vqgan_fix.eval()
            self.generate_idx_gt = True

            for param in self.hq_vqgan_fix.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(f'Shoule have network_vqgan config or pre-calculated latent code.')

        logger.info(f'Need to generate latent GT code: {self.generate_idx_gt}')

        self.hq_feat_loss = train_opt.get('use_hq_feat_loss', True)
        self.feat_loss_weight = train_opt.get('feat_loss_weight', 1.0)
        self.cross_entropy_loss = train_opt.get('cross_entropy_loss', True)
        self.entropy_loss_weight = train_opt.get('entropy_loss_weight', 0.5)
        self.fidelity_weight = train_opt.get('fidelity_weight', 1.0)
        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8)

        self.net_g.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ssim_opt'):
            self.cri_ssim = msssim
            self.ssim_weight = train_opt['ssim_opt']['loss_weight']
            self.use_normalize = train_opt['ssim_opt']['normalize']
        else:
            self.cri_ssim = None
            self.ssim_weight = None
            self.use_normalize = False

        self.fix_generator = train_opt.get('fix_generator', True)
        logger.info(f'fix_generator: {self.fix_generator}')

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


    def optimize_parameters(self, current_iter):
        logger = get_root_logger()

        self.optimizer_g.zero_grad()

        if self.generate_idx_gt:
            md = self.hq_vqgan_fix
        else:
            md = self.net_hq

        x, _ = md.hq_encoder(self.gt)
        _, _, quant_stats = md.quantize(x)
        min_encoding_indices = quant_stats['min_encoding_indices']
        self.idx_gt = min_encoding_indices.view(self.b, -1)

        if self.fidelity_weight > 0:
            self.output, logits, lq_feat = self.net_g(self.lq, w=self.fidelity_weight, detach_16=True)
        else:
            logits, lq_feat = self.net_g(self.lq, w=0, code_only=True)

        if self.hq_feat_loss:
            # quant_feats
            if hasattr(self.net_g, 'module'):
                quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[self.b, 16, 16, 256])
            else:
                quant_feat_gt = self.net_g.quantize.get_codebook_feat(self.idx_gt, shape=[self.b, 16, 16, 256])

        loss_total = 0
        l_g_total = 0
        loss_dict = OrderedDict()
        if current_iter > self.net_g_start_iter:
            # hq_feat_loss
            if self.hq_feat_loss:  # codebook loss
                l_feat_encoder = torch.mean((quant_feat_gt.detach() - lq_feat) ** 2) * self.feat_loss_weight
                l_g_total += l_feat_encoder
                loss_dict['l_feat_encoder'] = l_feat_encoder

            # cross_entropy_loss
            if self.cross_entropy_loss:
                # b(hw)n -> bn(hw)
                cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
                l_g_total += cross_entropy_loss
                loss_dict['cross_entropy_loss'] = cross_entropy_loss

            if self.fidelity_weight > 0:  # when fidelity_weight == 0 don't need image-level loss
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep = self.cri_perceptual(self.output, self.gt)
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep

                if self.cri_ssim:
                    l_g_ssim = (1 - self.cri_ssim(self.output, self.gt, normalize=self.use_normalize)) * self.ssim_weight
                    l_g_total += l_g_ssim
                    loss_dict['l_g_ssim'] = l_g_ssim

            l_g_total.backward()
            loss_total += l_g_total
            self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self, w=None):
        # B, C, H, W = self.lq.shape
        # if (H * 10 / 16) % 10 < 5:
        #     height = H // 16
        # else:
        #     height = H // 16 + 1
        # if (W * 10 / 16) % 10 < 5:
        #     width = W // 16
        # else:
        #     width = W // 16 + 1
        # test_shape = (B, height, width, 256)
        B, C, H, W = self.lq.shape
        if (H * 10 / 16) % 10 == 0:
            height = H // 16
        else:
            height = H // 16 + 1
        if (W * 10 / 16) % 10 == 0:
            width = W // 16
        else:
            width = W // 16 + 1
        test_shape = (B, height, width, 256)
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                if w is not None:
                    self.output, _, _ = self.net_g_ema(self.lq, w=w, mode='test', test_shape=test_shape)
                else:
                    self.output, _, _ = self.net_g_ema(self.lq, w=self.fidelity_weight, mode='test', test_shape=test_shape)
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                if w is not None:
                    self.output, _, _ = self.net_g(self.lq, w=w, mode='test', test_shape=test_shape)
                else:
                    self.output, _, _ = self.net_g(self.lq, w=self.fidelity_weight, mode='test', test_shape=test_shape)
                self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr=True, w=None):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr=rgb2bgr, w=w)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, w=None, rgb2bgr=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name, extension = osp.splitext(osp.basename(val_data['lq_path'][0]))[0], osp.splitext(osp.basename(val_data['lq_path'][0]))[1]
            self.feed_data(val_data)
            self.test(w)

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(f'/home/dell/桌面/drh/JiShe/results',
                                             f'{img_name}_handled{extension}')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(f'/home/dell/桌面/drh/JiShe/results',
                                                 f'{img_name}_{self.opt["val"]["suffix"]}_handled{extension}')
                    else:
                        save_img_path = osp.join(f'/home/dell/桌面/drh/JiShe/results',
                                                 f'{img_name}_handled{extension}')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        psnr = 0

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            # -------------------
            psnr = self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return psnr
        # -------------------

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
            # -------------
            if metric == 'psnr':
                psnr = value
            # ----------------
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

        # ---------------
        return psnr
        # ---------------

    def get_current_visuals(self):
        out_dict = OrderedDict()

        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
