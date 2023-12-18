#
# ----------------------------------------------
import os.path

import torch
import torch.nn.functional as F
from mmcv.runner.hooks import Hook
from fastda.utils.metrics import RunningMetric
from fastda.utils import get_root_logger, get_root_writer
from mmcv.runner import get_dist_info
import pickle
from fastda.hooks import HOOKS
import numpy as np
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import random
from PIL import Image


def reshape_transform(tensor, height=14, width=14):
    tensor = tensor.transpose(0, 1)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


@HOOKS.register_module()
class GradCAMHook(Hook):
    def __init__(self, runner, dataset_index, eigen_smooth=False, aug_smooth=False):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = list(runner.test_loaders.items())[dataset_index][0]
        self.dataset_index = dataset_index
        self.eigen_smooth = eigen_smooth
        self.aug_smooth = aug_smooth
        #
        model = runner.model_dict['base_model']
        target_layers = [model.module.image_encoder.transformer.resblocks[-1].ln_1]
        self.cam = LayerCAM(model=model, target_layers=target_layers,
                                use_cuda=True,
                                reshape_transform=reshape_transform)
        # self.mean = [122.771, 116.746, 104.093]
        # self.std = [68.500, 66.632, 70.323]
        self.mean = [104.093, 116.746, 122.771]
        self.std = [70.323, 66.632, 68.500]

    def before_val_epoch(self, runner):
        self.right_save_path = os.path.join(runner.logdir,
                                            '{}_{}_grad_cam_right'.format(self.dataset_name, runner.iteration))
        self.wrong_save_path = os.path.join(runner.logdir,
                                            '{}_{}_grad_cam_wrong'.format(self.dataset_name, runner.iteration))

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if self.dataset_name == dataset_name:
            img = batch_output['img']
            pred = batch_output['pred']
            pred_class = torch.argmax(pred, dim=1)
            self.cam.batch_size = img.size(0)
            gt = batch_output['gt']
            img_metas = batch_output['img_metas']
            self.cam.model.train(True)
            grayscale_cam = self.cam(input_tensor=img,
                                     targets=None,
                                     eigen_smooth=self.eigen_smooth,
                                     aug_smooth=self.aug_smooth)
            grayscale_cam = grayscale_cam
            #
            img = img.transpose(1, 2).transpose(2, 3)
            img = img.cpu().numpy()
            img = img[:, :, :, ::-1]
            rgb_img = img * np.array(self.std) + np.array(self.mean)
            rgb_img = np.clip(rgb_img, 0, 255)
            # rgb_img = rgb_img.astype(np.uint8)
            rgb_img /= 255.0
            rgb_img = rgb_img.astype(np.float32)
            #
            num_img = rgb_img.shape[0]
            for i in range(num_img):
                cam_image = show_cam_on_image(rgb_img[i, :], grayscale_cam[i, :])
                tmp_img_name = img_metas.data[0][i]['filename'].split('/')[-1][:-4]
                tmp_class_name = img_metas.data[0][i]['filename'].split('/')[-2]
                if pred_class[i] == gt[i]:
                    save_path = self.right_save_path
                else:
                    save_path = self.wrong_save_path
                tmp_class_path = os.path.join(save_path, tmp_class_name)
                if not os.path.exists(tmp_class_path):
                    os.makedirs(tmp_class_path)
                tmp_img_path = os.path.join(tmp_class_path, '{}.jpg'.format(tmp_img_name))
                cv2.imwrite(tmp_img_path, cam_image)

            #
            self.cam.model.train(False)

    def after_val_epoch(self, runner):
        pass
