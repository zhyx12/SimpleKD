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
import seaborn as sns
import matplotlib.pyplot as plt
import io
import math


@HOOKS.register_module()
class ClassRelationshipVis(Hook):
    def __init__(self, runner, dataset_index, group_index=None):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = list(runner.test_loaders.items())[dataset_index][0]
        self.dataset_index = dataset_index
        self.group_index = group_index
        self.class_num = list(runner.test_loaders.items())[dataset_index][1].dataset.n_classes

    def before_val_epoch(self, runner):
        pass
        # self.class_relation_mat = np.zeros((self.class_num, self.class_num))
        # self.base_class_relation_mat = np.zeros((base_class_num, base_class_num))
        # self.novel_class_relation_mat = np.zeros((novel_class_num, novel_class_num))

    def after_val_iter(self, runner):
        pass

    def after_val_epoch(self, runner):
        writer = get_root_writer()
        base_model = runner.model_dict['base_model']
        text_embedding = base_model.module.test_text_embedding
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        #
        base_class_num = math.ceil(self.class_num / 2.0)
        novel_class_num = math.floor(self.class_num / 2.0)
        #
        self.all_class_relation_mat = text_embedding @ text_embedding.T
        self.base_class_relation_mat = text_embedding[0:base_class_num] @ text_embedding[0:base_class_num].T
        self.novel_class_relation_mat = text_embedding[base_class_num:] @ text_embedding[base_class_num:].T
        non_diag_mask = 1 - torch.eye(self.class_num, device=text_embedding.device)
        non_diag_mask_base = 1 - torch.eye(base_class_num, device=text_embedding.device)
        non_diag_mask_novel = 1 - torch.eye(novel_class_num, device=text_embedding.device)
        all_class_relation_mean = torch.mean(self.all_class_relation_mat * non_diag_mask).item()
        base_class_relation_mean = torch.mean(self.base_class_relation_mat * non_diag_mask_base).item()
        novel_class_relation_mean = torch.mean(self.novel_class_relation_mat * non_diag_mask_novel).item()
        #
        figure = plt.figure()
        sns.heatmap(self.all_class_relation_mat.cpu().numpy())
        writer.add_figure('class_relation/all', figure, runner.iteration)
        writer.add_scalar('class_relation_mean/all', all_class_relation_mean, global_step=runner.iteration)
        #
        plt.close()
        #
        figure = plt.figure()
        sns.heatmap(self.base_class_relation_mat.cpu().numpy())
        writer.add_figure('class_relation/base', figure, runner.iteration)
        writer.add_scalar('class_relation_mean/base', base_class_relation_mean, global_step=runner.iteration)
        #
        plt.close()
        #
        figure = plt.figure()
        sns.heatmap(self.novel_class_relation_mat.cpu().numpy())
        writer.add_figure('class_relation/novel', figure, runner.iteration)
        writer.add_scalar('class_relation_mean/novel', novel_class_relation_mean, global_step=runner.iteration)
        #
        plt.close()



