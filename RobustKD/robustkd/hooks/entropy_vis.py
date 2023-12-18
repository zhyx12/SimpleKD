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

@HOOKS.register_module()
class EntropyVis(Hook):
    def __init__(self, runner, dataset_index, pred_key, group_index=None):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = list(runner.test_loaders.items())[dataset_index][0]
        self.dataset_index = dataset_index
        self.group_index = group_index
        self.pred_key = pred_key

    def before_val_epoch(self, runner):
        self.gt_array = np.zeros(60000)
        self.max_logits = np.zeros(60000)
        self.entropy_array = np.zeros(60000)
        self.max_prob = np.zeros(60000)
        self.orig_entropy = np.zeros(60000)
        self.orig_max_prob = np.zeros(60000)
        self.orig_max_logits = np.zeros(60000)
        self.index = 0

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if self.dataset_name == dataset_name:
            if self.index < 60000:
                gt = batch_output['gt']
                logits = batch_output[self.pred_key]
                self.gt_array[self.index:self.index + gt.shape[0]] = gt.cpu().numpy()
                prob = F.softmax(logits, dim=1)
                entropy = -torch.sum(prob * torch.log(prob), dim=1)
                self.max_prob[self.index:self.index + gt.shape[0]] = torch.max(prob, dim=1)[0].cpu().numpy()
                self.max_logits[self.index:self.index + gt.shape[0]] = torch.max(logits, dim=1)[0].cpu().numpy()
                self.entropy_array[self.index:self.index + gt.shape[0]] = entropy.cpu().numpy()
                #
                orig_prob = F.softmax(batch_output['orig_logits'], dim=1)
                orig_entropy = -torch.sum(orig_prob * torch.log(orig_prob), dim=1)
                self.orig_entropy[self.index:self.index + gt.shape[0]] = orig_entropy.cpu().numpy()
                self.orig_max_prob[self.index:self.index + gt.shape[0]] = torch.max(orig_prob, dim=1)[0].cpu().numpy()
                self.orig_max_logits[self.index:self.index + gt.shape[0]] = torch.max(batch_output['orig_logits'],
                                                                                      dim=1)[0].cpu().numpy()
                self.index += gt.shape[0]

    def after_val_epoch(self, runner):
        final_gt_array = self.gt_array[0:self.index]
        final_entropy_array = self.entropy_array[0:self.index]
        final_max_prob = self.max_prob[0:self.index]
        final_max_logits = self.max_logits[0:self.index]
        final_orig_entropy = self.orig_entropy[0:self.index]
        final_orig_max_prob = self.orig_max_prob[0:self.index]
        final_orig_max_logits = self.orig_max_logits[0:self.index]
        #
        mean_entropy = np.mean(final_entropy_array)
        #
        figure = plt.figure()
        bw_method = 0.1
        cumulative = False
        c1, c2, c3, c4 = sns.color_palette('Set2', 4)
        ax = sns.kdeplot(final_orig_max_logits, bw_method=bw_method, fill=True, color=c1,
                         cumulative=cumulative,
                         label='test')
        # plt.savefig('max_orig_logits_on_all_class_{}.svg'.format(self.dataset_name), dpi=300)
        #
        writer = get_root_writer()
        writer.add_figure('max_orig_logits_on_all_class_{}'.format(self.dataset_name), figure, runner.iteration)
