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


@HOOKS.register_module()
class CollectFeat(Hook):
    def __init__(self, runner, dataset_index, class_num, feat_dim=2048):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = list(runner.test_loaders.items())[dataset_index][0]
        self.dataset_index = dataset_index
        self.feat_dim = feat_dim
        self.class_num = class_num

    def before_val_epoch(self, runner):
        self.gt_array = np.zeros(60000)
        self.feat_array = np.zeros((60000, self.feat_dim))
        self.score_array = np.zeros((60000, self.class_num))
        self.index = 0

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if self.dataset_name == dataset_name:
            if self.index < 60000:
                gt = batch_output['gt']
                feat = batch_output['feat']
                logits = batch_output['pred']
                self.text_feat = batch_output['text_feat']
                self.gt_array[self.index:self.index + gt.shape[0]] = gt.cpu().numpy()
                self.feat_array[self.index:self.index + gt.shape[0]] = feat.cpu().numpy()
                self.score_array[self.index:self.index + gt.shape[0]] = F.softmax(logits, dim=1).cpu().numpy()
                self.index += gt.shape[0]

    def after_val_epoch(self, runner):
        final_gt_array = self.gt_array[0:self.index]
        final_feat_array = self.feat_array[0:self.index]
        #
        # classifier= runner.model_dict['base_model'].module.online_classifier
        # heuristic = (
        #         classifier.heuristic.weight.detach() + classifier.heuristic1.weight.detach() + classifier.heuristic2.weight.detach())
        # fc_weight = classifier.fc2.weight.detach() - heuristic
        # fc_weight = F.normalize(fc_weight)
        #
        # fc_weight = runner.model_dict['base_model'].module.online_classifier.fc.weight_v.detach()
        #
        class_prototype = np.zeros((self.class_num, self.feat_dim))
        for i in range(self.class_num):
            cls_ind = np.nonzero(final_gt_array == i)[0]
            tmp_cls_feat = final_feat_array[cls_ind]
            class_prototype[i, :] = np.mean(tmp_cls_feat, axis=0)
        save_path = os.path.join(runner.logdir, '{}_feat_array.pkl'.format(self.dataset_name))

        with open(save_path, 'wb') as f:
            # pickle.dump(class_prototype, f)
            pickle.dump([self.gt_array[0:self.index], self.feat_array[0:self.index], self.score_array[0:self.index],
                         self.text_feat.cpu().numpy()], f)
        #

        self.gt_array = None
        self.feat_array = None
