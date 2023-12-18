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
from PIL import Image
import numpy as np


@HOOKS.register_module()
class SplitHighLowSamples(Hook):
    def __init__(self, runner, dataset_index, pred_key='pred', threshold=0.98):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = list(runner.test_loaders.items())[dataset_index][0]
        self.dataset_index = dataset_index
        self.pred_key = pred_key
        self.threshold = threshold

    def before_val_epoch(self, runner):
        # high_root_path = os.path.join(runner.logdir, 'high_confidence')
        # self.high_right_path = os.path.join(high_root_path, 'high_classify_right')
        # self.high_wrong_path = os.path.join(high_root_path, 'high_classify_wrong')
        # low_root_path = os.path.join(runner.logdir, 'low_confidence')
        # self.low_right_path = os.path.join(low_root_path, 'low_classify_right')
        # self.low_wrong_path = os.path.join(low_root_path, 'low_classify_wrong')
        self.count_dict = {
            'high_wrong': [0 for i in range(65)],
            'high_right': [0 for i in range(65)],
            'low_wrong': [0 for i in range(65)],
            'low_right': [0 for i in range(65)],
        }
        self.category_list = {
            'high_wrong': [],
            'high_right': [],
            'low_wrong': [],
            'low_right': [],
        }

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        img = batch_output['img']
        img_metas = batch_output['img_metas']
        if self.dataset_name == dataset_name:
            gt = batch_output['gt']
            pred = batch_output[self.pred_key]
            prob = torch.softmax(pred, dim=1)
            pred_max = torch.argmax(pred, dim=1)
            #
            batch_size = gt.shape[0]
            for i in range(batch_size):
                tmp_img = img[i].cpu().transpose(0, 1).transpose(1, 2).numpy()
                tmp_gt = gt[i].item()
                tmp_pred = pred_max[i].item()
                tmp_max_prob = torch.max(prob[i]).item()
                tmp_metas = img_metas.data[0][i]
                tmp_filepath = tmp_metas['ori_filename'].split('/')
                tmp_class = tmp_filepath[1]
                tmp_img_name = tmp_filepath[2]
                tmp_mean = tmp_metas['img_norm_cfg']['mean']
                tmp_std = tmp_metas['img_norm_cfg']['std']
                #
                orig_img = Image.fromarray((tmp_img * tmp_std + tmp_mean).astype(np.uint8))
                orig_img.save('./tmp.jpg')
                #
                high_low_flag = 'high' if tmp_max_prob >= self.threshold else 'low'
                right_wrong_flag = 'right' if tmp_gt == tmp_pred else 'wrong'
                tmp_class_path = os.path.join(runner.logdir, '{}_confidence'.format(high_low_flag),
                                              '{}_classify_{}'.format(high_low_flag, right_wrong_flag), tmp_class)
                #
                self.count_dict['{}_{}'.format(high_low_flag, right_wrong_flag)][tmp_gt] += 1
                self.category_list['{}_{}'.format(high_low_flag, right_wrong_flag)].append(
                    '{}_{}'.format(tmp_class, tmp_img_name))
                #
                if not os.path.exists(tmp_class_path):
                    os.makedirs(tmp_class_path)
                tmp_img_path = os.path.join(tmp_class_path, tmp_img_name)
                orig_img.save(tmp_img_path)

    def after_val_epoch(self, runner):
        #
        res_path = os.path.join(runner.logdir, 'count_dict.pkl')
        with open(res_path, 'wb') as f:
            pickle.dump(self.count_dict, f)
        category_list_path = os.path.join(runner.logdir, 'category_list.pkl')
        with open(category_list_path, 'wb') as f:
            pickle.dump(self.category_list, f)
        print('high right {}'.format(sum(self.count_dict['high_right'])))
        print('high wrong {}'.format(sum(self.count_dict['high_wrong'])))
        print('low right {}'.format(sum(self.count_dict['low_right'])))
        print('low wrong {}'.format(sum(self.count_dict['low_wrong'])))
