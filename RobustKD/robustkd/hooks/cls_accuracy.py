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


@HOOKS.register_module()
class ClsAccuracy(Hook):
    def __init__(self, runner, dataset_index, major_comparison=False, pred_key='pred', class_acc=False, group_acc=None,
                 num_class=None):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = list(runner.test_loaders.items())[dataset_index][0]
        self.dataset_index = dataset_index
        if rank == 0:
            self.running_metrics = RunningMetric()  #
            log_interval = max(len(runner.test_loaders[self.dataset_name]) - 1, 1)
            self.running_metrics.add_metrics('{}_cls'.format(pred_key), group_name='val_loss',
                                             log_interval=log_interval)
        self.major_comparison = major_comparison
        self.best_acc = 0.0
        self.current_acc = 0.0
        self.pred_key = pred_key
        self.class_acc = class_acc or group_acc is not None
        self.group_acc = group_acc
        if self.group_acc is not None:
            assert isinstance(self.group_acc, dict), 'group_acc should be a dict'
        #
        self.num_class = runner.trainer.num_class if num_class is None else num_class

    def before_val_epoch(self, runner):
        self.confusion_metric = torch.zeros((self.num_class, self.num_class), device="cuda:{}".format(self.local_rank))

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if self.dataset_name == dataset_name:
            gt = batch_output['gt']
            pred = batch_output[self.pred_key]
            pred_max = torch.argmax(pred, dim=1)
            for i in range(gt.shape[0]):
                self.confusion_metric[gt[i], pred_max[i]] += 1
            #
            loss = F.cross_entropy(pred, gt)
            if self.local_rank == 0:
                batch_metrics = {'val_loss': {'{}_cls'.format(self.pred_key): loss.item()}}
                self.running_metrics.update_metrics(batch_metrics)

    def after_val_epoch(self, runner):
        #
        tmp_confusion_mat = self.confusion_metric
        torch.distributed.reduce(tmp_confusion_mat, dst=0, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.barrier()
        #
        if self.local_rank == 0:
            confusion_mat_save_path = os.path.join(runner.logdir, 'confusion_mat_{}.pkl'.format(runner.iteration))
            with open(confusion_mat_save_path, 'wb') as f:
                pickle.dump(tmp_confusion_mat.cpu().numpy(), f)
        correct_count = torch.sum(torch.diag(tmp_confusion_mat)).item()
        total_count = torch.sum(tmp_confusion_mat).item()
        if self.local_rank == 0:
            acc = correct_count / total_count
            self.current_acc = acc
            if acc > self.best_acc:
                self.best_acc = acc
                if self.major_comparison:
                    runner.save_flag = True
            #
            logger = get_root_logger()
            writer = get_root_writer()
            #
            self.running_metrics.log_metrics(runner.iteration, force_log=True)
            # overall accuracy
            writer.add_scalar('{}_acc_{}'.format(self.pred_key, self.dataset_name), acc,
                              global_step=runner.iteration)
            if self.class_acc:
                class_acc = []
                class_sum = torch.sum(tmp_confusion_mat, dim=1)
                # class-wise accuracy
                for i in range(self.num_class):
                    tmp_acc = tmp_confusion_mat[i, i].item() / (class_sum[i].item() + 1e-8)  # avoid zero division
                    class_acc.append(tmp_acc)
                    writer.add_scalar('{}_class_wise_acc_{}/class_{}'.format(self.pred_key, self.dataset_name, i),
                                      tmp_acc, global_step=runner.iteration)
                final_class_acc = sum(class_acc) / self.num_class
                writer.add_scalar('{}_class_wise_acc_{}/0_mean'.format(self.pred_key, self.dataset_name),
                                  final_class_acc, global_step=runner.iteration)
                logger.info(
                    'Iteration {}: {} {} class-wise acc {}'.format(runner.iteration, self.pred_key, self.dataset_name,
                                                                   final_class_acc))
                #
                if self.group_acc is not None:
                    for name, class_index in self.group_acc.items():
                        avg_acc = [class_acc[i] for i in class_index]
                        avg_acc = sum(avg_acc) / len(avg_acc)
                        writer.add_scalar('{}_group_{}_classwise_acc/{}'.format(self.pred_key, self.dataset_name, name),
                                          avg_acc, global_step=runner.iteration)
                        logger.info(
                            'Iteration {}: {} {} group {} classwise acc {}'.format(runner.iteration, self.pred_key,
                                                                                   self.dataset_name, name, avg_acc))
            #
            if self.group_acc is not None:
                for name, class_index in self.group_acc.items():
                    sub_confusion_mat = tmp_confusion_mat[class_index, :][:, class_index]
                    sub_correct_count = torch.sum(torch.diag(sub_confusion_mat)).item()
                    sub_total_count = torch.sum(tmp_confusion_mat[class_index, :]).item()
                    sub_acc = sub_correct_count / sub_total_count
                    writer.add_scalar('{}_group_{}_sample_wise_acc/{}'.format(self.pred_key, self.dataset_name, name),
                                      sub_acc, global_step=runner.iteration)
                    logger.info(
                        'Iteration {}: {} {} group {} sample-wise acc {}'.format(runner.iteration, self.pred_key,
                                                                                 self.dataset_name, name, sub_acc))
            #
            logger.info('Iteration {}: {} {} acc {}'.format(runner.iteration, self.pred_key, self.dataset_name, acc))
            logger.info(
                'total img of {} is {}, right {}'.format(self.dataset_name, total_count, correct_count))
            #

        #
        self.confusion_metric = None
