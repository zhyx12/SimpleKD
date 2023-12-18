# Author: Weinan He
# Mail: sapphire9877@gmail.com
# ----------------------------------------------
import os.path

import torch
import torch.nn.functional as F
from mmcv.runner.hooks import Hook
from fastda.utils.metrics import RunningMetric
from fastda.utils import get_root_logger, get_root_writer, concat_all_gather
from mmcv.runner import get_dist_info
import pickle
from fastda.hooks import HOOKS
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np


@HOOKS.register_module()
class ClsAccuracyForODA(Hook):
    def __init__(self, runner, dataset_index, major_comparison=False, pred_key='pred', class_acc=True):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = list(runner.test_loaders.items())[dataset_index][0]
        self.dataset_index = dataset_index
        if rank == 0:
            self.running_metrics = RunningMetric()  #
            log_interval = max(len(runner.test_loaders[self.dataset_name]) - 1, 1)
            # self.running_metrics.add_metrics('{}_cls'.format(pred_key), group_name='val_loss',
            #                                  log_interval=log_interval)
        self.major_comparison = major_comparison
        self.best_acc = 0.0
        self.current_acc = 0.0
        self.pred_key = pred_key
        self.class_acc = class_acc  # default: True ** 

        # oda
        self.current_os = 0.0
        self.current_os_star = 0.0
        self.current_unknown_acc = 0.0
        self.current_hos = 0.0
        #
        rank, world_size = get_dist_info()
        loader = list(enumerate(runner.test_loaders.items()))[0][1][1]
        num_image = len(loader.dataset)
        num_class = loader.dataset.n_classes
        self.logits_bank = torch.randn(num_image, num_class).to('cuda:{}'.format(rank))
        self.label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))

    def before_val_epoch(self, runner):
        num_class = runner.trainer.num_class  # 尽管只有25个类，但混淆矩阵需要26个类，这个runner.trainer是Validator
        self.confusion_metric = torch.zeros((num_class + 1, num_class + 1),
                                            device="cuda:{}".format(self.local_rank))  # ** 修改

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if self.dataset_name == dataset_name:
            gt = batch_output['gt']

            # bank
            image_ind = batch_output['image_ind']
            pred = batch_output[self.pred_key]
            # collect res from all gpu
            image_ind = concat_all_gather(image_ind)
            pred = concat_all_gather(pred)
            self.logits_bank[image_ind] = pred
            self.label_bank[image_ind] = gt.to('cuda:{}'.format(self.local_rank))

    def after_val_epoch(self, runner):

        # # ----------------
        # # 获取ODA混淆矩阵（针对其他ODA方法，这里可以修改）
        # # kmeans
        # all_output = F.softmax(runner.logits_bank, dim=1)
        # ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1) / np.log(runner.trainer.num_class)
        # ent = ent.float().cpu()
        # kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        # labels = kmeans.predict(ent.reshape(-1,1))
        #
        # # get unknown
        # idx = np.where(labels==1)[0]
        # iidx = 0
        # if ent[idx].mean() > ent.mean():
        #     iidx = 1
        #
        # _, predict = torch.max(all_output, 1)
        # predict[np.where(labels==iidx)[0]] = runner.trainer.num_class
        #
        # # confusion_metric
        # matrix = confusion_matrix(runner.label_bank.cpu(), predict.cpu())
        # self.confusion_metric += torch.Tensor(matrix).to('cuda:{}'.format(self.local_rank))
        #
        # # ----------------
        print(self.logits_bank.shape)
        _, predict = torch.max(self.logits_bank, dim=1)
        matrix = confusion_matrix(self.label_bank.cpu(), predict.cpu())
        # ----------------
        # ----------------（下面是所有ODA方法通用的）

        tmp_confusion_mat = torch.from_numpy(matrix)
        # torch.distributed.reduce(tmp_confusion_mat, dst=0, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.barrier()

        # save confusion_mat
        if self.local_rank == 0:
            confusion_mat_save_path = os.path.join(runner.logdir, 'confusion_mat_{}.pkl'.format(runner.iteration))
            with open(confusion_mat_save_path, 'wb') as f:
                pickle.dump(tmp_confusion_mat.cpu().numpy(), f)

        # calculate acc
        correct_count = torch.sum(torch.diag(tmp_confusion_mat)).item()
        total_count = torch.sum(tmp_confusion_mat).item()
        if self.local_rank == 0:
            acc = correct_count / total_count
            self.current_acc = acc
            if acc > self.best_acc:
                self.best_acc = acc
                if self.major_comparison:  # default: False
                    runner.save_flag = True

            logger = get_root_logger()
            writer = get_root_writer()
            #
            self.running_metrics.log_metrics(runner.iteration, force_log=True)
            # overall accuracy
            writer.add_scalar('{}_acc_{}'.format(self.pred_key, self.dataset_name), acc,
                              global_step=runner.iteration)

            if self.class_acc:  # default: True  (in ODA)
                class_acc = []
                class_sum = torch.sum(tmp_confusion_mat, dim=1)
                # class-wise accuracy
                for i in range(runner.trainer.num_class + 1):  # ODA中多一个未知类
                    tmp_acc = tmp_confusion_mat[i, i].item() / class_sum[i].item()
                    class_acc.append(tmp_acc)
                    writer.add_scalar('{}_class_wise_acc_{}/class_{}'.format(self.pred_key, self.dataset_name, i),
                                      tmp_acc, global_step=runner.iteration)

                # OS
                OS = sum(class_acc) / (runner.trainer.num_class + 1)
                # 已知类的均值acc，即OS*
                known_acc = class_acc[:runner.trainer.num_class]
                OS_star = sum(known_acc) / runner.trainer.num_class
                # HOS by AaD
                unknown_acc = class_acc[-1]
                HOS = 2 * OS_star * unknown_acc / (OS_star + unknown_acc)

                writer.add_scalar('{}_ODA_{}/OS'.format(self.pred_key, self.dataset_name),
                                  OS, global_step=runner.iteration)
                writer.add_scalar('{}_ODA_{}/OS_star'.format(self.pred_key, self.dataset_name),
                                  OS_star, global_step=runner.iteration)
                writer.add_scalar('{}_ODA_{}/unknown'.format(self.pred_key, self.dataset_name),
                                  unknown_acc, global_step=runner.iteration)
                writer.add_scalar('{}_ODA_{}/HOS'.format(self.pred_key, self.dataset_name),
                                  HOS, global_step=runner.iteration)
                logger.info(
                    '#### ODA CURRENT #### Iteration {}: {} {} ODA performance: OS {:.4f} OS_star {:.4f} unknown {:.4f} HOS {:.4f}'.format(
                        runner.iteration, self.pred_key, self.dataset_name, OS, OS_star, unknown_acc, HOS))

                self.current_os = OS
                self.current_os_star = OS_star
                self.current_unknown_acc = unknown_acc
                self.current_hos = HOS

            logger.info('Iteration {}: {} {} acc {:.4f}'.format(runner.iteration, self.pred_key, self.dataset_name, acc))
            logger.info(
                'total img of {} is {}, right {}'.format(self.dataset_name, total_count, correct_count))
        #
        self.confusion_metric = None
