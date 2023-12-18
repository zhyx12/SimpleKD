# Author: Weinan He
# Mail: sapphire9877@gmail.com
# ----------------------------------------------
from mmcv.runner.hooks import Hook
from fastda.utils import get_root_logger, get_root_writer
from .cls_accuracy_for_oda import ClsAccuracyForODA
from fastda.hooks import HOOKS


@HOOKS.register_module()
class ClsBestAccuracyByTestForODA(Hook):
    def __init__(self, runner, patience=100, pred_key='pred'):
        for ind, (key, _) in enumerate(runner.test_loaders.items()):
            if ind == 0:
                self.test_dataset_name = key
            elif ind == 1:
                self.val_dataset_name = key
        # assert self.val_dataset_name is not None, "you should specify val dataset"
        if not hasattr(self, 'val_dataset_name'):
            self.val_dataset_name = self.test_dataset_name
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_iteration = 0
        self.counter = 0
        self.patience = patience
        self.pred_key = pred_key

        # oda
        self.best_os = 0
        self.best_os_star = 0
        self.best_unknown_acc = 0
        self.best_hos = 0

    def after_val_iter(self, runner):
        pass

    def after_val_epoch(self, runner):
        logger = get_root_logger()
        writer = get_root_writer()
        #
        test_acc = None
        val_acc = None
        for hook in runner._hooks:
            if isinstance(hook, ClsAccuracyForODA):  # 修改处 **
                if hook.pred_key == self.pred_key:
                    if hook.dataset_name == self.test_dataset_name:
                        test_acc = hook.current_acc

                        # oda     
                        os = hook.current_os
                        os_star = hook.current_os_star
                        unknown_acc = hook.current_unknown_acc
                        hos = hook.current_hos

                    if hook.dataset_name == self.val_dataset_name:  # not elif,
                        val_acc = hook.current_acc
        assert val_acc is not None, "you should specify ClassAccuracy hook for val dataset"
        #
        if test_acc >= self.best_test_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
            self.best_iteration = runner.iteration
            self.counter = 0
            runner.save_flag = True
        else:
            self.counter += 1
            if self.counter > self.patience:
                runner.early_stop_flag = True

        # oda
        if os >= self.best_os:
            self.best_os = os
        if os_star >= self.best_os_star:
            self.best_os_star = os_star
        if unknown_acc >= self.best_unknown_acc:
            self.best_unknown_acc = unknown_acc
        if hos >= self.best_hos:
            self.best_hos = hos
        #
        logger.info(
            "Iteration {}, best test acc = {}, occured in {} iterations, with val acc {}".format(runner.iteration,
                                                                                                 self.best_test_acc,
                                                                                                 self.best_iteration,
                                                                                                 self.best_val_acc))
        logger.info(
            "#### ODA BEST ####  OS = {}, OS_star = {}, unknown_acc {}, HOS {}".format(self.best_os,
                                                                                       self.best_os_star,
                                                                                       self.best_unknown_acc,
                                                                                       self.best_hos))                                                                                     
        
        writer.add_scalar('best_acc', self.best_test_acc, global_step=runner.iteration)
