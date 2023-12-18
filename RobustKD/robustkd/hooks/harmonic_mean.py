# Author: Weinan He
# Mail: sapphire9877@gmail.com
# ----------------------------------------------
from mmcv.runner.hooks import Hook
from fastda.utils import get_root_logger, get_root_writer
from .cls_accuracy import ClsAccuracy
from fastda.hooks import HOOKS


@HOOKS.register_module()
class HarmonicMean(Hook):
    def __init__(self, runner, pred_key='pred'):
        for ind, (key, _) in enumerate(runner.test_loaders.items()):
            if ind == 0:
                self.test_dataset_name = key
            elif ind == 1:
                self.val_dataset_name = key
        # assert self.val_dataset_name is not None, "you should specify val dataset"
        if not hasattr(self, 'val_dataset_name'):
            self.val_dataset_name = self.test_dataset_name
        self.counter = 0
        self.pred_key = pred_key

    def after_val_iter(self, runner):
        pass

    def after_val_epoch(self, runner):
        logger = get_root_logger()
        writer = get_root_writer()
        #
        base_class_acc = None
        novel_class_acc = None
        for hook in runner._hooks:
            if isinstance(hook, ClsAccuracy):
                if hook.pred_key == self.pred_key:
                    if hook.dataset_name == self.test_dataset_name:
                        base_class_acc = hook.current_acc
                    if hook.dataset_name == self.val_dataset_name:  # not elif,
                        novel_class_acc = hook.current_acc
        assert novel_class_acc is not None, "you should specify ClassAccuracy hook for val dataset"
        #
        harmonic_mean = 2.0 / (1.0 / base_class_acc + 1.0 / novel_class_acc)
        #
        logger.info(
            "Iteration {}, harmonic_mean = {},".format(runner.iteration, harmonic_mean))
        writer.add_scalar('harmonic_mean_{}'.format(self.val_dataset_name), harmonic_mean, global_step=runner.iteration)
