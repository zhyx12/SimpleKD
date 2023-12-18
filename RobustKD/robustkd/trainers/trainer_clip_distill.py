#
# ----------------------------------------------
import torch
import torch.nn.functional as F
from fastda.hooks import MetricsLogger
from fastda.runner import BaseTrainer, BaseValidator, TRAINER, VALIDATOR
from mmcv.runner import get_dist_info
from ..utils.dkd_loss import dkd_loss, prob_product_kd

from ..models.LoRA_layer import lora_moving_average, lora_diversity_loss, copy_lora_parameters, merge_lora_and_reset, \
    reset_lora_parameter
from fastda.utils import get_root_writer, get_root_logger
import numpy as np


@VALIDATOR.register_module(name='clip_distill')
class ValidatorCLIPDistill(BaseValidator):
    def __init__(self, basic_parameters):
        super(ValidatorCLIPDistill, self).__init__(**basic_parameters)

    def eval_iter(self, val_batch_data):
        # val_img, val_label, val_name = val_batch_data
        #
        if isinstance(val_batch_data, dict):
            val_img = val_batch_data['img']
            val_label = val_batch_data['gt_label'].squeeze(1)
            val_metas = val_batch_data['img_metas']
        else:
            val_img, val_label = val_batch_data
        # coop clip
        label_list = self.test_loaders[self.val_dataset_key].dataset.label_transform_list
        model = self.model_dict['base_model']
        with torch.no_grad():
            logits, image_features, text_features = model(val_img, base_class_list=label_list)
        if isinstance(logits, dict):
            logits.update({
                'img': val_img,
                'gt': val_label,
                'feat': logits['pred'],
            })
            return logits
        else:
            return {'img': val_img,
                    'gt': val_label,
                    'img_metas': val_metas,
                    'feat': image_features,
                    'text_feat': text_features,
                    'pred': logits,
                    }


@TRAINER.register_module('clip_distill')
class TrainerCLIPDistill(BaseTrainer):
    def __init__(self, basic_parameters,
                 #
                 lambda_temp=1.0, base_class_list=None, ema_weight_decay=0.999,
                 moving_average_type=None, max_iters=None, lambda_class_relation=1.0, new_class_num=2,
                 lambda_kl_loss=0.0, kl_target_temp=0.07,
                 kl_target_moving_average_type=None, bank_size=64,
                  logits_fusion_type='mean', start_iteration=200,
                  num_branches=1,
                  target_type='dynamic',
                instance_relation_input_type='feat',
                 fusion_type_in_instance_relation='max',
                 use_orig_output_in_instance_relation=False, lambda_instance_relation=0.02,
                 prob_dkd_temp=0.07, instance_relation_alpha=0.0, instance_relation_beta=1.0,
                 mu=2500, sigma=1000,
                 ):
        super(TrainerCLIPDistill, self).__init__(**basic_parameters)
        #
        self.lambda_temp = lambda_temp
        self.base_class_list = base_class_list
        self.ema_weight_decay = ema_weight_decay
        self.moving_average_type = moving_average_type
        self.max_iters = max_iters
        self.lambda_class_relation = lambda_class_relation
        self.new_class_num = new_class_num
        self.lambda_kl_loss = lambda_kl_loss
        self.kl_target_temp = kl_target_temp
        self.kl_target_moving_average_type = kl_target_moving_average_type
        self.bank_size = bank_size
        self.logits_fusion_type = logits_fusion_type
        self.start_iteration = start_iteration
        self.num_branches = num_branches
        self.target_type = target_type
        self.use_rand_mask_in_instance_relation = use_rand_mask_in_instance_relation
        self.rand_mask_ratio = rand_mask_ratio
        self.instance_relation_input_type = instance_relation_input_type
        self.detach_text_in_instance_relation = detach_text_in_instance_relation
        self.mask_self_class_in_instance_relation = mask_self_class_in_instance_relation
        self.fusion_type_in_instance_relation = fusion_type_in_instance_relation
        self.detach_image_in_instance_relation = detach_image_in_instance_relation
        self.use_orig_output_in_instance_relation = use_orig_output_in_instance_relation
        self.lambda_instance_relation = lambda_instance_relation
        self.prob_dkd_temp = prob_dkd_temp
        self.use_class_bank = use_class_bank
        self.instance_relation_alpha = instance_relation_alpha
        self.instance_relation_beta = instance_relation_beta
        self.teacher_extra_temp = teacher_extra_temp
        # 修改成list形式
        if isinstance(self.lambda_kl_loss, float):
            self.lambda_kl_loss = [self.lambda_kl_loss] * self.num_branches
        if isinstance(self.dkd_loss_alpha, float):
            self.dkd_loss_alpha = [self.dkd_loss_alpha] * self.num_branches
        if isinstance(self.dkd_loss_beta, float):
            self.dkd_loss_beta = [self.dkd_loss_beta] * self.num_branches
        if isinstance(self.lambda_dkd_loss, float):
            self.lambda_dkd_loss = [self.lambda_dkd_loss] * self.num_branches
        if isinstance(self.teacher_extra_temp, float):
            self.teacher_extra_temp = [self.teacher_extra_temp] * self.num_branches
        if isinstance(self.target_type, str):
            self.target_type = [self.target_type] * self.num_branches
        if isinstance(self.loss_type, str):
            self.loss_type = [self.loss_type] * self.num_branches
        if isinstance(self.kl_target_temp, float):
            self.kl_target_temp = [self.kl_target_temp] * self.num_branches
        #
        self.logit_scale = self.model_dict['base_model'].module.logit_scale
        rank, world_size = get_dist_info()
        self.world_size = world_size
        self.num_class = self.train_loaders[0].dataset.n_classes
        # 增加记录
        if self.local_rank == 0:
            log_names = ['cls', ]
            # log_names = ['cls', 'ent', 'div']
            loss_metrics = MetricsLogger(log_names=log_names, group_name='loss', log_interval=self.log_interval)
            self.register_hook(loss_metrics)
        # 根据数据集生成文本特征
        self.class_names = self.train_loaders[0].dataset.get_classes()
        class_embeddings = self.model_dict['base_model'].module.init_prompt(self.class_names)
        #
        if self.base_class_list is not None:
            self.base_class_num = len(self.base_class_list)
        else:
            self.base_class_num = self.num_class
            self.base_class_list = list(range(self.num_class))
        self.real_class_num = self.base_class_num
        class_embedding = self.model_dict['base_model'].module.class_embeddings[self.base_class_list].detach()
        self.class_relation = class_embedding @ class_embedding.T
        #
        self.class_contrastive_simmat = torch.softmax(class_embedding @ class_embedding.T / 0.17, dim=-1)
        self.class_distance = (1 - self.class_contrastive_simmat) * 0.3
        #
        self.beta_generator = torch.distributions.beta.Beta(0.5, 0.5)
        self.beta_accumulate = 0.0
        if self.moving_average_type == 'gauss':
            self.gauss_generator = self.gauss_list_generator(mu, sigma)
            self.gauss_accumulate = 0.0
        else:
            self.gauss_generator = None
        #
        base_model = self.model_dict['base_model']
        feat_dim = base_model.module.image_encoder.output_dim
        self.feat_dim = feat_dim
        # bank
        train_sample_num = len(self.train_loaders[0].dataset)
        self.fill_bank = torch.zeros(train_sample_num, dtype=torch.float32).to('cuda:{}'.format(rank))
        self.train_label_bank = torch.zeros(train_sample_num, dtype=torch.long).to('cuda:{}'.format(rank))
        self.teacher_train_img_feat_bank = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.teacher_train_text_feat_bank = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.teacher_train_logits_bank = torch.randn(train_sample_num, self.real_class_num).to('cuda:{}'.format(rank))
        self.student_train_img_feat_bank = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.student_train_text_feat_bank = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.student_train_logits_bank = torch.randn(train_sample_num, self.real_class_num).to('cuda:{}'.format(rank))
        #
        self.teacher_train_img_feat_bank_2 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.teacher_train_text_feat_bank_2 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.teacher_train_logits_bank_2 = torch.randn(train_sample_num, self.real_class_num).to('cuda:{}'.format(rank))
        self.student_train_img_feat_bank_2 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.student_train_text_feat_bank_2 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.student_train_logits_bank_2 = torch.randn(train_sample_num, self.real_class_num).to('cuda:{}'.format(rank))
        #
        self.teacher_train_img_feat_bank_3 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.teacher_train_text_feat_bank_3 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.teacher_train_logits_bank_3 = torch.zeros(train_sample_num, self.real_class_num).to('cuda:{}'.format(rank))
        self.student_train_img_feat_bank_3 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.student_train_text_feat_bank_3 = torch.randn(train_sample_num, feat_dim).to('cuda:{}'.format(rank))
        self.student_train_logits_bank_3 = torch.zeros(train_sample_num, self.real_class_num).to('cuda:{}'.format(rank))
        #
        #

    def train_iter(self, *args):
        src_img_weak = args[0][0]['img']
        src_label_weak = args[0][0]['gt_label'].squeeze(1)
        image_ind = args[0][0]['image_ind']
        batch_size = src_img_weak.shape[0]
        #
        src_domain_label = args[0][0].get('domain_label', None)
        tgt_labeled_size = 0
        #
        #
        batch_metrics = {}
        batch_metrics['loss'] = {}
        #
        base_model = self.model_dict['base_model']  # CLIP模型要使用.model
        #
        self.zero_grad_all()
        tmp_base_class_list = self.base_class_list if not self.add_novel_class_in_dkd else None
        student_results, teacher_results, div_loss, orig_results = base_model(src_img_weak, label=src_label_weak,
                                                                              base_class_list=tmp_base_class_list,
                                                                              num_branches=self.num_branches)
        orig_logits, orig_image_feat, orig_text_feat = orig_results
        #
        loss = 0.0
        #
        for i in range(self.num_branches):
            end_index = len(self.base_class_list)
            if self.add_extra_prototype:
                end_index += self.num_extra_prototypes
            tmp_student_logits = student_results[0][i][:, 0:end_index]
            if i == 0:
                loss += F.cross_entropy(tmp_student_logits, src_label_weak, label_smoothing=self.lambda_label_smooth)
            elif i == 1:
                loss += F.cross_entropy(tmp_student_logits, src_label_weak, label_smoothing=self.lambda_label_smooth)
            else:
                raise NotImplementedError
        #
        if self.iteration > self.start_iteration:
            #
            loss += div_loss * self.lambda_div
            #
            teacher_logits_list = [x.unsqueeze(0) for x in teacher_results[0]]
            fusion_logits, _ = torch.max(torch.cat(teacher_logits_list, dim=0), dim=0)
            mean_fusion_logits = torch.mean(torch.cat(teacher_logits_list, dim=0), dim=0)
            for i in range(self.num_branches):
                tmp_student_logits = student_results[0][i]
                #####################################
                #
                if self.target_type[i] == 'fixed':
                    tmp_kl_target = self.class_relation[src_label_weak, :] / self.kl_target_temp[i]
                elif self.target_type[i] == 'orig_instance':
                    tmp_kl_target = orig_logits / self.teacher_extra_temp[i]
                elif self.target_type[i] == 'dynamic':
                    tmp_kl_target = teacher_results[0][i] / self.teacher_extra_temp[i]
                elif self.target_type[i] == 'dynamic_max_fusion':
                    tmp_kl_target = fusion_logits / self.teacher_extra_temp[i]
                elif self.target_type[i] == 'dynamic_mean_fusion':
                    tmp_kl_target = mean_fusion_logits / self.teacher_extra_temp[i]
                elif self.target_type[i] == 'dynamic_prototype':
                    tmp_text_prototpye = teacher_results[2][i]
                    tmp_class_relation = (tmp_text_prototpye @ tmp_text_prototpye.T / self.kl_target_temp[i])
                    tmp_kl_target = tmp_class_relation[src_label_weak, :]
                elif self.target_type[i] == 'dynamic_prototype_mean_fusion':
                    tmp_text_prototpye_1 = teacher_results[2][0]
                    tmp_text_prototpye_2 = teacher_results[2][1]
                    tmp_class_relation_1 = (tmp_text_prototpye_1 @ tmp_text_prototpye_1.T / self.kl_target_temp[i])
                    tmp_class_relation_2 = (tmp_text_prototpye_2 @ tmp_text_prototpye_2.T / self.kl_target_temp[i])
                    tmp_kl_target = (tmp_class_relation_1 + tmp_class_relation_2) / 2.0
                    tmp_kl_target = tmp_kl_target[src_label_weak, :]
                elif self.target_type[i] == 'dynamic_prototype_max_fusion':
                    tmp_text_prototpye_1 = teacher_results[2][0]
                    tmp_text_prototpye_2 = teacher_results[2][1]
                    tmp_class_relation_1 = (tmp_text_prototpye_1 @ tmp_text_prototpye_1.T / self.kl_target_temp[i])
                    tmp_class_relation_2 = (tmp_text_prototpye_2 @ tmp_text_prototpye_2.T / self.kl_target_temp[i])
                    tmp_kl_target = torch.maximum(tmp_class_relation_1, tmp_class_relation_2)
                    tmp_kl_target = tmp_kl_target[src_label_weak, :]
                else:
                    raise NotImplementedError
                #
                tmp_student_res = (student_results[0][i], student_results[1][i], student_results[2][i])
                if self.num_branches == 1:
                    teacher_res_1 = (teacher_results[0][0], teacher_results[1][0], teacher_results[2][0])
                    teacher_res_2 = (teacher_results[0][0], teacher_results[1][0], teacher_results[2][0])
                else:
                    teacher_res_1 = (teacher_results[0][0], teacher_results[1][0], teacher_results[2][0])
                    teacher_res_2 = (teacher_results[0][1], teacher_results[1][1], teacher_results[2][1])
                if self.use_orig_output_in_instance_relation:
                    teacher_res_1 = (orig_results[0], orig_results[1], orig_results[2])
                    teacher_res_2 = (orig_results[0], orig_results[1], orig_results[2])
                    tmp_bank_index = 2
                else:
                    tmp_bank_index = i
                loss += self.instance_relation_dkd_loss(tmp_student_res, teacher_res_1, teacher_res_2,
                                                        src_label_weak, bank_index=tmp_bank_index,
                                                        input_type=self.instance_relation_input_type,
                                                        alpha=self.instance_relation_alpha,
                                                        beta=self.instance_relation_beta
                                                        ) * self.lambda_instance_relation
                loss += F.kl_div(F.log_softmax(tmp_student_logits, dim=-1),
                                 F.softmax(tmp_kl_target, dim=-1),
                                 reduction='batchmean') * self.lambda_kl_loss[i]
                #
                tmp_student_res = (student_results[0][i], student_results[1][i], student_results[2][i])
                if self.num_branches == 1:
                    teacher_res_1 = (teacher_results[0][0], teacher_results[1][0], teacher_results[2][0])
                    teacher_res_2 = (teacher_results[0][0], teacher_results[1][0], teacher_results[2][0])
                else:
                    teacher_res_1 = (teacher_results[0][0], teacher_results[1][0], teacher_results[2][0])
                    teacher_res_2 = (teacher_results[0][1], teacher_results[1][1], teacher_results[2][1])
                if self.use_orig_output_in_instance_relation:
                    teacher_res_1 = (orig_results[0], orig_results[1], orig_results[2])
                    teacher_res_2 = (orig_results[0], orig_results[1], orig_results[2])
                    tmp_bank_index = 2
                else:
                    tmp_bank_index = i
                loss += self.instance_relation_dkd_loss(tmp_student_res, teacher_res_1, teacher_res_2,
                                                        src_label_weak, bank_index=tmp_bank_index,
                                                        input_type=self.instance_relation_input_type,
                                                        alpha=self.instance_relation_alpha,
                                                        beta=self.instance_relation_beta
                                                        ) * self.lambda_instance_relation
                loss += F.kl_div(F.log_softmax(tmp_student_logits, dim=-1),
                                 F.softmax(tmp_kl_target, dim=-1),
                                 reduction='batchmean') * self.lambda_kl_loss[i]
        #
        loss.backward()
        self.step_grad_all()
        self.update_training_bank(student_results, teacher_results, orig_results, src_label_weak, image_ind)
        if self.use_class_bank:
            self.update_class_bank(student_results, teacher_results, src_label_weak)
        #
        if self.moving_average_type is None:
            lambda_current, lambda_last = 1.0, 0.0
        elif self.moving_average_type == 'ema':
            lambda_current, lambda_last = 1 - self.ema_weight_decay, self.ema_weight_decay
        elif self.moving_average_type == 'simple':
            lambda_current = 1.0 / (self.iteration + 1)
            lambda_last = 1.0 - lambda_current
        elif self.moving_average_type == 'beta':
            process = torch.tensor((self.iteration + 0.5) / (self.max_iters + 1))
            alpha_current = torch.exp(self.beta_generator.log_prob(process)).to('cuda:{}'.format(self.local_rank))
            self.beta_accumulate += alpha_current
            lambda_current = alpha_current / self.beta_accumulate
            lambda_last = 1.0 - lambda_current
        elif self.moving_average_type == 'gauss':
            alpha_current = self.gauss_generator[self.iteration]
            self.gauss_accumulate += alpha_current
            lambda_current = alpha_current / self.gauss_accumulate
            lambda_last = 1.0 - lambda_current
        else:
            raise NotImplementedError
        lora_moving_average(self.model_dict['base_model'], lambda_last, lambda_current)
        batch_metrics['loss']['cls'] = loss.item()
        return batch_metrics

    def load_pretrained_model(self, weights_path):
        logger = get_root_logger()
        weights = torch.load(weights_path, map_location='cpu')
        weights = weights['base_model']
        self.model_dict['base_model'].load_state_dict(weights, strict=False)
        logger.info('load pretrained model {}'.format(weights_path))

    def js_div(self, pred_logits, target_logits):
        prob = F.softmax(pred_logits, dim=-1)
        branch_2_prob = F.softmax(target_logits, dim=-1)
        log_target_softmax_out = torch.log(branch_2_prob)
        kl_loss = torch.mean((F.kl_div(log_target_softmax_out, prob, reduction='none').sum(-1)))
        loss = kl_loss
        #
        softmax_out = F.log_softmax(pred_logits, dim=-1)
        kl_loss = torch.mean(
            (F.kl_div(softmax_out, branch_2_prob, reduction='none').sum(-1)))
        loss += kl_loss
        return loss

    def update_class_bank(self, student_results, teacher_result, label):
        """
        """
        branch_1_logits, branch_2_logits = teacher_result[0]
        branch_1_img_feat, branch_2_img_feat = teacher_result[1]
        branch_1_student_logits, branch_2_student_logits = student_results[0]
        branch_1_student_img_feat, branch_2_student_img_feat = student_results[1]
        unique_class = torch.unique(label)
        for class_id in unique_class:
            tmp_class = class_id.to(torch.int)
            tmp_class_ind = torch.nonzero(label == tmp_class).squeeze(1)
            tmp_num = tmp_class_ind.shape[0]
            tmp_start = int(self.class_bank_ptr[tmp_class])
            tmp_end = min(tmp_start + tmp_num, self.bank_size)
            tmp_real_size = tmp_end - tmp_start
            #
            tmp_class_image_feat_t1 = branch_1_img_feat[tmp_class_ind[0:tmp_real_size], :].detach()
            tmp_class_image_feat_t2 = branch_2_img_feat[tmp_class_ind[0:tmp_real_size], :].detach()
            tmp_class_image_feat_s1 = branch_1_student_img_feat[tmp_class_ind[0:tmp_real_size], :].detach()
            tmp_class_image_feat_s2 = branch_2_student_img_feat[tmp_class_ind[0:tmp_real_size], :].detach()
            tmp_class_logits_t1 = branch_1_logits[tmp_class_ind[0:tmp_real_size], :].detach()
            tmp_class_logits_t2 = branch_2_logits[tmp_class_ind[0:tmp_real_size], :].detach()
            tmp_class_logits_s1 = branch_1_student_logits[tmp_class_ind[0:tmp_real_size], :].detach()
            tmp_class_logits_s2 = branch_2_student_logits[tmp_class_ind[0:tmp_real_size], :].detach()
            #
            self.class_image_feat_bank_teacher_1[tmp_class, tmp_start:tmp_end, :] = tmp_class_image_feat_t1
            self.class_image_feat_bank_teacher_2[tmp_class, tmp_start:tmp_end, :] = tmp_class_image_feat_t2
            self.class_image_feat_bank_student_1[tmp_class, tmp_start:tmp_end, :] = tmp_class_image_feat_s1
            self.class_image_feat_bank_student_2[tmp_class, tmp_start:tmp_end, :] = tmp_class_image_feat_s2
            #
            self.class_logits_bank_teacher_1[tmp_class, tmp_start:tmp_end, :] = tmp_class_logits_t1
            self.class_logits_bank_teacher_2[tmp_class, tmp_start:tmp_end, :] = tmp_class_logits_t2
            self.class_logits_bank_student_1[tmp_class, tmp_start:tmp_end, :] = tmp_class_logits_s1
            self.class_logits_bank_student_2[tmp_class, tmp_start:tmp_end, :] = tmp_class_logits_s2
            #
            self.class_fill_bank[tmp_class, tmp_start:tmp_end] = 1.0
            #
            if tmp_end == self.bank_size:
                self.class_bank_ptr[tmp_class] = 0
            else:
                self.class_bank_ptr[tmp_class] = tmp_end
            # print(class_id.item(), self.class_bank_ptr[tmp_class].item())
        # print('fill bank sum ', torch.sum(self.class_fill_bank).item())

    def update_training_bank(self, student_output, teacher_output, orig_results, label, img_ind):
        if len(student_output[0]) == 1:
            branch_1_logits = teacher_output[0][0]
            branch_1_img_feat = teacher_output[1][0]
            branch_1_student_logits = student_output[0][0]
            branch_1_student_img_feat = student_output[1][0]
            branch_2_logits = branch_1_logits
            branch_2_img_feat = branch_1_img_feat
            branch_2_student_logits = branch_1_student_logits
            branch_2_student_img_feat = branch_1_student_img_feat
        else:
            branch_1_logits, branch_2_logits = teacher_output[0]
            branch_1_img_feat, branch_2_img_feat = teacher_output[1]
            branch_1_student_logits, branch_2_student_logits = student_output[0]
            branch_1_student_img_feat, branch_2_student_img_feat = student_output[1]
        #
        orig_logits, orig_img_feat, orig_text_feat = orig_results
        #
        self.teacher_train_img_feat_bank[img_ind, :] = branch_1_img_feat.detach()
        # self.teacher_train_text_feat_bank[img_ind, :] = branch_1_text_feat.detach()
        self.teacher_train_logits_bank[img_ind, :] = branch_1_logits.detach()
        #
        self.teacher_train_img_feat_bank_2[img_ind, :] = branch_2_img_feat.detach()
        # self.teacher_train_text_feat_bank_2[img_ind, :] = branch_2_text_feat.detach()
        self.teacher_train_logits_bank_2[img_ind, :] = branch_2_logits.detach()
        #
        if orig_logits is not None:
            self.teacher_train_img_feat_bank_3[img_ind, :] = orig_img_feat.detach()
            # self.teacher_train_text_feat_bank_3[img_ind, :] = orig_text_feat.detach()
            self.teacher_train_logits_bank_3[img_ind, :] = orig_logits.detach()
        #
        self.student_train_img_feat_bank[img_ind, :] = branch_1_student_img_feat.detach()
        # self.student_train_text_feat_bank[img_ind, :] = branch_1_student_text_feat.detach()
        self.student_train_logits_bank[img_ind, :] = branch_1_student_logits.detach()
        #
        self.student_train_img_feat_bank_2[img_ind, :] = branch_2_student_img_feat.detach()
        # self.student_train_text_feat_bank_2[img_ind, :] = branch_2_student_text_feat.detach()
        self.student_train_logits_bank_2[img_ind, :] = branch_2_student_logits.detach()
        #

        self.train_label_bank[img_ind] = label.detach()
        self.fill_bank[img_ind] = 1.0

    def instance_relation_dkd_loss(self, student_output, teacher_output_1, teacher_output_2, label, bank_index,
                                   input_type='feat', alpha=0.0, beta=1.0):
        teacher_logits_1, teacher_img_feat_1, teacher_text_feat_1 = teacher_output_1
        teacher_logits_2, teacher_img_feat_2, teacher_text_feat_2 = teacher_output_2
        student_logits, student_img_feat, student_text_feat = student_output
        #
        #
        if bank_index in [0, 1]:
            if self.use_class_bank:
                teacher_train_img_feat_bank_1 = self.class_image_feat_bank_teacher_1.view(-1, self.feat_dim)
                teacher_train_img_feat_bank_2 = self.class_image_feat_bank_teacher_2.view(-1, self.feat_dim)
                teacher_train_logits_bank_1 = self.class_logits_bank_teacher_1.view(-1, self.real_class_num)
                teacher_train_logits_bank_2 = self.class_logits_bank_teacher_2.view(-1, self.real_class_num)
            else:
                teacher_train_img_feat_bank_1 = self.teacher_train_img_feat_bank
                teacher_train_img_feat_bank_2 = self.teacher_train_img_feat_bank_2
                teacher_train_logits_bank_1 = self.teacher_train_logits_bank
                teacher_train_logits_bank_2 = self.teacher_train_logits_bank_2
        elif bank_index == 2:
            teacher_train_img_feat_bank_1 = self.teacher_train_img_feat_bank_3
            teacher_train_img_feat_bank_2 = self.teacher_train_img_feat_bank_3
            teacher_train_logits_bank_1 = self.teacher_train_logits_bank_3
            teacher_train_logits_bank_2 = self.teacher_train_logits_bank_3
        else:
            raise NotImplementedError
        if bank_index == 0:
            if self.use_class_bank:
                student_train_img_feat_bank = self.class_image_feat_bank_student_1.view(-1, self.feat_dim)
                student_train_logits_bank = self.class_logits_bank_student_1.view(-1, self.real_class_num)
            else:
                student_train_img_feat_bank = self.student_train_img_feat_bank
                student_train_logits_bank = self.student_train_logits_bank
        else:
            if self.use_class_bank:
                student_train_img_feat_bank = self.class_image_feat_bank_student_2.view(-1, self.feat_dim)
                student_train_logits_bank = self.class_logits_bank_student_2.view(-1, self.real_class_num)
            else:
                student_train_img_feat_bank = self.student_train_img_feat_bank_2
                student_train_logits_bank = self.student_train_logits_bank_2
        #######################
        # 图像-图像的dkd
        if self.use_class_bank:
            tmp_label_bank = self.class_training_label_bank
            tmp_fill_bank = self.class_fill_bank.view(-1)
        else:
            tmp_label_bank = self.train_label_bank
            tmp_fill_bank = self.fill_bank
        if self.mask_self_class_in_instance_relation:
            mask = (label.unsqueeze(1) != tmp_label_bank.unsqueeze(0)).to(torch.float32)
            mask *= tmp_fill_bank.unsqueeze(0)
        else:
            mask = (tmp_fill_bank.unsqueeze(0)).to(torch.float32)
        if self.use_rand_mask_in_instance_relation:
            rand_mask = (
                    torch.rand((tmp_label_bank.shape[0],),
                               device='cuda:{}'.format(self.local_rank)) < self.rand_mask_ratio).to(
                torch.float32)
            mask *= rand_mask.unsqueeze(0)
        # 基于特征
        if input_type == 'feat':
            student_image_relation = torch.log_softmax(
                (student_img_feat @ student_train_img_feat_bank.T) / self.instance_relation_dkd_temp - 1000.0 * (
                        1.0 - mask), dim=1)
            teacher_logits_1 = teacher_img_feat_1 @ teacher_train_img_feat_bank_1.T / self.instance_relation_dkd_temp
            teacher_logits_2 = teacher_img_feat_2 @ teacher_train_img_feat_bank_2.T / self.instance_relation_dkd_temp
            if self.fusion_type_in_instance_relation == 'max':
                fusion_logits = torch.maximum(teacher_logits_1, teacher_logits_2)
            elif self.fusion_type_in_instance_relation == 'min':
                fusion_logits = torch.minimum(teacher_logits_1, teacher_logits_2)
            elif self.fusion_type_in_instance_relation == 'mean':
                fusion_logits = (teacher_logits_1 + teacher_logits_2) / 2.0
            else:
                fusion_logits = teacher_logits_1 if bank_index == 0 else teacher_logits_2
            # softmax方式获取概率
            # teacher_image_relation = torch.softmax(fusion_logits - 1000.0 * (1.0 - mask), dim=1)
            # instance_dkd_loss = - torch.mean((student_image_relation * teacher_image_relation).sum(-1))
            # L1方式获取概率
            student_relation_logits = student_img_feat @ student_train_img_feat_bank.T
            student_relation_logits = 0.5 * (student_relation_logits + 1.0)
            student_relation_prob = student_relation_logits / student_relation_logits.sum(dim=1, keepdim=True)
            fusion_logits = 0.5 * (fusion_logits + 1.0)
            teacher_image_relation = fusion_logits / fusion_logits.sum(dim=1, keepdim=True)
            student_image_relation = torch.log(student_relation_prob)
            instance_dkd_loss = - torch.mean((student_image_relation * teacher_image_relation * mask).sum(-1))
        # 基于概率
        elif input_type == 'prob':
            if self.detach_text_in_instance_relation:
                logit_scale = self.logit_scale.exp()
                student_logits = logit_scale * student_img_feat @ student_text_feat.detach().t()
            if self.detach_image_in_instance_relation:
                logit_scale = self.logit_scale.exp()
                student_logits = logit_scale * student_img_feat.detach() @ student_text_feat.t()
            student_relation_logits = torch.softmax(student_logits, dim=1) @ torch.softmax(student_train_logits_bank,
                                                                                           dim=1).T / self.instance_relation_dkd_temp
            teacher_logits_1 = torch.softmax(teacher_logits_1, dim=1) @ torch.softmax(teacher_train_logits_bank_1.T,
                                                                                      dim=1) / self.instance_relation_dkd_temp
            teacher_logits_2 = torch.softmax(teacher_logits_2, dim=1) @ torch.softmax(teacher_train_logits_bank_2.T,
                                                                                      dim=1) / self.instance_relation_dkd_temp
            if self.fusion_type_in_instance_relation == 'max':
                fusion_logits = torch.maximum(teacher_logits_1, teacher_logits_2)
            elif self.fusion_type_in_instance_relation == 'min':
                fusion_logits = torch.minimum(teacher_logits_1, teacher_logits_2)
            elif self.fusion_type_in_instance_relation == 'mean':
                fusion_logits = (teacher_logits_1 + teacher_logits_2) / 2.0
            else:
                fusion_logits = teacher_logits_1 if bank_index == 0 else teacher_logits_2
            # softmax方式获取概率
            # student_image_relation = torch.log_softmax(student_relation_logits - 1000.0 * (1.0 - mask), dim=1)
            # teacher_image_relation = torch.softmax(fusion_logits - 1000.0 * (1.0 - mask), dim=1)
            # instance_dkd_loss_1 = - torch.mean((student_image_relation * teacher_image_relation).sum(-1))
            # L1方式获取概率
            student_relation_logits = 0.5 * (student_relation_logits + 1.0)
            student_relation_prob = student_relation_logits / student_relation_logits.sum(dim=1, keepdim=True)
            fusion_logits = 0.5 * (fusion_logits + 1.0)
            teacher_image_relation = fusion_logits / fusion_logits.sum(dim=1, keepdim=True)
            student_image_relation = torch.log(student_relation_prob)
            instance_dkd_loss_1 = - torch.mean((student_image_relation * teacher_image_relation * mask).sum(-1))
            # student_image_relation_2 = torch.log_softmax(student_relation_logits - 1000.0 * mask, dim=1)
            # teacher_image_relation_2 = torch.softmax(fusion_logits - 1000.0 * mask, dim=1)
            # instance_dkd_loss_2 = - torch.mean((student_image_relation_2 * teacher_image_relation_2).sum(-1))
            #
            # instance_dkd_loss = instance_dkd_loss_1 * beta + instance_dkd_loss_2 * alpha
            #
            instance_dkd_loss = instance_dkd_loss_1
        elif input_type == 'feat_representation':
            if self.detach_text_in_instance_relation:
                logit_scale = self.logit_scale.exp()
                student_logits = logit_scale * student_img_feat @ student_text_feat.detach().t()
            #
            student_rerep = self.feat_representation(student_logits, student_text_feat)
            student_bank_rerep = self.feat_representation(student_train_logits_bank, student_text_feat)
            student_relation_logits = student_rerep @ student_bank_rerep.T / self.instance_relation_dkd_temp

            #
            teacher_rerep_1 = self.feat_representation(teacher_logits_1, teacher_text_feat_1)
            teacher_bank_rerep_1 = self.feat_representation(teacher_train_logits_bank_1, teacher_text_feat_1)
            teacher_rerep_2 = self.feat_representation(teacher_logits_2, teacher_text_feat_2)
            teacher_bank_rerep_2 = self.feat_representation(teacher_train_logits_bank_2, teacher_text_feat_2)
            teacher_logits_1 = teacher_rerep_1 @ teacher_bank_rerep_1.T / self.instance_relation_dkd_temp
            teacher_logits_2 = teacher_rerep_2 @ teacher_bank_rerep_2.T / self.instance_relation_dkd_temp
            if self.fusion_type_in_instance_relation == 'max':
                fusion_logits = torch.maximum(teacher_logits_1, teacher_logits_2)
            elif self.fusion_type_in_instance_relation == 'mean':
                fusion_logits = (teacher_logits_1 + teacher_logits_2) / 2.0
            else:
                fusion_logits = teacher_logits_1 if bank_index == 0 else teacher_logits_2
            # softmax方式获取概率
            # student_image_relation = torch.log_softmax(student_relation_logits - 1000.0 * (1.0 - mask), dim=1)
            # teacher_image_relation = torch.softmax(fusion_logits - 1000.0 * (1.0 - mask), dim=1)
            # L1方式获取概率
            student_relation_logits = 0.5 * (student_relation_logits + 1.0)
            student_relation_prob = student_relation_logits / student_relation_logits.sum(dim=1, keepdim=True)
            fusion_logits = 0.5 * (fusion_logits + 1.0)
            teacher_image_relation = fusion_logits / fusion_logits.sum(dim=1, keepdim=True)
            student_image_relation = torch.log(student_relation_prob)
            instance_dkd_loss = - torch.mean((student_image_relation * teacher_image_relation * mask).sum(-1))
            # instance_dkd_loss = dkd_loss(student_relation_logits, fusion_logits, label, alpha, beta)
        else:
            raise NotImplementedError
        ################################################################
        #############################
        #####
        # instance_dkd_loss = torch.mean(
        #     F.kl_div(torch.log(student_text_relation + 1e-12), teacher_text_relation+1e-12, reduction='none').sum(-1))  # 这里teacher的1e-12是为了防止nan
        #
        return instance_dkd_loss

    def text_to_text_relation_dkd_loss(self, student_logits_list, teacher_logits_list, temperature):
        # tmp_class_num = self.real_class_num if not self.add_novel_class_in_dkd else self.num_class
        tmp_class_num = student_logits_list[0].shape[0]
        mask = torch.eye(tmp_class_num).to('cuda:{}'.format(self.local_rank))
        loss = 0.0
        #
        text_relation_logits_list = []
        for text_feat in teacher_logits_list:
            # print(text_feat.shape, temperature)
            tmp_text_relation_logits = text_feat @ text_feat.T / temperature
            text_relation_logits_list.append(tmp_text_relation_logits.unsqueeze(0))
        fusion_teacher_logits = torch.max(torch.cat(text_relation_logits_list, dim=0), dim=0)[0]
        teacher_text_relation = torch.softmax((fusion_teacher_logits - 10000 * mask), dim=1)
        #
        for text_feat in student_logits_list:
            student_text_relation = torch.log_softmax((text_feat @ text_feat.T / temperature - 10000 * mask), dim=1)
            loss += - torch.mean((student_text_relation * teacher_text_relation).sum(-1))
        return loss

    def text_to_image_relation(self, student_output, teacher_output_1, teacher_output_2, label, bank_index):
        teacher_logits_1, teacher_img_feat_1, teacher_text_feat_1 = teacher_output_1
        teacher_logits_2, teacher_img_feat_2, teacher_text_feat_2 = teacher_output_2
        student_logits, student_img_feat, student_text_feat = student_output
        #
        if self.use_class_bank:
            teacher_train_img_feat_bank_1 = self.class_image_feat_bank_teacher_1.view(-1, self.feat_dim)
            teacher_train_img_feat_bank_2 = self.class_image_feat_bank_teacher_2.view(-1, self.feat_dim)
        else:
            teacher_train_img_feat_bank_1 = self.teacher_train_img_feat_bank
            teacher_train_img_feat_bank_2 = self.teacher_train_img_feat_bank_2

        if bank_index == 0:
            if self.use_class_bank:
                student_train_img_feat_bank = self.class_image_feat_bank_student_1.view(-1, self.feat_dim)
                student_train_logits_bank = self.class_logits_bank_student_1.view(-1, self.real_class_num)
            else:
                student_train_img_feat_bank = self.student_train_img_feat_bank
                student_train_logits_bank = self.student_train_logits_bank
        elif bank_index == 1:
            if self.use_class_bank:
                student_train_img_feat_bank = self.class_image_feat_bank_student_2.view(-1, self.feat_dim)
                student_train_logits_bank = self.class_logits_bank_student_2.view(-1, self.real_class_num)
            else:
                student_train_img_feat_bank = self.student_train_img_feat_bank_2
                student_train_logits_bank = self.student_train_logits_bank_2
        else:
            raise RuntimeError('bank_index should be 0 or 1')
        ######################
        if self.use_class_bank:
            tmp_label_bank = self.class_training_label_bank
            tmp_fill_bank = self.class_fill_bank.view(-1)
        else:
            tmp_label_bank = self.train_label_bank
            tmp_fill_bank = self.fill_bank
        # text特征到图像特征的dkd损失, 只使用bank内的特征
        text_label = torch.arange(self.real_class_num).to('cuda:{}'.format(self.local_rank))
        mask = (text_label.unsqueeze(1) != tmp_label_bank.unsqueeze(0)).to(torch.float32)
        mask *= tmp_fill_bank.unsqueeze(0)
        # # 随机选择一部分样本
        if self.use_rand_mask_in_instance_relation:
            rand_mask = (
                    torch.rand((tmp_label_bank.shape[0],), device='cuda:{}'.format(self.local_rank)) < 0.14).to(
                torch.float32)
            mask *= rand_mask.unsqueeze(0)
        student_text_relation = torch.log_softmax((student_text_feat @ student_train_img_feat_bank.T) /
                                                  self.instance_relation_dkd_temp - 10000.0 * (1.0 - mask), dim=1)
        # fusion
        teacher_logits_1 = teacher_text_feat_1 @ teacher_train_img_feat_bank_1.T / self.instance_relation_dkd_temp
        teacher_logits_2 = teacher_text_feat_2 @ teacher_train_img_feat_bank_2.T / self.instance_relation_dkd_temp
        fusion_logits = torch.maximum(teacher_logits_1, teacher_logits_2)
        # fusion_logits = torch.mean(torch.stack([teacher_logits_1, teacher_logits_2]), dim=0)
        teacher_text_relation = torch.softmax(fusion_logits - 10000.0 * (1.0 - mask), dim=1)
        ######################
        # text特征到图像特征的dkd损失, 将当前的图像特征加入到bank中
        # text_label = torch.arange(self.real_class_num).to('cuda:{}'.format(self.local_rank))
        # all_student_image_feat = torch.cat((student_img_feat, student_train_img_feat_bank), dim=0)
        # all_label_bank = torch.cat((label, self.train_label_bank), dim=0)
        # rand_mask = (torch.rand((self.train_label_bank.shape[0],), device='cuda:{}'.format(self.local_rank)) < 0.14).to(
        #     torch.float32)
        # fill_bank = self.fill_bank * rand_mask
        # all_fill_bank = torch.cat((torch.ones_like(label), fill_bank), dim=0)
        # # mask = (text_label.unsqueeze(1) != all_label_bank.unsqueeze(0)).to(torch.float32)
        # mask = torch.ones((text_label.shape[0], all_label_bank.shape[0]),device='cuda:0').to(torch.float32)
        # mask *= all_fill_bank.unsqueeze(0)
        # student_text_relation = torch.log_softmax(
        #     (student_text_feat @ all_student_image_feat.T) / self.instance_relation_dkd_temp - 1000 * (1 - mask),
        #     dim=1)
        # all_teacher_image_feat_1 = torch.cat((teacher_img_feat_1, teacher_train_img_feat_bank_1), dim=0)
        # all_teacher_image_feat_2 = torch.cat((teacher_img_feat_2, teacher_train_img_feat_bank_2), dim=0)
        # teacher_logits_1 = teacher_text_feat_1 @ all_teacher_image_feat_1.T / self.instance_relation_dkd_temp
        # teacher_logits_2 = teacher_text_feat_2 @ all_teacher_image_feat_2.T / self.instance_relation_dkd_temp
        # # fusion_logits = torch.mean(torch.stack([teacher_logits_1, teacher_logits_2]), dim=0)
        # fusion_logits = torch.maximum(teacher_logits_1, teacher_logits_2)
        # teacher_text_relation = torch.softmax(fusion_logits - 1000.0 * (1 - mask), dim=1)
        #
        instance_dkd_loss = - torch.mean((student_text_relation * teacher_text_relation).sum(-1))
        return instance_dkd_loss

    def feat_representation(self, logits, prototpye):
        prototpye = prototpye.detach()
        student_prob = torch.softmax(logits, dim=1)
        student_rerep = student_prob @ prototpye
        student_rerep = student_rerep / student_rerep.norm(dim=1, keepdim=True)
        return student_rerep

    def gauss_list_generator(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        gauss = [gauss(a) for a in range(1, self.max_iters + 1)]
        gauss = gauss / sum(gauss)
        return gauss
