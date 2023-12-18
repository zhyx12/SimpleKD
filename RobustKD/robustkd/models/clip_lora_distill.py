import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import ModifiedResNet, VisionTransformer
from fastda.models import MODELS
from typing import Union
import random
from .LoRA_layer import LoRA_ViT, set_lora_index, set_lora_moving_average, set_merge_index, set_lora_coeff, \
    lora_diversity_loss
from .VPT_layer import Prompt_Vit, set_use_prompt_flag
import random

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


@MODELS.register_module()
class CLIPLoRADistill(nn.Module):
    def __init__(self, name, template=None,
                 model_device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                 jit: bool = False, download_root: str = None,
                 optimize_parameters=None,
                 lora_alpha: float = 2.0, lora_r: int = 2, lora_other_linear: bool = False,
                 test_lora_index=0, image_lora_init_train_index=0,
                 text_lr_coeff=1.0, text_test_lora_index=0, image_lr_coeff=1.0,
                 image_test_merge_index=(0, 1, 2), text_test_merge_index=(0, 1, 2), lora_dropout=0.0,
                 text_lora_dropout=0.0,
                 text_lora_rank=1, text_lora_alpha=1.0, image_lora_layers=list(range(12)),
                 text_lora_layers=list(range(12)), diversity_loss_index=(0, 1), test_with_full_moving_mat=False,
                 image_use_lora_for_k=False, text_use_lora_for_k=False, add_extra_prototype=False,
                 extra_prototypes_lr_coeff=1.0, num_extra_prototypes=1, test_with_student_branch=False,
                 vis_mode=False,
                 ):
        super().__init__()
        clip_model, _ = clip.load(name, model_device, jit, download_root)
        self.template = template
        self.text_lora_rank = text_lora_rank
        self.image_encoder = LoRA_ViT(clip_model.visual, lora_r=lora_r, lora_alpha=lora_alpha,
                                      lora_other_linear=lora_other_linear, lora_dropout=lora_dropout,
                                      lora_layers=image_lora_layers,
                                      test_with_full_moving_mat=test_with_full_moving_mat,
                                      use_lora_for_k=image_use_lora_for_k)
        set_lora_index(self.image_encoder, image_lora_init_train_index)
        self.text_encoder = LoRA_ViT(TextEncoder(clip_model), lora_r=text_lora_rank, lora_alpha=text_lora_alpha,
                                     lora_other_linear=lora_other_linear, lora_dropout=text_lora_dropout,
                                     lora_layers=text_lora_layers, test_with_full_moving_mat=test_with_full_moving_mat,
                                     use_lora_for_k=text_use_lora_for_k)
        prototype_vectors = torch.empty((num_extra_prototypes, 512), dtype=clip_model.dtype)
        nn.init.normal_(prototype_vectors, std=0.02)
        self.extra_prototypes = nn.Parameter(prototype_vectors)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model_token_embedding = clip_model.token_embedding
        self.encode_text = clip_model.encode_text
        self.mean_std = None
        self.test_lora_index = test_lora_index
        self.text_test_lora_index = text_test_lora_index
        self.image_test_merge_index = image_test_merge_index
        self.text_test_merge_index = text_test_merge_index
        self.diversity_loss_index = diversity_loss_index
        #
        self.optimize_parameters = optimize_parameters
        self.text_lr_coeff = text_lr_coeff
        self.image_lr_coeff = image_lr_coeff
        assert self.optimize_parameters is not None, "optimize_parameters must be set"
        #
        self.test_ready_flag = False
        self.test_text_embedding = None
        self.add_extra_prototype = add_extra_prototype
        self.extra_prototypes_lr_coeff = extra_prototypes_lr_coeff
        self.test_with_student_branch = test_with_student_branch
        self.vis_mode = vis_mode

    def forward(self, image, label=None, base_class_list=None, num_branches=1, **kwargs):
        if self.training:
            prompts = self.prompts_embedding[base_class_list, :,
                      :] if base_class_list is not None else self.prompts_embedding
            tokenized_prompts = self.tokenized_prompts
            tokenized_prompts = tokenized_prompts if base_class_list is None else tokenized_prompts[base_class_list]
            self.test_ready_flag = False
            # student分支的前向
            student_results = self.multi_branch_forward(num_branches=num_branches, prompts=prompts,
                                                        tokenized_prompts=tokenized_prompts,
                                                        image=image)
            #
            # teacher分支的前向
            # set_lora_moving_average(self.text_encoder, True)
            # set_lora_moving_average(self.image_encoder, True)
            # with torch.no_grad():
            #     # if isinstance(image, (list, tuple)):
            #     #     tmp_image = torch.cat(image, dim=0)
            #     teacher_results = self.multi_branch_forward(num_branches=num_branches, prompts=prompts,
            #                                                 tokenized_prompts=tokenized_prompts,
            #                                                 image=image)
            # set_lora_moving_average(self.text_encoder, False)
            # set_lora_moving_average(self.image_encoder, False)
            #
            # div_loss = lora_diversity_loss(self.image_encoder, (0, 1))
            # div_loss += lora_diversity_loss(self.text_encoder, (0, 1))
            div_loss = 0.0
            # 原始模型的预测结果
            # orig_logits = self.predict_from_original_model(image, base_class_list)
            # orig_logits = (None, None, None)
            orig_logits = self.predict_from_fusion_model(prompts=prompts,
                                                         tokenized_prompts=tokenized_prompts, image=image)
            if self.vis_mode:
                return student_results[0][0]
            else:
                teacher_results = (
                    (orig_logits[0], orig_logits[0]), (orig_logits[1], orig_logits[1]),
                    (orig_logits[2], orig_logits[2]))
                return student_results, teacher_results, div_loss, orig_logits
        else:
            if not self.test_ready_flag:
                # 使用单个提示词
                prompts = self.prompts_embedding
                tokenized_prompts = self.tokenized_prompts
                self.test_text_embedding = self.text_encoder(prompts, tokenized_prompts)
                self.test_ready_flag = True
                print('set flag')
            text_features = self.test_text_embedding if base_class_list is None else \
                self.test_text_embedding[base_class_list]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            #
            return logits, image_features, text_features

    def init_prompt(self, classnames):
        self.classnames = classnames
        with torch.no_grad():
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [self.template.format(name) for name in classnames]
            print(prompts)
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to('cuda')
            #
            embedding = self.clip_model_token_embedding(tokenized_prompts).type(self.dtype)  # num_class, 77, 512
            self.prompts_embedding = embedding
            ################
            # 这里的self.encode_text是clip的encode_text, 执行过程中还是会调用到clip的参数，可能存在不在同一个设备上的情况
            self.tokenized_prompts = tokenized_prompts.to('cuda')
            # class_embeddings = self.encode_text(self.tokenized_prompts.to('cuda'))
            class_embeddings = self.text_encoder(embedding, self.tokenized_prompts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            self.class_embeddings = class_embeddings.detach()
            #
            self.text_class_features = [class_embeddings.clone().detach() for i in range(self.text_lora_rank)]
            return class_embeddings

    def optim_parameters(self, lr):
        parameter_list = []
        # optim_name = ['prompt_learner', 'text_projector']
        # optim_name = ['prompt_learner', 'text_projector', 'lora_a', 'lora_b']
        optim_name = self.optimize_parameters

        # optim_name = ['prompt_learner', 'text_projector', 'lora_a', 'lora_b']

        def check_in(name):
            for n in optim_name:
                if n in name:
                    return True
            return False

        for name, param in self.named_parameters():
            if not check_in(name):
                print('not optimizing {}'.format(name))
                param.requires_grad_(False)
            else:
                if 'text_encoder' in name:
                    real_lr = lr * self.text_lr_coeff
                elif 'image_encoder' in name:
                    real_lr = lr * self.image_lr_coeff
                elif 'extra_prototypes' in name:
                    real_lr = lr * self.extra_prototypes_lr_coeff
                else:
                    real_lr = lr
                # real_lr = lr if not 'text_encoder' in name else lr * self.text_lr_coeff
                # if 'text_encoder' in name and ('lora_a.0' in name or 'lora_b.0' in name):
                #     real_lr *= 2.0
                # if 'text_encoder' in name and ('lora_a.1' in name or 'lora_b.1' in name):
                #     real_lr *= 2.0
                print('optimizing {}, lr {}'.format(name, real_lr))
                param.requires_grad_(True)
                parameter_list.append({'params': param, "lr": real_lr})
        # parameter_list = [{"params": self.prompt_learner.parameters(), "lr": lr},
        #                   # {"params": self.text_projector.parameters(), "lr": lr},
        #                   # {"params": self.image_encoder.parameters(), "lr": lr},
        #                   ]
        # if self.use_text_projector:
        #     parameter_list.append({"params": self.text_projector, "lr": lr, "weight_decay": 0.0})
        return parameter_list

    def train(self, mode):
        super().train(mode)
        # self.text_encoder.eval()
        # self.image_encoder.eval()
        #
        if not self.test_with_student_branch:
            set_lora_moving_average(self, not mode)
        else:
            set_lora_moving_average(self, False)  # test with running branch
        if not mode:
            set_lora_index(self.image_encoder, self.test_lora_index)
            set_merge_index(self.image_encoder, self.image_test_merge_index)
            set_lora_index(self.text_encoder, self.text_test_lora_index)
            set_merge_index(self.text_encoder, self.text_test_merge_index)
        else:
            set_lora_index(self.image_encoder, 0)
            set_lora_index(self.text_encoder, 0)

        return self

    def single_branch_forward(self, prompts, tokenized_prompts, image):
        text_features = self.text_encoder(prompts, tokenized_prompts)
        if self.add_extra_prototype:
            text_features = torch.cat([text_features, self.extra_prototypes], dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits, image_features, text_features

    def multi_branch_forward(self, num_branches, prompts, tokenized_prompts, image):
        image_feat_list = []
        text_feat_list = []
        logits_list = []
        #
        for i in range(num_branches):
            # if isinstance(image, (list, tuple)):
            #     tmp_image = image[i]
            # else:
            #     tmp_image = image
            set_lora_index(self.text_encoder, i)
            set_lora_index(self.image_encoder, i)
            result = self.single_branch_forward(prompts, tokenized_prompts, image.type(self.dtype))
            logits_list.append(result[0])
            image_feat_list.append(result[1])
            text_feat_list.append(result[2])
        return logits_list, image_feat_list, text_feat_list

    def predict_from_original_model(self, image, base_class_list):
        with torch.no_grad():
            set_lora_index(self.image_encoder, -1)
            set_lora_index(self.text_encoder, -1)
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ self.class_embeddings[base_class_list, :].t()
            set_lora_index(self.image_encoder, 0)
            set_lora_index(self.text_encoder, 0)
            return logits, image_features, self.class_embeddings[base_class_list, :]

    def predict_from_fusion_model(self, prompts, tokenized_prompts, image):
        self.train(False)
        with torch.no_grad():
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
        self.train(True)
        return logits, image_features, text_features
