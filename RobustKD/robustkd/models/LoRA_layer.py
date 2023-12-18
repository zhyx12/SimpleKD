from copy import deepcopy
from math import sqrt
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import bmm, softmax, cat, Tensor, empty, addmm
from torch.nn import Linear as tLinear, Parameter, init
from torch.nn.functional import dropout

NUM_LoRA_SET = 4


class Linear(tLinear):
    """
    LoRA wrapped Linear layer.
    """

    def __init__(self, in_features: int, out_features: int, *args, lora_r: tuple = (0,), lora_alpha: tuple = (1.,),
                 lora_dropout: float = 0., test_with_full_moving_mat=False, **kwargs):
        """
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout

        See torch.nn.Linear for other params
        """
        super().__init__(in_features, out_features, *args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        if isinstance(lora_r, int):
            self.lora_r = [lora_r, ] * NUM_LoRA_SET
        else:
            self.lora_r = lora_r
        self.lora_index = 0
        self.using_moving_average = False
        self.test_with_full_moving_mat = test_with_full_moving_mat
        if self.lora_r[0]:  # enable lora
            self.weight.requires_grad = False  # freeze main weights
            self.lora_a = nn.ParameterList(
                [Parameter(init.kaiming_uniform_(empty(self.lora_r[i], in_features), a=sqrt(5)))
                 for i in range(NUM_LoRA_SET)])
            self.lora_b = nn.ParameterList([Parameter(init.zeros_(empty(out_features, self.lora_r[i])))
                                            for i in range(NUM_LoRA_SET)])
            #
            # self.lora_a = Parameter(init.kaiming_uniform_(empty(lora_r, in_features), a=sqrt(5)))
            # self.lora_b = Parameter(init.zeros_(empty(out_features, lora_r)))
            self.lora_dropout = lora_dropout
            if isinstance(lora_alpha, float):
                self.lora_alpha = [lora_alpha, ] * NUM_LoRA_SET
            else:
                self.lora_alpha = lora_alpha
            self._lora_scaling = [self.lora_alpha[i] / self.lora_r[i] for i in range(NUM_LoRA_SET)]
            self.merge_lora_index = list(range(NUM_LoRA_SET))
            # for moving average
            # self.moving_avg_lora_a = []
            # self.moving_avg_lora_b = []
            for i in range(NUM_LoRA_SET):
                self.register_buffer(f'buffer_moving_avg_lora_a_{i}', self.lora_a[i].data.clone())
                self.register_buffer(f'buffer_moving_avg_lora_b_{i}', self.lora_b[i].data.clone())
                self.register_buffer(f'buffer_moving_avg_full_mat_{i}', torch.zeros((out_features, in_features)))
                # 下面这种方式不行，因为moving_avg_lora_a[0]和moving_avg_lora_a_0是不同的对象
                # self.moving_avg_lora_a.append(getattr(self, f'moving_avg_lora_a_{i}'))
                # self.moving_avg_lora_b.append(getattr(self, f'moving_avg_lora_b_{i}'))
            # # 不同lora参数统一初始化
            # for i in range(1, NUM_LoRA_SET):
            #     self.lora_a[i].data = self.lora_a[0].data.clone()
            #     self.lora_b[i].data = self.lora_b[0].data.clone()
            # 注册可学习的key参数，
            # self.lora_coeff = nn.ParameterList([Parameter(init.kaiming_uniform_(empty(64, 1), a=sqrt(5)))
            #                                     for i in range(NUM_LoRA_SET)])

            # self.transform_net = nn.Sequential(
            #     nn.Linear(768 * 2, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, 64),
            # )
            # # # 生成lora参数
            # self.generate_lora_a_a = nn.ParameterList(
            #     [Parameter(init.kaiming_uniform_(empty(1, 64), a=sqrt(5)))
            #      for i in range(NUM_LoRA_SET)])
            # self.generate_lora_a_b = nn.ParameterList(
            #     [Parameter(init.kaiming_uniform_(empty(self.lora_r[i] * in_features, 1), a=sqrt(5)))
            #      for i in range(NUM_LoRA_SET)])
            # self.generate_lora_b_a = nn.ParameterList(
            #     [Parameter(init.kaiming_uniform_(empty(2, 64), a=sqrt(5)))
            #      for i in range(NUM_LoRA_SET)])
            # self.generate_lora_b_b = nn.ParameterList(
            #     [Parameter(init.zeros_(empty(self.lora_r[i] * out_features, 2)))
            #      for i in range(NUM_LoRA_SET)])
            # for i in range(NUM_LoRA_SET):
            #     self.register_buffer(f'buffer_moving_avg_generate_lora_a_a_{i}', self.generate_lora_a_a[i].data.clone())
            #     self.register_buffer(f'buffer_moving_avg_generate_lora_a_b_{i}', self.generate_lora_a_b[i].data.clone())
            #     self.register_buffer(f'buffer_moving_avg_generate_lora_b_a_{i}', self.generate_lora_b_a[i].data.clone())
            #     self.register_buffer(f'buffer_moving_avg_generate_lora_b_b_{i}', self.generate_lora_b_b[i].data.clone())
            # # 可学习的参数mask
            # # TODO:换成了0初始化
            # self.lora_mask = Parameter(init.ones_(empty(out_features, in_features)), requires_grad=True)
            # self.lora_mask = Parameter(init.uniform_(empty(out_features, in_features)), requires_grad=True)
            # self.lora_mask_a = Parameter(init.kaiming_uniform_(empty(2, in_features), a=sqrt(5)), requires_grad=True)
            # self.lora_mask_b = Parameter(init.zeros_(empty(out_features, 2)), requires_grad=True)
            # self.lora_mask_a = Parameter(init.ones_(empty(1, in_features)), requires_grad=True)
            # self.lora_mask_b = Parameter(init.ones_(empty(out_features, 1)), requires_grad=True)
            # self.register_buffer(f'buffer_moving_avg_lora_mask_a', self.lora_mask_a.data.clone())
            # self.register_buffer(f'buffer_moving_avg_lora_mask_b', self.lora_mask_b.data.clone())
            self.lora_coeff = [1 / NUM_LoRA_SET for i in range(NUM_LoRA_SET)]

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        orig_shape = out.shape
        if self.lora_r[0] and self.lora_index >= 0:
            if self.training and self.lora_dropout > 0:
                x = dropout(x, self.lora_dropout)
            #
            actual_lora_a = self.lora_a if not self.using_moving_average else self.moving_avg_lora_a
            actual_lora_b = self.lora_b if not self.using_moving_average else self.moving_avg_lora_b
            # actual_lora_mask_a = self.lora_mask_a if not self.using_moving_average else self.moving_avg_lora_mask_a
            # actual_lora_mask_b = self.lora_mask_b if not self.using_moving_average else self.moving_avg_lora_mask_b
            if self.lora_index < 100:
                if self.test_with_full_moving_mat and not self.training:
                    residual_res = x @ (self.moving_avg_full_mat[self.lora_index].transpose(0, 1)) * self._lora_scaling[
                        self.lora_index]
                    res = out + residual_res
                else:
                    a = x @ actual_lora_a[self.lora_index].transpose(0, 1)
                    # 随机打乱lora_a的第一个通道的顺序
                    # tmp_lora_a = self.lora_a[torch.randperm(self.lora_a.shape[0]), :].transpose(0, 1)
                    # a = x @ tmp_lora_a
                    res = addmm(out.flatten(end_dim=-2), a.flatten(end_dim=-2),
                                actual_lora_b[self.lora_index].transpose(0, 1),
                                alpha=self._lora_scaling[self.lora_index]).view(out.shape)
                return res
            elif self.lora_index == 100:
                out = out.flatten(end_dim=-2)
                # coeff_list = [0.5, 0.5, -0.1]
                for i in range(NUM_LoRA_SET):
                    a = x @ actual_lora_a[i].transpose(0, 1)
                    tmp_res = (a.flatten(end_dim=-2) @ actual_lora_b[i].transpose(0, 1)).view(out.shape)
                    # out += tmp_res * self._lora_scaling[i] * coeff_list[i]
                    out += tmp_res * self._lora_scaling[i] / float(NUM_LoRA_SET)
                out = out.view(orig_shape)
            elif self.lora_index == 101:
                # 多组的AB做平均
                out = out.flatten(end_dim=-2)
                #
                mean_a = sum([actual_lora_a[i] for i in range(NUM_LoRA_SET)]) / float(NUM_LoRA_SET)
                mean_b = sum([actual_lora_b[i] for i in range(NUM_LoRA_SET)]) / float(NUM_LoRA_SET)
                a = x @ mean_a.transpose(0, 1)
                tmp_res = (a.flatten(end_dim=-2) @ mean_b.transpose(0, 1)).view(out.shape)
                out += tmp_res * self._lora_scaling[0]
                out = out.view(orig_shape)
            elif self.lora_index == 102:
                input_mean = x[0:, :, :].mean(dim=0).detach()
                input_std = x[0:, :, :].std(dim=0).detach()  # batch_size, hidden_size
                style_feat = torch.cat([input_mean, input_std], dim=-1)
                style_feat = self.transform_net(style_feat)
                normalized_style_feat = style_feat / style_feat.norm(dim=-1, keepdim=True).detach()
                normalized_lora_coeff = [self.lora_coeff[i] / self.lora_coeff[i].norm(dim=0, keepdim=True).detach()
                                         for i in range(NUM_LoRA_SET)]
                #
                coeff_cos_sim = [normalized_style_feat @ normalized_lora_coeff[i] for i in self.merge_lora_index]
                coeff_cos_sim = torch.cat(coeff_cos_sim, dim=-1)
                coeff = F.softmax(coeff_cos_sim / 0.07, dim=-1)
                # coeff = coeff_cos_sim / coeff_cos_sim.sum(dim=-1, keepdim=True)
                #
                orig_out_shape = out.shape
                out = out.flatten(end_dim=-2)
                for ind_1, ind_2 in zip(self.merge_lora_index, range(len(self.merge_lora_index))):
                    a = x @ self.moving_avg_lora_a[ind_1].transpose(0, 1)
                    tmp_res = (a.flatten(end_dim=-2) @ self.moving_avg_lora_b[ind_1].transpose(0, 1)).view(out.shape)
                    tmp_coeff = coeff[:, ind_2].unsqueeze(-1).unsqueeze(0).expand(orig_out_shape)
                    out += tmp_res * self._lora_scaling[ind_1] * tmp_coeff.reshape(out.shape)
                out = out.view(orig_out_shape)
            elif self.lora_index == 103:
                for i in self.merge_lora_index:
                    # 换成了moving average参数
                    delta_w = actual_lora_a[i].transpose(0, 1) @ actual_lora_b[i].transpose(0, 1)
                    # tmp_mask = actual_lora_mask_b @ actual_lora_mask_a
                    # final_lora_mask = torch.ones_like(tmp_mask) + tmp_mask * 100
                    # final_lora_mask = torch.sigmoid(torch.ones_like(tmp_mask) + tmp_mask * 10000)
                    final_lora_mask = torch.sigmoid(self.lora_mask)
                    # tmp_mask = self.lora_mask_b @ self.lora_mask_a
                    # final_lora_mask = torch.sigmoid(tmp_mask)
                    masked_delta_w = delta_w.detach() * final_lora_mask
                    out += (x @ masked_delta_w).view(out.shape) * self._lora_scaling[i] / float(
                        len(self.merge_lora_index))
            elif self.lora_index == 104:
                for i in range(NUM_LoRA_SET):
                    a = x @ self.moving_avg_lora_a[i].transpose(0, 1)
                    tmp_res = (a.flatten(end_dim=-2) @ self.moving_avg_lora_b[i].transpose(0, 1)).view(out.shape)
                    out += tmp_res * self._lora_scaling[i] * self.lora_coeff[i]
            elif self.lora_index == 105:
                for i in self.merge_lora_index:
                    a = x @ self.moving_avg_lora_a[i].transpose(0, 1)
                    tmp_res = (a.flatten(end_dim=-2) @ self.moving_avg_lora_b[i].transpose(0, 1)).view(out.shape)
                    out += tmp_res * self._lora_scaling[i] / float(len(self.merge_lora_index))
            elif self.lora_index == 106:
                # 前面NUM_SET个lora做平均，然后再和最后一个lora做平均
                for i in range(NUM_LoRA_SET - 1):
                    a = x @ self.moving_avg_lora_a[i].transpose(0, 1)
                    tmp_res = (a.flatten(end_dim=-2) @ self.moving_avg_lora_b[i].transpose(0, 1)).view(out.shape)
                    out += tmp_res * self._lora_scaling[i] / float(NUM_LoRA_SET - 1) / 2.0
                a = x @ self.moving_avg_lora_a[NUM_LoRA_SET - 1].transpose(0, 1)
                tmp_res = (a.flatten(end_dim=-2) @ self.moving_avg_lora_b[NUM_LoRA_SET - 1].transpose(0, 1)).view(
                    out.shape)
                out += tmp_res * self._lora_scaling[NUM_LoRA_SET - 1] / 2.0
            elif self.lora_index == 107:
                # 前面2个lora做平均，然后再和最后一个lora做平均
                for i in range(2):
                    a = x @ self.moving_avg_lora_a[i].transpose(0, 1)
                    tmp_res = (a.flatten(end_dim=-2) @ self.moving_avg_lora_b[i].transpose(0, 1)).view(out.shape)
                    out += tmp_res * self._lora_scaling[i] / float(2) * 0.5
                a = x @ self.moving_avg_lora_a[2].transpose(0, 1)
                tmp_res = (a.flatten(end_dim=-2) @ self.moving_avg_lora_b[2].transpose(0, 1)).view(out.shape)
                out += tmp_res * self._lora_scaling[2] * 0.5
            elif 200 <= self.lora_index < 300:
                actual_generate_lora_a_a = self.generate_lora_a_a if not self.using_moving_average else self.moving_avg_generate_lora_a_a
                actual_generate_lora_a_b = self.generate_lora_a_b if not self.using_moving_average else self.moving_avg_generate_lora_a_b
                actual_generate_lora_b_a = self.generate_lora_b_a if not self.using_moving_average else self.moving_avg_generate_lora_b_a
                actual_generate_lora_b_b = self.generate_lora_b_b if not self.using_moving_average else self.moving_avg_generate_lora_b_b
                if self.lora_index == 200:
                    pass
                else:
                    # lora_a是生成的
                    # tmp_index = self.lora_index - 200
                    # tmp_generate_a_a = actual_generate_lora_a_a[tmp_index]
                    # tmp_generate_a_b = actual_generate_lora_a_b[tmp_index]
                    # transform_mat = tmp_generate_a_b @ tmp_generate_a_a
                    # #
                    # input_mean = x[0:, :, :].mean(dim=0).detach()
                    # input_std = x[0:, :, :].std(dim=0).detach()  # batch_size, hidden_size
                    # style_feat = torch.cat([input_mean, input_std], dim=-1)
                    # style_feat = self.transform_net(style_feat)
                    #
                    # final_lora_a = (style_feat @ transform_mat.T).reshape(style_feat.shape[0],
                    #                                                     self.lora_r[tmp_index],
                    #                                                     self.in_features)
                    # #
                    # final_lora_a = final_lora_a.unsqueeze(0).expand(x.shape[0], -1, -1, -1).transpose(2, 3)
                    # shape_list = final_lora_a.shape
                    # final_lora_a = final_lora_a.reshape(shape_list[0] * shape_list[1], shape_list[2], shape_list[3])
                    # x_shape_list = x.shape
                    # x = x.unsqueeze(2).reshape(x_shape_list[0] * x_shape_list[1], 1, x_shape_list[2])
                    # a = torch.bmm(x, final_lora_a).reshape(x_shape_list[0], x_shape_list[1], -1)
                    # a = a.reshape(x_shape_list[0], x_shape_list[1], self.lora_r[tmp_index])
                    # res = addmm(out.flatten(end_dim=-2), a.flatten(end_dim=-2),
                    #             actual_lora_b[tmp_index].transpose(0, 1),
                    #             alpha=self._lora_scaling[tmp_index]).view(out.shape)
                    # lora_b是生成的
                    tmp_index = self.lora_index - 200
                    tmp_generate_b_a = actual_generate_lora_b_a[tmp_index]
                    tmp_generate_b_b = actual_generate_lora_b_b[tmp_index]
                    transform_mat = tmp_generate_b_b @ tmp_generate_b_a
                    input_mean = x[0:, :, :].mean(dim=0).detach()
                    input_std = x[0:, :, :].std(dim=0).detach()  # batch_size, hidden_size
                    style_feat = torch.cat([input_mean, input_std], dim=-1)
                    style_feat = self.transform_net(style_feat)
                    #
                    a = x @ actual_lora_a[tmp_index].transpose(0, 1)
                    final_lora_b = (style_feat @ transform_mat.T).reshape(style_feat.shape[0],
                                                                          self.lora_r[tmp_index],
                                                                          self.out_features)
                    ensemble_lora_b = actual_lora_b[tmp_index].transpose(0, 1).unsqueeze(0)
                    final_lora_b = ensemble_lora_b + final_lora_b
                    #
                    final_lora_b = final_lora_b.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                    delta_w = torch.bmm(a.flatten(end_dim=-2).unsqueeze(1), final_lora_b.flatten(end_dim=-3)).squeeze(1)
                    res = out + delta_w.view(out.shape) * self._lora_scaling[tmp_index]
                    return res

        return out

    def merge_lora(self):
        """
        Transform LoRA linear to normal
        """
        if not self.lora_r:
            return
        self.weight.data += (self.lora_b @ self.lora_a) * self._lora_scaling
        self.weight.requires_grad = True
        self.lora_r = 0
        del self.lora_a, self.lora_b, self.lora_dropout, self.lora_alpha, self._lora_scaling

    def merge_lora_and_reset(self, index):
        """
        Transform LoRA linear to normal
        """
        if not self.lora_r:
            return
        new_weight = self.moving_avg_lora_b[index].data @ self.moving_avg_lora_a[index].data
        self.weight.data += new_weight * self._lora_scaling[index]
        self.weight.requires_grad = False  # freeze main weights
        #
        new_a_data = Parameter(init.kaiming_uniform_(empty(self.lora_r[index], self.in_features), a=sqrt(5)),
                               requires_grad=False).to(self.weight.device)
        new_b_data = Parameter(init.zeros_(empty(self.out_features, self.lora_r[index])), requires_grad=False).to(
            self.weight.device)
        self.moving_avg_lora_a[index].data = new_a_data.data.clone()
        self.moving_avg_lora_b[index].data = new_b_data.data.clone()
        self.lora_a[index].data = new_a_data
        self.lora_b[index].data = new_b_data

    def reset_lora_parameter(self, index):
        new_a_data = Parameter(init.kaiming_uniform_(empty(self.lora_r[index], self.in_features), a=sqrt(5)),
                               requires_grad=False).to(self.weight.device)
        new_b_data = Parameter(init.zeros_(empty(self.out_features, self.lora_r[index])), requires_grad=False).to(
            self.weight.device)
        self.moving_avg_lora_a[index].data = new_a_data.data.clone()
        self.moving_avg_lora_b[index].data = new_b_data.data.clone()
        self.lora_a[index].data = new_a_data
        self.lora_b[index].data = new_b_data

    def gather_multi_lora(self):
        """
        将多组lora参数合并到一组
        :return:
        """
        pass

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if self.lora_r:
            return r + f', lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout}'
        return r

    def set_lora_index(self, lora_index):
        self.lora_index = lora_index

    def set_merge_index(self, merge_index):
        self.merge_lora_index = merge_index

    def set_lora_moving_average(self, using_moving_average):
        self.using_moving_average = using_moving_average

    def set_lora_coeff(self, lora_coeff):
        self.lora_coeff = lora_coeff

    @property
    def moving_avg_lora_a(self):
        return [getattr(self, f"buffer_moving_avg_lora_a_{i}") for i in range(NUM_LoRA_SET)]

    @property
    def moving_avg_lora_b(self):
        return [getattr(self, f"buffer_moving_avg_lora_b_{i}") for i in range(NUM_LoRA_SET)]

    @property
    def moving_avg_full_mat(self):
        return [getattr(self, f"buffer_moving_avg_full_mat_{i}") for i in range(NUM_LoRA_SET)]

    @property
    def moving_avg_generate_lora_a_a(self):
        return [getattr(self, f"buffer_moving_avg_generate_lora_a_a_{i}") for i in range(NUM_LoRA_SET)]

    @property
    def moving_avg_generate_lora_a_b(self):
        return [getattr(self, f"buffer_moving_avg_generate_lora_a_b_{i}") for i in range(NUM_LoRA_SET)]

    @property
    def moving_avg_generate_lora_b_a(self):
        return [getattr(self, f"buffer_moving_avg_generate_lora_b_a_{i}") for i in range(NUM_LoRA_SET)]

    @property
    def moving_avg_generate_lora_b_b(self):
        return [getattr(self, f"buffer_moving_avg_generate_lora_b_b_{i}") for i in range(NUM_LoRA_SET)]

    @property
    def moving_avg_lora_mask_a(self):
        return getattr(self, f"buffer_moving_avg_lora_mask_a")

    @property
    def moving_avg_lora_mask_b(self):
        return getattr(self, f"buffer_moving_avg_lora_mask_b")

    def lora_moving_average(self, lambda_last=None, lambda_current=None, index=None, ):
        # 需要设置data属性
        if index is not None:
            tmp_list = [index, ]
        else:
            tmp_list = list(range(NUM_LoRA_SET))
        for i in tmp_list:
            self.moving_avg_lora_a[i].data = self.lora_a[i].data * lambda_current + self.moving_avg_lora_a[
                i].data * lambda_last
            self.moving_avg_lora_b[i].data = self.lora_b[i].data * lambda_current + self.moving_avg_lora_b[
                i].data * lambda_last
            tmp_mat = self.lora_a[i].T @ self.lora_b[i].T
            self.moving_avg_full_mat[i].data = tmp_mat.data * lambda_current + self.moving_avg_full_mat[
                i].data * lambda_last
            #
            # self.moving_avg_generate_lora_a_a[i] = self.generate_lora_a_a[i].data * lambda_current + \
            #                                        self.moving_avg_generate_lora_a_a[i].data * lambda_last
            # self.moving_avg_generate_lora_a_b[i] = self.generate_lora_a_b[i].data * lambda_current + \
            #                                        self.moving_avg_generate_lora_a_b[i].data * lambda_last
            # #
            # self.moving_avg_generate_lora_b_a[i] = self.generate_lora_b_a[i].data * lambda_current + \
            #                                        self.moving_avg_generate_lora_b_a[i].data * lambda_last
            # self.moving_avg_generate_lora_b_b[i] = self.generate_lora_b_b[i].data * lambda_current + \
            #                                        self.moving_avg_generate_lora_b_b[i].data * lambda_last
            # #
            # self.moving_avg_lora_mask_a.data = self.lora_mask_a.data * lambda_current + self.moving_avg_lora_mask_a.data * lambda_last
            # self.moving_avg_lora_mask_b.data = self.lora_mask_b.data * lambda_current + self.moving_avg_lora_mask_b.data * lambda_last

    def lora_diversity_loss(self, lora_index, detach_first=False):
        src_index = lora_index[0]
        tgt_index = lora_index[1]
        #
        # 对A做正交化损失
        div_loss = self.lora_a[src_index] @ self.lora_a[tgt_index].T
        div_loss = torch.sum(div_loss ** 2)
        return div_loss
        # 单独对lora_a 和 lora_b做余弦损失
        # norm_a = self.lora_a[src_index] / (torch.norm(self.lora_a[src_index], p='fro', keepdim=True) + 1e-8)
        # norm_a_tgt = self.moving_avg_lora_a[tgt_index] / (
        #         torch.norm(self.moving_avg_lora_a[tgt_index], p='fro', keepdim=True) + 1e-8)
        # loss = torch.sum(norm_a * norm_a_tgt.detach())
        # norm_b = self.lora_b[src_index] / (torch.norm(self.lora_b[src_index], p='fro', keepdim=True) + 1e-8)
        # norm_b_tgt = self.moving_avg_lora_b[tgt_index] / (
        #         torch.norm(self.moving_avg_lora_b[tgt_index], p='fro', keepdim=True) + 1e-8)
        # loss += torch.sum(norm_b * norm_b_tgt.detach())
        # return loss
        ########################################
        # 之前使用的版本
        # loss = 0
        # lora_len = len(lora_index)
        # # original_param = self.weight.data.clone().detach()
        # for i in range(lora_len - 1):
        #     for j in range(i + 1, lora_len):
        #         param_mat_i = self.lora_a[lora_index[i]].transpose(0, 1) @ self.lora_b[lora_index[i]].transpose(0, 1)
        #         param_mat_j = self.lora_a[lora_index[j]].transpose(0, 1) @ self.lora_b[lora_index[j]].transpose(0, 1)
        #         #
        #         # loss += - torch.norm(param_mat_i - param_mat_j, p='fro')
        #         #
        #         param_mat_i = param_mat_i.view(-1)  # + original_param.view(-1)
        #         param_mat_j = param_mat_j.view(-1)  # + original_param.view(-1)
        #         param_mat_i = param_mat_i / (torch.norm(param_mat_i, p='fro', keepdim=True) + 1e-8)
        #         param_mat_j = param_mat_j / (torch.norm(param_mat_j, p='fro', keepdim=True) + 1e-8)
        #         if detach_first:
        #         #             loss += torch.sum(param_mat_i.detach() * param_mat_j)
        #         #         else:
        #         #             loss += torch.sum(param_mat_i * param_mat_j)
        ###################################
        # i = lora_index[0]
        # j = lora_index[1]
        # param_mat_i = self.lora_a[lora_index[i]].transpose(0, 1) @ self.lora_b[lora_index[i]].transpose(0, 1)
        # param_mat_j = self.lora_a[lora_index[j]].transpose(0, 1) @ self.lora_b[lora_index[j]].transpose(0, 1)
        # param_mat_i = param_mat_i.view(-1)
        # param_mat_j = param_mat_j.view(-1)
        # param_mat_i = param_mat_i / (torch.norm(param_mat_i, p='fro', keepdim=True) + 1e-8)
        # param_mat_j = param_mat_j / (torch.norm(param_mat_j, p='fro', keepdim=True) + 1e-8)
        # if detach_first:

        #     loss += torch.sum(param_mat_i.detach() * param_mat_j)
        # else:
        #     loss += torch.sum(param_mat_i * param_mat_j)
        return loss

    def copy_lora_parameters(self, source_index, target_index, only_copy_to_running=False, exact=False,
                             lambda_current=1.0):
        def _copy_lora_parameters(source, target):
            target.data = source.data.clone() * lambda_current + target.data.clone() * (1 - lambda_current)

        if only_copy_to_running:
            _copy_lora_parameters(self.moving_avg_lora_a[source_index], self.lora_a[target_index])
            _copy_lora_parameters(self.moving_avg_lora_b[source_index], self.lora_b[target_index])
        elif exact:
            _copy_lora_parameters(self.lora_a[source_index], self.lora_a[target_index])
            _copy_lora_parameters(self.lora_b[source_index], self.lora_b[target_index])
            _copy_lora_parameters(self.moving_avg_lora_a[source_index], self.moving_avg_lora_a[target_index])
            _copy_lora_parameters(self.moving_avg_lora_b[source_index], self.moving_avg_lora_b[target_index])
        else:
            _copy_lora_parameters(self.moving_avg_lora_a[source_index], self.lora_a[target_index])
            _copy_lora_parameters(self.moving_avg_lora_b[source_index], self.lora_b[target_index])
            _copy_lora_parameters(self.moving_avg_lora_a[source_index], self.moving_avg_lora_a[target_index])
            _copy_lora_parameters(self.moving_avg_lora_b[source_index], self.moving_avg_lora_b[target_index])


def _update_lora(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        q_w, k_w, v_w = state_dict.pop(prefix + 'in_proj_weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'in_proj_bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
        state_dict[prefix + 'q_proj.bias'] = q_b
        state_dict[prefix + 'k_proj.bias'] = k_b
        state_dict[prefix + 'v_proj.bias'] = v_b
    elif prefix + 'qkv_proj.weight' in state_dict:  # transform packed projection
        q_w, k_w, v_w = state_dict.pop(prefix + 'qkv_proj.weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'qkv_proj.bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
        state_dict[prefix + 'q_proj.bias'] = q_b
        state_dict[prefix + 'k_proj.bias'] = k_b
        state_dict[prefix + 'v_proj.bias'] = v_b


def _update_packed(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        state_dict[prefix + 'qkv_proj.weight'] = state_dict.pop(prefix + 'in_proj_weight')
        state_dict[prefix + 'qkv_proj.bias'] = state_dict.pop(prefix + 'in_proj_bias')
    elif prefix + 'q_proj.weight' in state_dict:  # transform unpacked projection
        q_w = state_dict.pop(prefix + 'q_proj.weight')
        k_w = state_dict.pop(prefix + 'k_proj.weight')
        v_w = state_dict.pop(prefix + 'v_proj.weight')
        q_b = state_dict.pop(prefix + 'q_proj.bias')
        k_b = state_dict.pop(prefix + 'k_proj.bias')
        v_b = state_dict.pop(prefix + 'v_proj.bias')
        state_dict[prefix + 'qkv_proj.weight'] = cat([q_w, k_w, v_w])
        state_dict[prefix + 'qkv_proj.bias'] = cat([q_b, k_b, v_b])


class LORAMultiheadAttention(nn.Module):
    """
    LoRA wrapped Multi-Head Attention
    """

    def __init__(self, embed_dim, num_heads, dropout=0., lora_r: int = 0, lora_alpha: float = 1.,
                 lora_dropout: float = 0., test_with_full_moving_mat=False, use_lora_for_k=False):
        """
        :param embed_dim: the size of each embedding vector
        :param num_heads: number of heads
        :param dropout: attention dropout
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout
        """
        assert not embed_dim % num_heads, 'embed_dim must be divisible by num_heads'
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.lora_r = lora_r
        self._scale = 1 / sqrt(self.head_dim)
        self.use_lora_for_k = use_lora_for_k

        if lora_r:
            self.q_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 test_with_full_moving_mat=test_with_full_moving_mat)
            if self.use_lora_for_k:
                # 如果要换k_proj，后面的set函数也要换
                self.k_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha,
                                     lora_dropout=lora_dropout,
                                     test_with_full_moving_mat=test_with_full_moving_mat)
            else:
                self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 test_with_full_moving_mat=test_with_full_moving_mat)
            self._register_load_state_dict_pre_hook(_update_lora)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
            self._register_load_state_dict_pre_hook(_update_packed)
        # self.o_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, attn_mask, need_weights: bool = True):
        tgt_len, bsz, _ = q.shape
        src_len, _, _ = k.shape

        # do projection
        if self.lora_r:
            # print('gogogo')
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)
        else:  # self-attention
            q, k, v = self.qkv_proj(q).chunk(3, dim=-1)

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).permute(1, 2, 0)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if attn_mask is not None:
            a = attn_mask + self._scale * q @ k
        else:
            a = self._scale * q @ k

        a = softmax(a, dim=-1)
        # print(q.shape, k.shape, v.shape, a.shape)
        # print('lora input')
        # exit(0)
        if self.training and self.dropout:
            a = F.dropout(a, self.dropout)

        o = bmm(a, v).transpose(0, 1).contiguous().view(tgt_len * bsz, self.embed_dim)
        o = self.o_proj(o).view(tgt_len, bsz, -1)  # switch dimensions back
        if need_weights:
            a = a.view(bsz, self.num_heads, tgt_len, src_len)
            a = a.sum(dim=1) / self.num_heads
            return o, a
        else:
            return o, None

    def merge_lora_and_reset(self, index):
        self.q_proj.merge_lora_and_reset(index)
        if self.use_lora_for_k:
            self.k_proj.merge_lora_and_reset(index)
        self.v_proj.merge_lora_and_reset(index)
        # self.o_proj.merge_lora_and_reset(index)

    def reset_lora_parameter(self, index):
        self.q_proj.reset_lora_parameter(index)
        if self.use_lora_for_k:
            self.k_proj.reset_lora_parameter(index)
        self.v_proj.reset_lora_parameter(index)
        # self.o_proj.reset_lora_parameter(index)

    def lora_diversity_loss(self, lora_index, detach_first=False):
        loss = self.q_proj.lora_diversity_loss(lora_index, detach_first) + \
               self.v_proj.lora_diversity_loss(lora_index, detach_first)
        if self.use_lora_for_k:
            loss += self.k_proj.lora_diversity_loss(lora_index, detach_first)
        return loss

    def copy_lora_parameters(self, source_index, target_index, only_copy_to_running=False, exact=False,
                             lambda_current=1.0, ):
        self.q_proj.copy_lora_parameters(source_index, target_index, only_copy_to_running, exact, lambda_current)
        if self.use_lora_for_k:
            self.k_proj.copy_lora_parameters(source_index, target_index, only_copy_to_running, exact, lambda_current)
        self.v_proj.copy_lora_parameters(source_index, target_index, only_copy_to_running, exact, lambda_current)
        # self.o_proj.copy_lora_parameters(source_index, target_index)

    def set_lora_index(self, lora_index):
        if self.lora_r:
            self.q_proj.set_lora_index(lora_index)
            if self.use_lora_for_k:
                self.k_proj.set_lora_index(lora_index)
            self.v_proj.set_lora_index(lora_index)
            # self.o_proj.set_lora_index(lora_index)
        else:
            self.qkv_proj.set_lora_index(lora_index)

    def set_merge_index(self, merge_index):
        if self.lora_r:
            self.q_proj.set_merge_index(merge_index)
            if self.use_lora_for_k:
                self.k_proj.set_merge_index(merge_index)
            self.v_proj.set_merge_index(merge_index)
            # self.o_proj.set_merge_index(merge_index)
        else:
            self.qkv_proj.set_merge_index(merge_index)

    def set_lora_coeff(self, lora_coeff):
        if self.lora_r:
            self.q_proj.set_lora_coeff(lora_coeff)
            if self.use_lora_for_k:
                self.k_proj.set_lora_coeff(lora_coeff)
            self.v_proj.set_lora_coeff(lora_coeff)
            # self.o_proj.set_lora_coeff(lora_coeff)
        else:
            self.qkv_proj.set_lora_coeff(lora_coeff)

    def set_lora_moving_average(self, using_moving_average=False):
        if self.lora_r:
            self.q_proj.set_lora_moving_average(using_moving_average)
            if self.use_lora_for_k:
                self.k_proj.set_lora_moving_average(using_moving_average)
            self.v_proj.set_lora_moving_average(using_moving_average)
            # self.o_proj.set_lora_moving_average(using_moving_average)
        else:
            self.qkv_proj.set_lora_moving_average(using_moving_average)

    def lora_moving_average(self, lambda_last=None, lambda_current=None, index=None):
        if self.lora_r:
            self.q_proj.lora_moving_average(lambda_last, lambda_current, index=None)
            if self.use_lora_for_k:
                self.k_proj.lora_moving_average(lambda_last, lambda_current, index=None)
            self.v_proj.lora_moving_average(lambda_last, lambda_current, index=None)
            # self.o_proj.lora_moving_average(lambda_last, lambda_current)
        else:
            self.qkv_proj.lora_moving_average(lambda_last, lambda_current)


def LoRA_ViT(old_model, lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.,
             lora_other_linear: bool = False, lora_layers=None, test_with_full_moving_mat=False,
             use_lora_for_k=False):
    """
    replace all multihead attention with LoRA multihead attention, and copy the weights to the new model
    :param old_model:
    :return:
    """
    model = deepcopy(old_model)

    # for name, module in old_model.named_modules():
    #     if isinstance(module, nn.MultiheadAttention):
    #         setattr(model, name, LORAMultiheadAttention(module.embed_dim, module.num_heads, module.dropout,
    #                                                     lora_r=lora_r, lora_alpha=lora_alpha,
    #                                                     lora_dropout=lora_dropout))
    #         getattr(model, name).load_state_dict(module.state_dict(), strict=False)
    #
    #
    def change_module(module, prefix):
        actual_prefix = prefix + '.' if prefix else ''
        for name, child in module.named_children():
            if isinstance(child, nn.MultiheadAttention):
                if lora_layers is not None:
                    if not any(["resblocks.{}.".format(num) in actual_prefix for num in lora_layers]):
                        continue
                setattr(module, name, LORAMultiheadAttention(child.embed_dim, child.num_heads, child.dropout,
                                                             lora_r=lora_r, lora_alpha=lora_alpha,
                                                             lora_dropout=lora_dropout,
                                                             test_with_full_moving_mat=test_with_full_moving_mat,
                                                             use_lora_for_k=use_lora_for_k))
            elif lora_other_linear:
                if isinstance(child, nn.Linear):
                    setattr(module, name, Linear(child.in_features, child.out_features, lora_r=lora_r,
                                                 lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                 test_with_full_moving_mat=test_with_full_moving_mat,
                                                 use_lora_for_k=use_lora_for_k))
                else:
                    change_module(child, actual_prefix + name)
            else:
                change_module(child, actual_prefix + name)

    change_module(model, '')
    model.load_state_dict(old_model.state_dict(), strict=False)
    return model


def set_lora_index(model, lora_index):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                # print('set {} {}'.format(name,lora_index))
                child.set_lora_index(lora_index)
            else:
                # print('name {}, type {}'.format(name, type(child)))
                change_module(child)

    change_module(model)


def set_merge_index(model, merge_index):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.set_merge_index(merge_index)
            else:
                change_module(child)

    #
    change_module(model)


def set_lora_coeff(model, lora_coeff):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.set_lora_coeff(lora_coeff)
            else:
                change_module(child)

    #
    change_module(model)


def set_lora_moving_average(model, using_moving_average=False):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.set_lora_moving_average(using_moving_average)
            else:
                change_module(child)

    #
    change_module(model)


def lora_moving_average(model, lambda_last=None, lambda_current=None, index=None):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.lora_moving_average(lambda_last=lambda_last, lambda_current=lambda_current, index=None)
            else:
                change_module(child)

    #
    change_module(model)


def lora_diversity_loss(model, lora_index=None, detach_first=False):
    def change_module(module, loss_list):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                loss_list.append(child.lora_diversity_loss(lora_index, detach_first))
            else:
                change_module(child, loss_list)

    #
    final_loss_list = []
    change_module(model, final_loss_list)
    return sum(final_loss_list)


def copy_lora_parameters(model, source_index, target_index, only_copy_to_running=False, exact=False,
                         lambda_current=1.0):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.copy_lora_parameters(source_index, target_index, only_copy_to_running, exact, lambda_current)
            else:
                change_module(child)

    #
    change_module(model)


def merge_lora_and_reset(model, index):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.merge_lora_and_reset(index)
            else:
                change_module(child)

    #
    change_module(model)


def reset_lora_parameter(model, index):
    def change_module(module):
        for name, child in module.named_children():
            if isinstance(child, (LORAMultiheadAttention, Linear)):
                child.reset_lora_parameter(index)
            else:
                change_module(child)

    #
    change_module(model)
