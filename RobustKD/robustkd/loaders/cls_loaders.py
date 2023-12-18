import os
import os.path
from PIL import Image
import random
import torch
from fastda.loaders import DATASETS
from mmcls.datasets import BaseDataset
from .pipelines.pipelines import Compose
import copy
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def pil_loader(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert("RGB")
    return img


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [' '.join(x.split(' ')[:-1]) for x in f.readlines()]  # 允许路径中存在空格
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[-1].strip()  # 假设最后一个都是标签，前面路径中可能存在空格
            label_list.append(int(label))
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


def make_oda_dataset_fromlist(image_list, source_classes, include_unknown):
    with open(image_list) as f:
        image_index = []
        label_list = []
        for ind, x in enumerate(f.readlines()):
            split_results = x.split(' ')
            label = split_results[1].strip()
            if int(label) in source_classes:
                image_index.append(split_results[0])
                label_list.append(int(label))
            else:
                if include_unknown:
                    image_index.append(split_results[0])
                    label_list.append(len(source_classes))
    return image_index, label_list


@DATASETS.register_module(name='ssda_cls_dataset')
class SSDA_CLS_Datasets(BaseDataset):
    def __init__(self, data_root, name, split, shot='', pipeline=None, min_len=0, label_transform=None,
                 shot_per_class=None, ratio_per_class=None):
        name_split = name.split('_')
        task = name_split[0]
        dataset = name_split[1]
        img_root = os.path.join(data_root, task)
        if shot != '':
            shot = '_' + str(shot)
        # +表示多个域的数据集同时使用
        if "+" in dataset:
            sub_dataset_name = dataset.split('+')
            image_list = []
            for sub_dataset in sub_dataset_name:
                tmp_image_path = os.path.join(data_root, 'txt', task, split + '_images_' + sub_dataset + shot + '.txt')
                image_list.append(tmp_image_path)
        else:
            image_list = os.path.join(data_root, 'txt', task, split + '_images_' + dataset + shot + '.txt')
        #
        self.min_len = min_len
        # for get_classes function, new version of mmcls,
        self.ann_file = image_list
        #
        self.label_transform_list = label_transform
        self.label_transform = label_transform
        self.shot_per_class = shot_per_class
        self.ratio_per_class = ratio_per_class
        self.class_names = None
        super(SSDA_CLS_Datasets, self).__init__(data_prefix=img_root, pipeline=pipeline, ann_file=image_list)
        #
        self.name = name
        self.split = split

    def load_annotations(self):
        domain_labels = []
        if isinstance(self.ann_file, list):
            imgs = []
            labels = []
            for domain_ind, ann_file in enumerate(self.ann_file):
                img_index, label_list = make_dataset_fromlist(ann_file)
                imgs.extend(img_index)
                labels.extend(label_list)
                domain_labels.extend([domain_ind] * len(img_index))
        else:
            imgs, labels = make_dataset_fromlist(self.ann_file)
            domain_labels.extend([0] * len(labels))
        # 设置最大类别
        self.n_classes = max(labels) + 1
        #
        #
        repeat_num = int(float(self.min_len) / len(imgs)) + 1
        imgs = imgs * repeat_num
        labels = labels * repeat_num
        domain_labels = domain_labels * repeat_num  #
        # 根据 shot_per_class 采样，只针对训练集
        if self.shot_per_class is not None:
            assert self.ratio_per_class is None, 'ratio_per_class should not be specified'
            new_imgs, new_labels, new_domain_labels = [], [], []
            for i in range(self.n_classes):
                tmp_inds = [ind for ind, label in enumerate(labels) if label == i]
                tmp_inds = random.sample(tmp_inds, self.shot_per_class)
                new_imgs.extend([imgs[ind] for ind in tmp_inds])
                new_labels.extend([labels[ind] for ind in tmp_inds])
                new_domain_labels.extend([domain_labels[ind] for ind in tmp_inds])
            imgs, labels, domain_labels = new_imgs, new_labels, new_domain_labels
        # 根据 ratio_per_class 采样，只针对训练集
        if self.ratio_per_class is not None:
            assert self.shot_per_class is None, 'shot_per_class should not be specified'
            new_imgs, new_labels, new_domain_labels = [], [], []
            for i in range(self.n_classes):
                tmp_inds = [ind for ind, label in enumerate(labels) if label == i]
                tmp_num = int(len(tmp_inds) * self.ratio_per_class)
                tmp_inds = random.sample(tmp_inds, tmp_num)
                new_imgs.extend([imgs[ind] for ind in tmp_inds])
                new_labels.extend([labels[ind] for ind in tmp_inds])
                new_domain_labels.extend([domain_labels[ind] for ind in tmp_inds])
            imgs, labels, domain_labels = new_imgs, new_labels, new_domain_labels
        #
        class_names = {}
        # 根据label_transform处理label的映射关系
        if self.label_transform is not None:
            if isinstance(self.label_transform, list):
                assert len(
                    self.label_transform) <= self.n_classes, 'label_transform list {} is too long for {} classes, {}'.format(
                    len(self.label_transform), self.n_classes, self.ann_file)
                new_label_transform = {}
                current_ind = 0
                for i in range(self.n_classes):
                    if i in self.label_transform:
                        new_label_transform[i] = current_ind
                        current_ind += 1
                    else:
                        new_label_transform[i] = None
                self.label_transform = new_label_transform
            new_imgs, new_labels, new_domain_labels = [], [], []
            for img, label, domain_label in zip(imgs, labels, domain_labels):
                ##################
                if label not in class_names:
                    tmp_class_name = img.split('/')[-2]
                    class_names[label] = tmp_class_name
                ##################
                new_label = self.label_transform[label]
                if new_label is not None:
                    new_imgs.append(img)
                    new_labels.append(new_label)
                    new_domain_labels.append(domain_label)
        else:
            new_imgs, new_labels, new_domain_labels = imgs, labels, domain_labels
            #
            for img, label, domain_label in zip(imgs, labels, domain_labels):
                ##################
                if label not in class_names:
                    tmp_class_name = img.split('/')[-2]
                    class_names[label] = tmp_class_name
        ################################
        existed_class_ind = list(class_names.keys())
        existed_class_ind.sort()
        self.class_names = [class_names[ind] for ind in existed_class_ind]
        ################################
        data_infos = []
        for ind, (img, label, domain_label) in enumerate(zip(new_imgs, new_labels, new_domain_labels)):
            info = {'img_prefix': self.data_prefix, 'img_info': {'filename': img}, 'gt_label': label,
                    'domain_label': domain_label, 'image_ind': ind}
            data_infos.append(info)

        # self.n_classes = len(class_names)
        return data_infos

    def get_classes(self, classes=None):
        # class_names = []
        # if isinstance(self.ann_file, list):
        #     tmp_ann_file = self.ann_file[0]
        # else:
        #     tmp_ann_file = self.ann_file
        # with open(tmp_ann_file, 'rb') as f:
        #     for line in f.readlines():
        #         line = line.decode()
        #         tmp_res = line.strip().split(' ')
        #         tmp_path = ' '.join(tmp_res[0:-1])
        #         # print(tmp_res[0],tmp_res[1])
        #         tmp_ind = int(tmp_res[-1])
        #         tmp_name = tmp_path.split('/')[1]
        #         # if tmp_ind == len(class_names):  # 这种只适用于有序的类别
        #         #    class_names.append(tmp_name)
        #         if tmp_name not in class_names:
        #             class_names.append(tmp_name)
        # class_names.sort()
        #
        return self.class_names


@DATASETS.register_module(name='ssda_cls_double_dataset')
class SSDA_CLS_Double_Datasets(SSDA_CLS_Datasets):
    def __init__(self, data_root, name, split, shot='', pipeline=None, pipeline2=None, min_len=0, label_transform=None,
                 shot_per_class=None, ratio_per_class=None):
        super(SSDA_CLS_Double_Datasets, self).__init__(data_root, name, split, shot, pipeline, min_len, label_transform,
                                                       shot_per_class, ratio_per_class)
        self.pipeline_2 = Compose(pipeline2)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        aug_data_1 = self.pipeline(results)
        aug_data_2 = self.pipeline_2(results)
        return (aug_data_1, aug_data_2)


@DATASETS.register_module(name='ssda_cls_triple_dataset')
class SSDA_CLS_Triple_Datasets(SSDA_CLS_Datasets):
    def __init__(self, data_root, name, split, shot='', pipeline=None, pipeline2=None, min_len=0, repeat_pipeline=2,
                 label_transform=None, shot_per_class=None, ratio_per_class=None):
        super(SSDA_CLS_Triple_Datasets, self).__init__(data_root, name, split, shot, pipeline, min_len, label_transform,
                                                       shot_per_class, ratio_per_class)
        self.pipeline_2 = Compose(pipeline2)
        self.repeat_pipeline = repeat_pipeline
        assert self.repeat_pipeline in [1, 2], 'wrong type of repeat_pipeline'

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        aug_data_1 = self.pipeline(results)
        aug_data_2 = self.pipeline_2(results)
        if self.repeat_pipeline == 2:
            aug_data_3 = self.pipeline_2(results)
        else:
            aug_data_3 = self.pipeline(results)
        return (aug_data_1, aug_data_2, aug_data_3)


@DATASETS.register_module(name='ssda_cross_domain_dataset')
class CrossDomainSampleDataset(object):
    def __init__(self, data_root, name, num_episode_classes, num_support, num_query, shot='',
                 pipeline=None, min_len=0, sample_mode='episode',
                 same_sample=(1, 1), unlabeled_target_name='unlabeled_target', return_target_data=True):
        name_split = name.split('_')
        assert len(name_split) == 3, 'the len of cross domain sample dataset name should be 3'
        task = name_split[0]
        source_dataset = name_split[1]
        target_dataset = name_split[2]
        img_root = os.path.join(data_root, task)
        if shot != '':
            shot = '_' + str(shot)
        #
        src_dataset = self._process_dir(data_root, task, source_dataset, 'labeled_source', '', min_len)
        tgt_labeled_dataset = self._process_dir(data_root, task, target_dataset, 'labeled_target', shot, min_len)
        tgt_unlabeled_dataset = self._process_dir(data_root, task, target_dataset, unlabeled_target_name, shot, min_len)
        #
        self.src_imgs, self.src_labels, self.src_label_to_index = src_dataset
        self.tgt_labeled_imgs, self.tgt_labeled_labels, self.tgt_labeled_label_to_index = tgt_labeled_dataset
        self.tgt_unlabeled_imgs, self.tgt_unlabeled_labels, self.tgt_unlabeled_label_to_index = tgt_unlabeled_dataset
        #
        self.transform = Compose(pipeline)
        self.loader = pil_loader
        self.root = img_root
        self.name = name
        self.n_classes = max(self.src_labels) + 1
        self.label_ids = list(range(self.n_classes))
        print('there is {} classes in dataset'.format(self.n_classes))
        #
        self.num_episode_cls = num_episode_classes
        self.num_support_per_cls = num_support
        self.num_query_per_cls = num_query
        #
        self.sample_mode = sample_mode
        self.same_sample = same_sample
        self.return_target_data = return_target_data
        # print('sample {}'.format(same_sample))
        # exit(0)

    def _process_dir(self, root, task, dataset_name, split, shot, min_len):
        #
        source_image_list = os.path.join(root, 'txt', task, split + '_images_' + dataset_name + shot + '.txt')
        imgs, labels = make_dataset_fromlist(source_image_list)
        #
        repeat_num = int(float(min_len) / len(imgs)) + 1
        imgs = imgs * repeat_num
        labels = labels * repeat_num
        #
        label_to_index = {}
        for idx, label in enumerate(labels):
            if label not in label_to_index:
                label_to_index[label] = []
            label_to_index[label].append(idx)
        return imgs, labels, label_to_index
        #

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """
        cls_inds = random.choices(self.label_ids, k=self.num_episode_cls * 2, )
        # labeled_inds = cls_inds[0:self.num_episode_cls]

        src_support, src_query = self._sample_specific_cls(cls_inds, self.src_label_to_index)
        tgt_labeled_support, tgt_labeled_query = self._sample_specific_cls(cls_inds, self.tgt_labeled_label_to_index)
        # TODO: 去掉重新采样类别，保持一致
        # cls_inds = random.sample(self.label_ids, self.num_episode_cls * 2)
        tgt_unlabeled_support, tgt_unlabeled_query = self._sample_specific_cls(cls_inds,
                                                                               self.tgt_unlabeled_label_to_index)

        return src_support, src_query, tgt_labeled_support, tgt_labeled_query, tgt_unlabeled_support, tgt_unlabeled_query

    def _sample__diff_cls(self):
        cls_inds = random.sample(self.label_ids, self.num_episode_cls)
        src_support, src_query = self._sample_specific_cls(cls_inds, self.src_label_to_index)
        #
        if self.same_sample[0] == 1:
            pass
        else:
            cls_inds = random.sample(self.label_ids, self.num_episode_cls)
        tgt_labeled_support, tgt_labeled_query = self._sample_specific_cls(cls_inds, self.tgt_labeled_label_to_index)
        #
        if self.same_sample[1] == 1:
            pass
        else:
            cls_inds = random.sample(self.label_ids, self.num_episode_cls)
        tgt_unlabeled_support, tgt_unlabeled_query = self._sample_specific_cls(cls_inds,
                                                                               self.tgt_unlabeled_label_to_index)
        return src_support, src_query, tgt_labeled_support, tgt_labeled_query, tgt_unlabeled_support, tgt_unlabeled_query

    def _CreateTensorData(self, data_sample, img_list, label_list, ):
        tmp_res = []
        for global_ind, task_ind in data_sample:
            tmp_data = {}
            tmp_data['img_prefix'] = self.root
            tmp_data['img_info'] = {'filename': img_list[global_ind]}
            tmp_data['gt_label'] = label_list[global_ind]
            # tmp_data['task_label'] = task_ind
            tmp_data = self.transform(tmp_data)
            tmp_res.append(tmp_data)
        #
        img_metas = []
        img_list = []
        label_list = []
        for i in range(len(tmp_res)):
            img_metas.append(tmp_res[i]['img_metas'])
            img_list.append(tmp_res[i]['img'])
            label_list.append(tmp_res[i]['gt_label'])
        final_result = {
            'img_metas': img_metas,
            'img': torch.stack(img_list, dim=0),
            'gt_label': torch.stack(label_list, dim=0)
        }

        return final_result

    def __getitem__(self, index):
        if self.sample_mode == 'episode':
            res = self._sample_episode()
        elif self.sample_mode == 'episode_diff_cls':
            res = self._sample__diff_cls()
        else:
            raise RuntimeError('unknown sample mode {}'.format(self.sample_mode))
        # print('len src {}'.format(len(self.src_imgs)))
        src_support_res = self._CreateTensorData(res[0], self.src_imgs, self.src_labels)
        src_query_res = self._CreateTensorData(res[1], self.src_imgs, self.src_labels)
        #
        # print('len tgt1 {}'.format(len(self.src_imgs)))
        tgt_labeled_support_res = self._CreateTensorData(res[2], self.tgt_labeled_imgs, self.tgt_labeled_labels)
        tgt_labeled_query_res = self._CreateTensorData(res[3], self.tgt_labeled_imgs, self.tgt_labeled_labels)
        #
        output_data = {
            'src_support': src_support_res,
            'src_query': src_query_res,
            'tgt_labeled_support': tgt_labeled_support_res,
            'tgt_labeled_query': tgt_labeled_query_res,

        }
        if self.return_target_data:
            tgt_unlabeled_support_res = self._CreateTensorData(res[4], self.tgt_unlabeled_imgs,
                                                               self.tgt_unlabeled_labels)
            tgt_unlabeled_query_res = self._CreateTensorData(res[5], self.tgt_unlabeled_imgs, self.tgt_unlabeled_labels)
            target_data = {
                'tgt_unlabeled_support': tgt_unlabeled_support_res,
                'tgt_unlabeled_query': tgt_unlabeled_query_res,
            }
            output_data.update(target_data)

        return output_data

    def __len__(self):
        min_img_num = min([len(self.src_imgs), len(self.tgt_labeled_imgs), len(self.tgt_unlabeled_imgs)])
        return int(min_img_num / (self.num_episode_cls * (self.num_query_per_cls + self.num_support_per_cls)))

    def _sample_specific_cls(self, cls_inds, label_to_index):
        support_list = []
        query_list = []
        for task_cls_ind, cls_ind in enumerate(cls_inds):
            len_per_cls = self.num_query_per_cls + self.num_support_per_cls
            all_ids = random.sample(label_to_index[cls_ind], len_per_cls)
            #
            query_ind = all_ids[:self.num_query_per_cls]
            support_ind = all_ids[self.num_query_per_cls:]
            task_inds = [task_cls_ind, ] * len_per_cls
            #
            query_list += [(global_ind, task_ind) for global_ind, task_ind in zip(query_ind, task_inds)]
            support_list += [(global_ind, task_ind) for global_ind, task_ind in zip(support_ind, task_inds)]
        return support_list, query_list


@DATASETS.register_module(name='single_domain_episode_dataset')
class SingleDomainEpisodeDataset(SSDA_CLS_Datasets):
    def __init__(self, data_root, name, split, num_episode_classes, num_sample_per_class, shot='', pipeline=None,
                 pipeline2=None, min_len=0):
        super(SingleDomainEpisodeDataset, self).__init__(data_root, name, split, shot, pipeline, min_len)
        self.num_episode_classes = num_episode_classes
        self.num_sample_per_class = num_sample_per_class
        if pipeline2 is not None:
            self.pipeline_2 = Compose(pipeline2)
        else:
            self.pipeline_2 = None
        self.label_ids = list(range(self.n_classes))
        self.label_to_index = self._get_label_to_index()

    def __getitem__(self, idx):
        all_inds = []
        cls_inds = random.sample(self.label_ids, self.num_episode_classes)
        for task_cls_ind, cls_ind in enumerate(cls_inds):
            cls_ids = random.sample(self.label_to_index[cls_ind], self.num_sample_per_class)
            all_inds.extend(cls_ids)
        #
        tmp_res = []
        tmp_res_2 = []
        for global_ind in all_inds:
            tmp_data = {}
            tmp_data['img_prefix'] = self.data_prefix
            tmp_data['img_info'] = {'filename': self.data_infos[global_ind]['img_info']['filename']}
            tmp_data['gt_label'] = self.data_infos[global_ind]['gt_label']
            aug_data_1 = self.pipeline(tmp_data)
            tmp_res.append(aug_data_1)
            if self.pipeline_2 is not None:
                aug_data_2 = self.pipeline_2(tmp_data)
                tmp_res_2.append(aug_data_2)

        #
        def _process_res(transformed_res):
            img_metas = []
            img_list = []
            label_list = []
            for i in range(len(transformed_res)):
                img_metas.append(transformed_res[i]['img_metas'])
                img_list.append(transformed_res[i]['img'])
                label_list.append(transformed_res[i]['gt_label'])
            final_result = {
                'img_metas': img_metas,
                'img': torch.stack(img_list, dim=0),
                'gt_label': torch.stack(label_list, dim=0)
            }
            return final_result

        data_1 = _process_res(tmp_res)
        if self.pipeline_2 is not None:
            data_2 = _process_res(tmp_res_2)
            return data_1, data_2
        else:
            return data_1

    def _get_label_to_index(self):
        label_to_index = {}
        for idx, data_info in enumerate(self.data_infos):
            label = data_info['gt_label']
            if label not in label_to_index:
                label_to_index[label] = []
            label_to_index[label].append(idx)
        return label_to_index

    def __len__(self):
        return 100000


@DATASETS.register_module(name='ssda_cls_oda_dataset')
class SSDA_CLS_ODA_Datasets(SSDA_CLS_Datasets):
    def __init__(self, data_root, name, split, shot='', pipeline=None, min_len=0,
                 source_classes=None, include_unknown=True):
        self.source_classes = source_classes
        self.include_unknown = include_unknown
        assert source_classes is not None, 'you should specific source classes index for ODA datasets'
        super(SSDA_CLS_ODA_Datasets, self).__init__(data_root, name, split, shot, pipeline, min_len)

    def load_annotations(self):
        domain_labels = []
        if isinstance(self.ann_file, list):
            imgs = []
            labels = []
            for domain_ind, ann_file in enumerate(self.ann_file):
                img_index, label_list = make_oda_dataset_fromlist(ann_file, self.source_classes, self.include_unknown)
                imgs.extend(img_index)
                labels.extend(label_list)
                domain_labels.extend([domain_ind] * len(img_index))
        else:
            imgs, labels = make_oda_dataset_fromlist(self.ann_file, self.source_classes, self.include_unknown)
            domain_labels.extend([0] * len(labels))
        #
        self.n_classes = max(labels) + 1
        #
        #
        repeat_num = int(float(self.min_len) / len(imgs)) + 1
        imgs = imgs * repeat_num
        labels = labels * repeat_num
        domain_labels = domain_labels * repeat_num  #
        data_infos = []
        for ind, (img, label, domain_label) in enumerate(zip(imgs, labels, domain_labels)):
            info = {'img_prefix': self.data_prefix, 'img_info': {'filename': img}, 'gt_label': label,
                    'domain_label': domain_label, 'image_ind': ind}
            data_infos.append(info)
        return data_infos


@DATASETS.register_module(name='ssda_cls_oda_triple_dataset')
class SSDA_CLS_ODA_Triple_Datasets(SSDA_CLS_Triple_Datasets):
    def __init__(self, data_root, name, split, shot='', pipeline=None, pipeline2=None, min_len=0, repeat_pipeline=2,
                 source_classes=None, include_unknown=True):
        self.source_classes = source_classes
        self.include_unknown = include_unknown
        assert source_classes is not None, 'you should specific source classes index for ODA datasets'
        super(SSDA_CLS_ODA_Triple_Datasets, self).__init__(data_root, name, split, shot, pipeline, pipeline2, min_len)
        self.pipeline_2 = Compose(pipeline2)
        self.repeat_pipeline = repeat_pipeline
        assert self.repeat_pipeline in [1, 2], 'wrong type of repeat_pipeline'

    def load_annotations(self):
        domain_labels = []
        if isinstance(self.ann_file, list):
            imgs = []
            labels = []
            for domain_ind, ann_file in enumerate(self.ann_file):
                img_index, label_list = make_oda_dataset_fromlist(ann_file, self.source_classes, self.include_unknown)
                imgs.extend(img_index)
                labels.extend(label_list)
                domain_labels.extend([domain_ind] * len(img_index))
        else:
            imgs, labels = make_oda_dataset_fromlist(self.ann_file, self.source_classes, self.include_unknown)
            domain_labels.extend([0] * len(labels))
        #
        self.n_classes = max(labels) + 1
        #
        #
        repeat_num = int(float(self.min_len) / len(imgs)) + 1
        imgs = imgs * repeat_num
        labels = labels * repeat_num
        domain_labels = domain_labels * repeat_num  #
        data_infos = []
        for ind, (img, label, domain_label) in enumerate(zip(imgs, labels, domain_labels)):
            info = {'img_prefix': self.data_prefix, 'img_info': {'filename': img}, 'gt_label': label,
                    'domain_label': domain_label, 'image_ind': ind}
            data_infos.append(info)
        return data_infos


@DATASETS.register_module(name='my_image_folder')
class MyImageFolder(ImageFolder):
    def __init__(self, data_root, name, split, shot='', pipeline=None, min_len=0):
        root_path = os.path.join(data_root, name, 'all')
        self.name = name
        self.split = split
        clip_transforms = transforms.Compose([
            transforms.Resize(224, ),  # 用Image.BICUBIC效果会下降,  256效果会下降
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # tmp_path = '/data/zhangyx/dataset/few-shot/organized_oxford_pets/images'
        tmp_path = '/data/zhangyx/dataset/VLCS/new_images/sun'
        self.label_transform_list = None
        super(MyImageFolder, self).__init__(tmp_path, transform=clip_transforms)

    def find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        # classes.sort()
        classes = sorted(classes, key=lambda x: x.lower())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
