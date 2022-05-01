# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
# from numpy import random
import random

from PIL import Image
import torch
from ..builder import PIPELINES
import queue
import copy

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

# import mmdet.datasets.pipelines.augmentation as augmentation
import mmdet.datasets.pipelines.policy as policy

from pycocotools.coco import COCO
from torch.utils.data import Dataset
import os

# all data�ǵø���ȫ����·��!!!!!!!!!!!!!!!!!
train_image_dir = 'data/train_val_split/train'
train_ann = 'data/train_val_split/train.json'


class COCO_detection(Dataset):
    def __init__(self, img_dir, ann, transforms=None):
        super(COCO_detection, self).__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = COCO(ann)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.label_map = {raw_label:i for i, raw_label in enumerate(self.coco.getCatIds())}

    def _load_image(self, id_):
        img = self.coco.loadImgs(id_)[0]['file_name']
        return Image.open(os.path.join(self.img_dir, img)).convert('RGB')

    def _load_target(self, id_):
        if len(self.coco.loadAnns(self.coco.getAnnIds(id_))) == 0: return None, None
        bboxs, labels = [], []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(id_)):
            min_x, min_y, w, h = ann['bbox']
            bboxs.append(torch.FloatTensor([min_x, min_y, min_x+w, min_y+h]))
            labels.append(self.label_map[ann['category_id']])
        bboxs, labels = torch.stack(bboxs, 0), torch.LongTensor(labels)
        return bboxs, labels

    def __getitem__(self, index):
        id_ = self.ids[index]
        image, (bboxs, labels) = self._load_image(id_), self._load_target(id_)
        if self.transforms is not None:
            image, bboxs = self.transforms(image, bboxs)

        return image, bboxs, labels

    def __len__(self):
        return len(self.ids)


my_dataset = COCO_detection(train_image_dir, train_ann)

def get_mix_data():
    index = np.random.randint(len(my_dataset))
    image, bboxs, label = my_dataset[index]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    bboxs = bboxs.numpy()
    label = label.numpy()

    return image, bboxs, label


@PIPELINES.register_module
class Mixup(object):
    def __init__(self, alpha=2, p=0.3):
        self.alpha = alpha
        self.prob = p

    def __call__(self, results):
        results['is_mixup'] = False
        if np.random.rand() < self.prob:
            self.lambd = np.random.beta(self.alpha, self.alpha)
            img1 = results['img']

            img2, bboxs2, label2 = get_mix_data()

            height = max(img1.shape[0], img2.shape[0])
            width = max(img1.shape[1], img2.shape[1])
            mix_img = np.zeros(shape=(height, width, 3), dtype='float32')
            mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32')*self.lambd
            mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32')*(1.-self.lambd)
            mix_img = mix_img.astype('uint8')
            results['img'] = mix_img
            results['gt_bboxes'] = np.concatenate([results['gt_bboxes'], bboxs2], axis=0)
            results['gt_labels'] = np.concatenate([results['gt_labels'], label2], axis=0)
            results['ann_info']['bboxes'] = np.concatenate([results['ann_info']['bboxes'],
                                                            bboxs2], axis=0)
            results['ann_info']['labels'] = np.concatenate([results['ann_info']['labels'],
                                                            label2], axis=0)
            results['is_mixup'] = True


            # if True:
            #     img = copy.deepcopy(results['img'])
            #     img = img.astype('uint8')
            #     for bbox, lab in zip(results['gt_bboxes'], results['gt_labels']):
            #         xmin, ymin, xmax, ymax = np.int32(bbox)
            #         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            #         cv2.putText(img, str(lab), (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
            #         cv2.imwrite('mixip_img.jpg', img)

        return results


@PIPELINES.register_module
class CopyPaste(object):
    def __init__(self, alpha=2, p=0.3):
        self.alpha = alpha
        self.prob = p


    def limit(self, x, minv, maxv):
        return np.minimum(np.maximum(x, minv), maxv)

    def __call__(self, results):
        # mixup��copypaste���ܼ��
        # if results['is_mixup']:
        #     return results

        if np.random.rand() < self.prob:
            self.lambd = np.random.beta(self.alpha, self.alpha)
            img1 = results['img']
            img2, bboxs2, label2 = get_mix_data()

            height1, width1 = img1.shape[0], img1.shape[1]
            height2, width2 = img2.shape[0], img2.shape[1]
            h_scale, w_scale = height1/height2, width1/width2
            # random jitter 0.5-1.5
            h_scale = np.random.randint(50, 150)/100.0 * h_scale
            w_scale = np.random.randint(50, 150)/100.0 * w_scale

            paste_img = np.zeros(shape=(height1, width1, 3), dtype='float32')

            bbox_num = len(bboxs2)
            indexs = random.choices(range(bbox_num), k=np.random.randint(1, bbox_num+1))
            indexs = np.array(indexs, dtype=np.int32)

            gt_bboxes = bboxs2[indexs]
            gt_labels = label2[indexs]

            for i, bbox in enumerate(gt_bboxes):
                xmin, ymin, xmax, ymax = np.array(bbox, dtype=np.int32)
                sub_img = copy.deepcopy(img2[ymin:ymax, xmin:xmax, :])
                # pasteĿ��ߴ粻�ܳ���ͼ���һ��, Ҳ����̫С
                # w, h = int(self.limit((xmax-xmin)*w_scale, width1*0.02, width1*0.5)), \
                #        int(self.limit((ymax-ymin)*h_scale, height1*0.02, height1*0.5))
                w, h = int(self.limit((xmax - xmin) * w_scale, 1, width1 * 0.5)), \
                       int(self.limit((ymax - ymin) * h_scale, 1, height1 * 0.5))
                new_x, new_y = np.random.randint(width1-w), np.random.randint(height1-h)
                xmin_n, ymin_n, xmax_n, ymax_n = new_x, new_y, new_x+w, new_y+h
                # update
                paste_img[ymin_n:ymax_n, xmin_n:xmax_n, :] = cv2.resize(sub_img, (w, h))
                gt_bboxes[i] = float(xmin_n), float(ymin_n), float(xmax_n), float(ymax_n)

            mix_img = img1.astype('float32')*self.lambd + paste_img.astype('float32')*(1.-self.lambd)
            mix_img = mix_img.astype('uint8')
            results['img'] = mix_img
            results['gt_bboxes'] = np.concatenate([results['gt_bboxes'], gt_bboxes], axis=0)
            results['gt_labels'] = np.concatenate([results['gt_labels'], gt_labels], axis=0)
            results['ann_info']['bboxes'] = np.concatenate([results['ann_info']['bboxes'],
                                                            gt_bboxes], axis=0)
            results['ann_info']['labels'] = np.concatenate([results['ann_info']['labels'],
                                                            gt_labels], axis=0)

            # if True:
            #     img = copy.deepcopy(results['img'])
            #     img = img.astype('uint8')
            #     for bbox, lab in zip(results['gt_bboxes'], results['gt_labels']):
            #         xmin, ymin, xmax, ymax = np.int32(bbox)
            #         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            #         cv2.putText(img, str(lab), (xmin, ymin-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 1)
            #         cv2.imwrite('copypaste_img.jpg', img)

        return results



@PIPELINES.register_module
class AutoAug(object):
    def __init__(self, policy_v='policy_v0'):
        self.policy = policy.Policy(policy=policy_v,
                                    pre_transform=[],
                                    post_transform=[])


    def __call__(self, results):
        img = results['img']
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        aug_image, aug_bboxes = self.policy(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), torch.tensor(bboxes))

        results['img'] = cv2.cvtColor(np.array(aug_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        results['gt_bboxes'] = aug_bboxes.numpy()
        results['ann_info']['bboxes'] = aug_bboxes.numpy()

        #
        # if True:
        #     mix_img = copy.deepcopy(results['img'])
        #     for bbox, lab in zip(results['gt_bboxes'], results['gt_labels']):
        #         xmin, ymin, xmax, ymax = np.int32(bbox)
        #         cv2.rectangle(mix_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        #         cv2.putText(mix_img, str(lab), (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        #         # cv2.imwrite('autoaug_img_post.jpg', mix_img)
        #
        #     img = copy.deepcopy(img)
        #     for bbox, lab in zip(bboxes, labels):
        #         xmin, ymin, xmax, ymax = np.int32(bbox)
        #         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        #         cv2.putText(img, str(lab), (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        #         cv2.imwrite('autoaug_img.jpg', np.hstack([img, mix_img]))
        #         # cv2.waitKey(0)


        return results