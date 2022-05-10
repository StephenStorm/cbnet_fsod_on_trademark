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
from pycocotools import mask as maskUtils


# all data�ǵø���ȫ����·��!!!!!!!!!!!!!!!!!
train_image_dir = 'data/trademark/train/images'
train_ann = 'data/trademark/train/annotations/instances_train2017.json'


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
            # mix_img = copy.deepcopy(img1)

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
                # mix_img[ymin_n:ymax_n, xmin_n:xmax_n, :] = cv2.resize(sub_img, (w, h))

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



#
# @PIPELINES.register_module
# class CopyPaste(object):
#     """CopyPaste images & bbox
#     Args:
#         prob (float): the probability of carrying out mixup process.
#         copy_object_num (tuple): mixup switch.
#         json_path (string): the path to label json file.
#         img_path (sting): the path to image path
#     """
#
#     def __init__(self, prob=0.5,
#                  copy_object_num=(1, 5),
#                  json_path=train_ann,
#                  img_path=train_image_dir,
#                  aug_patch=False):
#         self.prob = prob
#         self.copy_object_num = copy_object_num
#         self.json_path = json_path
#         self.img_path = img_path
#         self.aug_patch = aug_patch
#         import json
#         with open(json_path, 'r') as json_file:
#             all_labels = json.load(json_file)
#         self.all_labels = all_labels
#         self.imageid2name = dict()
#         for image in self.all_labels['images']:
#             self.imageid2name[image['id']] = image['file_name']
#         cat_ids = []
#         for catid in self.all_labels['categories']:
#             cat_ids.append(catid['id'])
#         self.cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
#
#     def _get_crop_image_annos(self):
#         # random get annos for paste
#         anno_num = np.random.randint(self.copy_object_num[0], self.copy_object_num[1])
#         crop_annos = np.random.choice(self.all_labels['annotations'], anno_num, replace=False)
#         crop_results = []
#         for crop_anno in crop_annos:
#             image_id = crop_anno['image_id']
#             file_name = self.imageid2name[image_id]
#             img = cv2.imread(os.path.join(self.img_path, file_name))
#             assert img is not None
#             crop_results.append({'img': img,
#                                  'bbox': crop_anno['bbox'],
#                                  'segmentation': crop_anno['segmentation'],
#                                  'category_id': crop_anno['category_id']})
#         return crop_results
#
#     def _mask2polygon(self, mask):
#         contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         segmentation = []
#         for contour in contours:
#             contour_list = contour.flatten().tolist()
#             if len(contour_list) > 4:  # and cv2.contourArea(contour)>10000
#                 segmentation.append(contour_list)
#         return segmentation
#
#
#     def _annTosegmap(self, segm, h, w):
#         """
#         Convert annotation which can be polygons, uncompressed RLE to RLE.
#         :return: binary mask (numpy 2D array)
#         """
#         if type(segm) == list:
#             # polygon -- a single object might consist of multiple parts
#             # we merge all parts into one mask rle code
#             rles = maskUtils.frPyObjects(segm, h, w)
#             rle = maskUtils.merge(rles)
#         else:
#             # rle
#             rle = segm
#         m = maskUtils.decode(rle)
#         return m
#
#     def _filter_gt_mask(self, gt_mask, paste_mask):
#         paste_mask = paste_mask.astype('uint8')
#         gt_m = self._annTosegmap(gt_mask, paste_mask.shape[0], paste_mask.shape[1]) == 1
#         if np.sum(gt_m * paste_mask) > 0:
#             gt_m[gt_m * paste_mask] = 0
#             m = maskUtils.encode(gt_m)
#             new_bbox = maskUtils.toBbox(m)
#             seg = self._mask2polygon(gt_m)
#             return seg, new_bbox
#         else:
#             return gt_mask, None
#
#
#     def __call__(self, results):
#         if random.uniform(0, 1) < self.prob:
#             img = results['img']
#             # mmcv.imwrite(img, 'raw.jpg')
#             gt_labels = results['ann_info']['labels']
#             gt_masks = results['ann_info']['masks']
#             gt_bboxes = results['ann_info']['bboxes']
#             # gt_semantic_seg = results['gt_semantic_seg']
#
#             crop_img_annos = self._get_crop_image_annos()
#             paste_bboxes = []
#             paste_labels = []
#             paste_segmentations = []
#
#             imageSize = (img.shape[0], img.shape[1])
#             labelMap = np.zeros(imageSize)
#
#             new_gt_masks = []
#             for crop_img_anno in crop_img_annos:
#                 crop_img = crop_img_anno['img']
#                 # mmcv.imwrite(crop_img, 'crop_img.jpg')
#                 crop_bbox = crop_img_anno['bbox']
#                 crop_segmentation = crop_img_anno['segmentation']
#
#
#                 if int(img.shape[1]-crop_bbox[2]-1) <= 1 or int(img.shape[0]-crop_bbox[3]-1) <= 1:
#                     h_s, w_s = img.shape[0]/crop_img.shape[0], img.shape[1]/crop_img.shape[1]
#                     # print(h_s, w_s)
#                     crop_img = cv2.resize(crop_img, (img.shape[1], img.shape[0]))
#                     crop_bbox = int(crop_bbox[0])*w_s, int(crop_bbox[1]*h_s), \
#                                 int(crop_bbox[2]*w_s), int(crop_bbox[3]*h_s)
#
#                     crop_segmentation[0][0::2] = list(np.array(crop_segmentation[0][0::2]) * w_s)
#                     crop_segmentation[0][1::2] = list(np.array(crop_segmentation[0][1::2]) * h_s)
#
#                 crop_image_labelMask = self._annTosegmap(crop_segmentation, crop_img.shape[0], crop_img.shape[1]) == 1
#                 # random location (xmin, ymin)
#                 # if int(img.shape[1]-crop_bbox[2]-1) <= 1:
#                 #     continue
#                 # elif int(img.shape[0]-crop_bbox[3]-1) <= 1:
#                 #     continue
#
#                 paste_xmin = random.randint(1, int(img.shape[1]-crop_bbox[2]-1))
#                 paste_ymin = random.randint(1, int(img.shape[0]-crop_bbox[3]-1))
#                 # TODO box resize, translate, rotate, etc
#                 # print(paste_xmin, paste_ymin)
#                 # print(crop_bbox)
#                 # print(crop_img.shape)
#                 # TODO crop object from segmentation, not from bbox
#                 # img[paste_ymin: paste_ymin+int(crop_bbox[3]), paste_xmin: paste_xmin+int(crop_bbox[2]), :] = \
#                 #     crop_img[int(crop_bbox[1]): int(crop_bbox[1])+int(crop_bbox[3]), int(crop_bbox[0]): int(crop_bbox[0])+int(crop_bbox[2]), :]
#                 paste_bboxes.append([np.float32(paste_xmin),
#                                        np.float32(paste_ymin),
#                                        np.float32(paste_xmin + crop_bbox[2] - 1),
#                                        np.float32(paste_ymin + crop_bbox[3] - 1)])
#                 paste_labels.append(np.int64(self.cat2label[crop_img_anno['category_id']]))
#                 offset_w = crop_bbox[0] - paste_xmin
#                 offset_h = crop_bbox[1] - paste_ymin
#                 paste_segmentation = []
#                 # for i, cs in enumerate(crop_segmentation):
#                 #     paste_segmentation.append([])
#                 #     for j, cs_xy in enumerate(cs):
#                 #         if j%2==0:
#                 #             paste_segmentation[i].append(cs_xy - offset_w)
#                 #         else:
#                 #             paste_segmentation[i].append(cs_xy - offset_h)
#                 #
#                 # paste_segmentations.append(paste_segmentation)
#                 # labelMask = self._annTosegmap(paste_segmentation, img.shape[0], img.shape[1]) == 1
#                 # newLabel = crop_img_anno['category_id']
#                 # labelMap[labelMask] = newLabel
#                 # # img = img * (1-labelMask)
#                 # img[labelMask==1] = 0
#                 # # mmcv.imwrite(np.uint8(crop_image_labelMask*255), 'crop_image_labelMask.jpg')
#                 # # crop_img = crop_img * crop_image_labelMask
#                 # crop_img[crop_image_labelMask==0] = 0
#                 # mmcv.imwrite(crop_img, 'crop_img_binary.jpg')
#                 crop_patch = crop_img[int(crop_bbox[1]): int(crop_bbox[1])+int(crop_bbox[3]), int(crop_bbox[0]): int(crop_bbox[0])+int(crop_bbox[2]), :]
#                 # mmcv.imwrite(crop_patch, 'crop_patch.jpg')
#                 # img[paste_ymin: paste_ymin + int(crop_bbox[3]), paste_xmin: paste_xmin + int(crop_bbox[2]), :] = 0
#                 # flip patch horilize
#
#                 # img[paste_ymin: paste_ymin + int(crop_bbox[3]), paste_xmin: paste_xmin + int(crop_bbox[2]), :] += crop_patch
#                 img[paste_ymin: paste_ymin + int(crop_bbox[3]), paste_xmin: paste_xmin + int(crop_bbox[2]), :] = crop_patch
#
#                 # mmcv.imwrite(img, 'paste_img.jpg')
#                 # filter gt mask
#             # for i, gt_mask in enumerate(gt_masks):
#             #     new_gt_mask, new_box = self._filter_gt_mask(gt_mask, labelMap)
#             #     new_gt_masks.append(new_gt_mask)
#             #     if new_box is not None:
#             #         new_box[2] = new_box[0] + new_box[2]
#             #         new_box[3] = new_box[1] + new_box[3]
#             #         gt_bboxes[i] = new_box
#             #
#             # for gt_mask, gt_label in zip(gt_masks, gt_labels):
#             #     labelMask = self._annTosegmap(gt_mask, img.shape[0], img.shape[1]) == 1
#             #     # newLabel = gt_label
#             #     newLabel = 1  # gt_label is 0, so hard code to 1
#             #     labelMap[labelMask] = newLabel
#             # # mmcv.imwrite(img, 'paste_img.jpg')
#             # # mmcv.imwrite(np.uint8(labelMap*255), 'labelMap.jpg')
#             # labelMap = labelMap.astype(np.uint8)
#             # gt_semantic_seg = labelMap
#             results['img'] = img
#             # print(gt_labels, np.array(paste_labels))
#             # print(list(gt_bboxes), paste_bboxes)
#             # results['ann_info']['labels'] = np.hstack((gt_labels, np.array(paste_labels)))
#             # results['ann_info']['bboxes'] = np.vstack((gt_bboxes, np.array(paste_bboxes)))
#             results['ann_info']['labels'] = np.concatenate((gt_labels, np.array(paste_labels)), axis=0)
#             results['ann_info']['bboxes'] = np.concatenate((gt_bboxes, np.array(paste_bboxes)), axis=0)
#
#             # print('*'*20)
#             # print(np.array(gt_masks).shape)
#             # print('*' * 20)
#             # print(gt_masks)
#             # print('*' * 20)
#             # print(paste_segmentations)
#             # gt_masks.extend(paste_segmentations)
#             # results['ann_info']['masks'] = gt_masks
#             # # results['ann_info']['masks'] = np.vstack((list(gt_masks), paste_segmentations))
#             # # print('*' * 20)
#             # # print(results['ann_info']['masks'])
#             # # input()
#             # results['gt_semantic_seg'] = gt_semantic_seg
#             return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(prob={}, copy_object_num={}, json_path={}, img_path={})'.\
#             format(self.prob,
#                    self.copy_object_num,
#                    self.json_path,
#                    self.img_path)

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