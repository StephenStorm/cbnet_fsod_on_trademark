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


@PIPELINES.register_module
class GridMask(object):
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=0, prob=0.7):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob


    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        img,boxes=[results[k] for k in ('img','gt_bboxes')]
        h,w,_=img.shape
        # h = img.size(1)
        # w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        # mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(img)
        mask=mask[:,:,np.newaxis]
        mask=np.repeat(mask,3,axis=2)
        # print('mask shape:', mask.shape)
        # print('img shape:', img.shape)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        results['img'] = img
        cv2.imwrite('grid_img.jpg', img)
        return results



@PIPELINES.register_module
class BBoxJitter(object):
    """
    bbox jitter
    Args:
        min (int, optional): min scale
        max (int, optional): max scale
        ## origin w scale
    """

    def __init__(self, min=0, max=2):
        self.min_scale = min
        self.max_scale = max
        self.count = 0


    def bbox_jitter(self, bboxes, img_shape):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        if len(bboxes) == 0:
            return bboxes
        jitter_bboxes = []
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            scale = np.random.uniform(self.min_scale, self.max_scale)
            w = w * scale / 2.
            h = h * scale / 2.
            xmin = center_x - w
            ymin = center_y - h
            xmax = center_x + w
            ymax = center_y + h
            box2 = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            jitter_bboxes.append(box2)
        jitter_bboxes = np.array(jitter_bboxes, dtype=np.float32)
        jitter_bboxes[:, 0::2] = np.clip(jitter_bboxes[:, 0::2], 0, img_shape[1] - 1)
        jitter_bboxes[:, 1::2] = np.clip(jitter_bboxes[:, 1::2], 0, img_shape[0] - 1)
        return jitter_bboxes

    def __call__(self, results):
        for key in results.get('bbox_fields', []):

            results[key] = self.bbox_jitter(results[key],
                                          results['img_shape'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_jitter={}-{})'.format(
            self.min_scale, self.max_scale)




@PIPELINES.register_module
class Mixup(object):
    def __init__(self, alpha=2, p=0.3):
        self.alpha = alpha
        self.prob = p
        self.q_size = 100
        self.q = queue.Queue(self.q_size)
        self.before_cnt = 0

    def __call__(self, results):
        results['is_mixup'] = False
        if self.q.qsize() >= self.q.maxsize:
            self.q.get()
        self.q.put(copy.deepcopy(results))

        if self.before_cnt < self.q_size:
            self.before_cnt += 1
        # queue塞满之前不进行mixup，避免刚训练过的样本加入mixup
        if np.random.rand() < self.prob and self.before_cnt >= self.q_size:
            self.lambd = np.random.beta(self.alpha, self.alpha)
            img1 = results['img']
            results2 = self.q.get()
            img2 = results2['img']

            height = max(img1.shape[0], img2.shape[0])
            width = max(img1.shape[1], img2.shape[1])
            mix_img = np.zeros(shape=(height, width, 3), dtype='float32')
            mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32')*self.lambd
            mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32')*(1.-self.lambd)
            mix_img = mix_img.astype('uint8')
            results['img'] = mix_img
            results['gt_bboxes'] = np.concatenate([results['gt_bboxes'], results2['gt_bboxes']], axis=0)
            results['gt_labels'] = np.concatenate([results['gt_labels'], results2['gt_labels']], axis=0)
            results['ann_info']['bboxes'] = np.concatenate([results['ann_info']['bboxes'],
                                                            results2['ann_info']['bboxes']], axis=0)
            results['ann_info']['labels'] = np.concatenate([results['ann_info']['labels'],
                                                            results2['ann_info']['labels']], axis=0)
            results['ann_info']['bboxes_ignore'] = np.concatenate([results['ann_info']['bboxes_ignore'],
                                                                   results2['ann_info']['bboxes_ignore']], axis=0)
            # results['ann_info']['masks'] = results['ann_info']['masks'].extend(results2['ann_info']['masks'])
            results['is_mixup'] = True

            # if True:
            #     for bbox, lab in zip(results['gt_bboxes'], results['gt_labels']):
            #         xmin, ymin, xmax, ymax = np.int32(bbox)
            #         cv2.rectangle(mix_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            #         cv2.putText(mix_img, str(lab), (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
            #         cv2.imwrite('mixip_img.jpg', mix_img)

        return results


@PIPELINES.register_module
class CopyPaste(object):
    def __init__(self, alpha=2, p=0.3):
        self.alpha = alpha
        self.prob = p
        self.q_size = 100
        self.q = queue.Queue(self.q_size)
        self.before_cnt = 0

    def limit(self, x, minv, maxv):
        return np.minimum(np.maximum(x, minv), maxv)

    def __call__(self, results):
        # mixup与copypaste不能兼得
        if results['is_mixup']:
            return results

        if self.q.qsize() >= self.q.maxsize:
            self.q.get()

        self.q.put(copy.deepcopy(results))
        if self.before_cnt < self.q_size:
            self.before_cnt += 1

        # queue塞满之前不进行copypaste
        if np.random.rand() < self.prob and self.before_cnt >= self.q_size:
            self.lambd = np.random.beta(self.alpha, self.alpha)
            img1 = results['img']

            results2 = self.q.get()
            copy_img = results2['img']

            height1, width1 = img1.shape[0], img1.shape[1]
            height2, width2 = copy_img.shape[0], copy_img.shape[1]
            h_scale, w_scale = height1/height2, width1/width2
            # random jitter 0.5-1.5
            h_scale = np.random.randint(50, 150)/100.0 * h_scale
            w_scale = np.random.randint(50, 150)/100.0 * w_scale

            paste_img = np.zeros(shape=(height1, width1, 3), dtype='float32')

            bbox_num = len(results2['gt_bboxes'])
            indexs = random.choices(range(bbox_num), k=np.random.randint(1, bbox_num+1))
            indexs = np.array(indexs, dtype=np.int32)

            gt_bboxes = results2['gt_bboxes'][indexs]
            gt_labels = results2['gt_labels'][indexs]
            ann_info_bboxes = results2['ann_info']['bboxes'][indexs]
            ann_info_labels = results2['ann_info']['labels'][indexs]
            ann_info_bboxes_ignore = results2['ann_info']['bboxes_ignore']


            for i, bbox in enumerate(gt_bboxes):
                xmin, ymin, xmax, ymax = np.array(bbox, dtype=np.int32)
                sub_img = copy.deepcopy(copy_img[ymin:ymax, xmin:xmax, :])
                # paste目标尺寸不能超过图像的一半, 也不能太小
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
                                                            ann_info_bboxes], axis=0)
            results['ann_info']['labels'] = np.concatenate([results['ann_info']['labels'],
                                                            ann_info_labels], axis=0)
            results['ann_info']['bboxes_ignore'] = np.concatenate([results['ann_info']['bboxes_ignore'],
                                                                   ann_info_bboxes_ignore], axis=0)

            # if True:
            #     for bbox, lab in zip(results['gt_bboxes'], results['gt_labels']):
            #         xmin, ymin, xmax, ymax = np.int32(bbox)
            #         cv2.rectangle(mix_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            #         cv2.putText(mix_img, str(lab), (xmin, ymin-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 1)
            #         cv2.imwrite('copypaste_img.jpg', mix_img)

        return results