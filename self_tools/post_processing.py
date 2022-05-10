import json
from tqdm import tqdm
import numpy as np
import queue
import sys
import os

def most_frequent(list):
    return max(set(list), key=list.count)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def add_on_imageid(json_data):
    imageid_info = {}
    for cell in tqdm(json_data):
        image_id = cell['image_id']
        if not image_id in imageid_info.keys():
            imageid_info[image_id] = []
        imageid_info[image_id].append(cell)

    return imageid_info

def get_imageid_category(imageid_info):
    imageid_cates = {}
    for image_id in tqdm(imageid_info.keys()):

        det_on_imageid = imageid_info[image_id]

        max_score = 0
        # imageid_cate = 0
        for cell in det_on_imageid:
            score = cell['score']
            cate  = cell['category_id']
            if score > max_score:
                max_score = score
                imageid_cate = cate

        imageid_cates[image_id] = imageid_cate

        # print(image_id, imageid_cate)
    return imageid_cates

def filter_cate(imageid_cates):
    cates = [imageid_cates[key] for key in imageid_cates.keys()]
    new_cates = [imageid_cates[key] for key in imageid_cates.keys()]
    for i in range(5, len(cates)-5):
        first_num = most_frequent(cates[i-5:i])
        last_num = most_frequent(cates[i+1:i+6])
        if first_num == last_num:
            # print('before change: ', cates[i - 5:i], new_cates[i], cates[i + 1:i + 6])
            new_cates[i] = first_num
            # print('after change: ', cates[i-5:i], new_cates[i], cates[i+1:i+6])
    new_imageid_cates = {}
    for i, key in enumerate(imageid_cates.keys()):
        new_imageid_cates[key] = new_cates[i]
        # print(i, cates[i], new_cates[i])

    return new_imageid_cates




# def processing_json(json_data, imageid_cates, save_path):
#     new_json_data = []
#     for data in tqdm(json_data):
#         imageid = data['image_id']
#         cate = data['category_id']
#         if cate == imageid_cates[imageid]:
#             new_json_data.append(data)

#     with open(save_path, 'w') as fp:
#         json.dump(new_json_data, fp, indent=4)

def processing_json(json_data, imageid_cates, save_path):
    new_json_data = []
    for data in tqdm(json_data):
        imageid = data['image_id']
        # cate = data['category_id']
        # if cate == imageid_cates[imageid]:
        #     new_json_data.append(data)
        data['category_id'] = imageid_cates[imageid]
        new_json_data.append(data)

    with open(save_path, 'w') as fp:
        json.dump(new_json_data, fp, indent=4)




if __name__=='__main__':
    file = '/opt/tige/haggs/CBNetV2/work_dirs/cascade_mask_rcnn_cbv2_swin_base_cp_mixup0.5_affine_freeze_stage1_resize/0510TTA_swa.bbox.json'
    file_post = file.replace('.bbox', '_postprocess')
    # print(file_post)
    if os.path.exists(file_post) :
        sys.exit()
    # file_post = '/opt/tiger/haggs/CBNetV2/work_dirs/cascade_mask_rcnn_cbv2_swin_large_sabl_gc_gn_alldata_simmim_drop0.3/res_drop0.3_postprocess.json'

    json_data = load_json(file)
    imageid_info = add_on_imageid(json_data)
    imageid_cates = get_imageid_category(imageid_info)
    print(imageid_cates)

    new_imageid_cates = filter_cate(imageid_cates)
    print(new_imageid_cates)

    change_num = 0
    for key in new_imageid_cates.keys():
        before = imageid_cates[key]
        after = new_imageid_cates[key]
        if before != after:
            change_num += 1
            print('change_num', change_num)
        print(key, imageid_cates[key], new_imageid_cates[key])

    print('change_num', change_num)

    processing_json(json_data, new_imageid_cates, file_post)
    print(file_post)




