import numpy as np
import os
import cv2
from scipy import misc
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import pdb

from tensorpack import *

# import models
from detect.train import Model as DetectModel
from segment_sealion.train import Model as SegmentSealionModel
from segment_fox.train import Model as SegmentFoxModel
from bone_sealion.bone_point import Model as BoneSealionModel
from bone_fox.bone_point import Model as BoneFoxModel

# import configurations
from detect.cfgs.config import cfg as detect_cfg
from segment_sealion.cfgs.config import cfg as segment_sealion_cfg
from segment_fox.cfgs.config import cfg as segment_fox_cfg
from bone_sealion.cfgs.config import cfg as bone_sealion_cfg
from bone_fox.cfgs.config import cfg as bone_fox_cfg

from time import gmtime, strftime

animals = ["sealion", "fox"]
SegmentModels = [SegmentSealionModel, SegmentFoxModel]
BoneModels = [BoneSealionModel, BoneFoxModel]
bone_cfgs = [bone_sealion_cfg, bone_fox_cfg]
segment_cfgs = [segment_sealion_cfg, segment_fox_cfg]

def initialize(path_prefix):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # initialize detection model
    detect_sess_init = SaverRestore(os.path.join(path_prefix, "models/detect"))
    detect_model = DetectModel()
    predict_config_detect = PredictConfig(session_init=detect_sess_init,
                                          model=detect_model,
                                          input_names=["input", "spec_mask"],
                                          output_names=["pred_x", "pred_y", "pred_w", "pred_h", "pred_conf", "pred_prob"])
    predict_func_detect = OfflinePredictor(predict_config_detect)


    # initialize segment model
    predict_funcs_segment = []
    for idx, animal in enumerate(animals):
        segment_sess_init = SaverRestore(os.path.join(path_prefix, "models/segment-" + animal))
        segment_model = SegmentModels[idx]()
        predict_config_segment = PredictConfig(session_init=segment_sess_init,
                                              model=segment_model,
                                              input_names=["input"],
                                              output_names=["softmax_output"])
        predict_funcs_segment.append(OfflinePredictor(predict_config_segment))


    # initialize bone model
    predict_funcs_bone = []
    for idx, animal in enumerate(animals):
        bone_sess_init = SaverRestore(os.path.join(path_prefix, "models/bone-" + animal))
        bone_model = BoneModels[idx](18)
        predict_config_bone = PredictConfig(session_init=bone_sess_init,
                                            model=bone_model,
                                            input_names=["input"],
                                            output_names=["logits"])
        predict_funcs_bone.append(OfflinePredictor(predict_config_bone))

    return [predict_func_detect, predict_funcs_segment, predict_funcs_bone]

def detect(predict_funcs, input_img):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    detect_func, _, _ = predict_funcs

    # detect
    width_crop = 1.2
    height_crop = 1.2
    ori_height, ori_width, _ = input_img.shape
    detect_input = cv2.resize(input_img, (detect_cfg.img_w, detect_cfg.img_h))
    detect_input = np.expand_dims(detect_input, axis=0)
    spec_mask = np.zeros((1, detect_cfg.n_boxes, detect_cfg.img_w // 32, detect_cfg.img_h // 32), dtype=float) == 0
    predictions = detect_func([detect_input, spec_mask])

    [pred_x, pred_y, pred_w, pred_h, pred_conf, pred_prob] = predictions

    _, box_n, klass_num, grid_h, grid_w = pred_prob.shape
    pred_conf_tile = np.tile(pred_conf, (1, 1, klass_num, 1, 1))
    klass_conf = pred_prob * pred_conf_tile

    width_rate = ori_width / float(detect_cfg.img_w)
    height_rate = ori_height / float(detect_cfg.img_h)

    max_conf = np.max(klass_conf)

    print("Max confidence: " + str(max_conf))

    # no objects found in detection model
    if max_conf <= 0.25:
        print("Maximum confidence lower then the threshold. Exit")
        return -1

    [_, n, _, gh, gw] = np.unravel_index(klass_conf.argmax(), klass_conf.shape)
    anchor = detect_cfg.anchors[n]
    w = pred_w[0, n, 0, gh, gw]
    h = pred_h[0, n, 0, gh, gw]
    x = pred_x[0, n, 0, gh, gw]
    y = pred_y[0, n, 0, gh, gw]

    center_w_cell = gw + x
    center_h_cell = gh + y
    box_w_cell = np.exp(w) * anchor[0]
    box_h_cell = np.exp(h) * anchor[1]

    center_w_pixel = center_w_cell * 32
    center_h_pixel = center_h_cell * 32
    box_w_pixel = box_w_cell * 32 * width_crop
    box_h_pixel = box_h_cell * 32 * height_crop

    xmin = center_w_pixel - box_w_pixel // 2
    ymin = center_h_pixel - box_h_pixel // 2
    xmax = center_w_pixel + box_w_pixel // 2
    ymax = center_h_pixel + box_h_pixel // 2

    xmin = np.max([xmin, 0])
    ymin = np.max([ymin, 0])
    xmax = np.min([xmax, float(detect_cfg.img_w)])
    ymax = np.min([ymax, float(detect_cfg.img_h)])

    box = np.asarray([xmin, ymin, xmax, ymax])
    return box

def segment_and_bone(predict_funcs, animal_idx, input_img, crf=True):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    batch_input_img = np.expand_dims(input_img, axis=0)

    _, segment_funcs, bone_funcs = predict_funcs
    segment_func = segment_funcs[animal_idx]
    bone_func = bone_funcs[animal_idx]

    bone_cfg = bone_cfgs[animal_idx]
    segment_cfg = segment_cfgs[animal_idx]

    (height, width, _) = input_img.shape

    predictions = segment_func([batch_input_img])[0]
    class_num = predictions.shape[1]

    predictions = np.reshape(predictions, (height, width, class_num))

    if crf == True:
        d = dcrf.DenseCRF2D(height, width, segment_cfg.class_num)

        # set unary potential
        predictions = np.transpose(predictions, (2, 0, 1))
        U = unary_from_softmax(predictions)
        d.setUnaryEnergy(U)

        # set pairwise potential
        # This creates the color-independent features and then add them to the CRF
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(8, 8), srgb=(13, 13, 13), rgbim=input_img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        iter_num = 5
        result = np.argmax(d.inference(iter_num), axis=0)
        result = np.reshape(result, (height, width))
    else:
        result = np.argmax(predictions, axis=2)

    seg = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            seg[h, w] = result[h, w]

    # get bboxes from segment results
    bboxes = []
    for label_idx in range(bone_cfg.obj_num):
        cur_label_img = (seg == (label_idx + 1)).astype(int)

        t = np.nonzero(cur_label_img)
        ymin = np.min(t[0])
        ymax = np.max(t[0])
        xmin = np.min(t[1])
        xmax = np.max(t[1])
        bboxes.append([xmin, ymin, xmax, ymax])

    # bone
    # bone_input = np.expand_dims(bone_input, axis=0)

    predictions = bone_func([batch_input_img])[0][0]
    norm_bones = predictions + np.asarray(bone_cfg.anchor_bones)

    # restore abolute coords for bones
    bones = []
    i = 0
    for box_idx in bone_cfg.match:
        norm_bone = norm_bones[i:i+2]
        i += 2
        [xmin, ymin, xmax, ymax] = bboxes[box_idx]
        xcenter = (xmax + xmin) / 2
        ycenter = (ymax + ymin) / 2
        box_width = (xmax - xmin)
        box_height = (ymax - ymin)
        x = norm_bone[0] * box_width + xcenter
        y = norm_bone[1] * box_height + ycenter
        bones.append([int(x), int(y)])

    bones = np.asarray(bones)
    # seg = seg.astype(int)

    return [bones, seg]



# input_img should be numpy array with RGB channels
def predict(predict_funcs, animal_idx, input_img, crf=True, test=False, pad=False):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    [detect_func, segment_funcs, bone_funcs] = predict_funcs
    segment_func = segment_funcs[animal_idx]
    bone_func = bone_funcs[animal_idx]

    bone_cfg = bone_cfgs[animal_idx]
    segment_cfg = segment_cfgs[animal_idx]

    # detect
    width_crop = 1.2
    height_crop = 1.2
    ori_height, ori_width, _ = input_img.shape
    detect_input = cv2.resize(input_img, (detect_cfg.img_w, detect_cfg.img_h))
    detect_input = np.expand_dims(detect_input, axis=0)
    spec_mask = np.zeros((1, detect_cfg.n_boxes, detect_cfg.img_w // 32, detect_cfg.img_h // 32), dtype=float) == 0
    predictions = detect_func([detect_input, spec_mask])

    [pred_x, pred_y, pred_w, pred_h, pred_conf, pred_prob] = predictions

    _, box_n, klass_num, grid_h, grid_w = pred_prob.shape
    pred_conf_tile = np.tile(pred_conf, (1, 1, klass_num, 1, 1))
    klass_conf = pred_prob * pred_conf_tile

    width_rate = ori_width / float(detect_cfg.img_w)
    height_rate = ori_height / float(detect_cfg.img_h)

    max_conf = np.max(klass_conf)

    print("Max confidence: " + str(max_conf))

    # no objects found in detection model
    if max_conf <= 0.25:
        print("Maximum confidence lower then the threshold. Exit")
        return -1

    [_, n, _, gh, gw] = np.unravel_index(klass_conf.argmax(), klass_conf.shape)
    anchor = detect_cfg.anchors[n]
    w = pred_w[0, n, 0, gh, gw]
    h = pred_h[0, n, 0, gh, gw]
    x = pred_x[0, n, 0, gh, gw]
    y = pred_y[0, n, 0, gh, gw]

    center_w_cell = gw + x
    center_h_cell = gh + y
    box_w_cell = np.exp(w) * anchor[0]
    box_h_cell = np.exp(h) * anchor[1]

    center_w_pixel = center_w_cell * 32
    center_h_pixel = center_h_cell * 32
    box_w_pixel = box_w_cell * 32 * width_crop
    box_h_pixel = box_h_cell * 32 * height_crop

    xmin = float(center_w_pixel - box_w_pixel // 2)
    ymin = float(center_h_pixel - box_h_pixel // 2)
    xmax = float(center_w_pixel + box_w_pixel // 2)
    ymax = float(center_h_pixel + box_h_pixel // 2)
    d_xmin = int(np.max([xmin, 0]) * width_rate)
    d_ymin = int(np.max([ymin, 0]) * height_rate)
    d_xmax = int(np.min([xmax, float(detect_cfg.img_w)]) * width_rate)
    d_ymax = int(np.min([ymax, float(detect_cfg.img_h)]) * height_rate)

    detect_target = input_img[d_ymin:d_ymax, d_xmin:d_xmax, :]

    # save the detect result
    if test: misc.imsave('output_images/detect_output.png', detect_target)

    if pad == True:
        crop_height = ymax - ymin
        crop_width = xmax - xmin
        pad_val = int(np.abs(crop_height - crop_width) / 2)
        if crop_height > crop_width:
            pad = [[0, 0], [pad_val, pad_val], [0, 0]]
        else:
            pad = [[pad_val, pad_val], [0, 0], [0, 0]]

        detect_target = np.pad(detect_target, pad, 'edge')
        if test: misc.imsave('output_images/pad_output.png', detect_target)


    # segment
    segment_input = cv2.resize(detect_target, (224, 224))
    if test: misc.imsave('output_images/segment_input.png', segment_input)

    (height, width, _) = segment_input.shape
    batch_segment_input = np.expand_dims(segment_input, axis=0)
    predictions = segment_func([batch_segment_input])[0]

    class_num = predictions.shape[1]

    predictions = np.reshape(predictions, (height, width, class_num))

    if crf == True:
        d = dcrf.DenseCRF2D(height, width, segment_cfg.class_num)

        # set unary potential
        predictions = np.transpose(predictions, (2, 0, 1))
        U = unary_from_softmax(predictions)
        d.setUnaryEnergy(U)

        # set pairwise potential
        # This creates the color-independent features and then add them to the CRF
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(8, 8), srgb=(13, 13, 13), rgbim=segment_input,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        iter_num = 5
        result = np.argmax(d.inference(iter_num), axis=0)
        result = np.reshape(result, (height, width))
    else:
        result = np.argmax(predictions, axis=2)

    output = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            output[h, w] = result[h, w]
    if test: misc.imsave("output_images/segment_output.png", output)

    # get bboxes from segment results
    bboxes = []
    for label_idx in range(bone_cfg.obj_num):
        cur_label_img = (output == (label_idx + 1)).astype(int)

        t = np.nonzero(cur_label_img)
        ymin = np.min(t[0])
        ymax = np.max(t[0])
        xmin = np.min(t[1])
        xmax = np.max(t[1])
        bboxes.append([xmin, ymin, xmax, ymax])

    # bone
    bone_input = np.expand_dims(segment_input, axis=0)

    predictions = bone_func([bone_input])[0][0]
    norm_bones = predictions + np.asarray(bone_cfg.anchor_bones)

    # restore abolute coords for bones
    bones = []
    i = 0
    for box_idx in bone_cfg.match:
        norm_bone = norm_bones[i:i+2]
        i += 2
        [xmin, ymin, xmax, ymax] = bboxes[box_idx]
        xcenter = (xmax + xmin) / 2
        ycenter = (ymax + ymin) / 2
        box_width = (xmax - xmin)
        box_height = (ymax - ymin)
        x = norm_bone[0] * box_width + xcenter
        y = norm_bone[1] * box_height + ycenter
        bones.append([int(x), int(y)])

    # transform the result back to original image and return
    box = np.asarray([d_xmin, d_ymin, d_xmax, d_ymax])
    box = box.astype(int)
    d_height = d_ymax - d_ymin
    d_width = d_xmax - d_xmin
    h_over_w = d_height / d_width
    if h_over_w > 1:
        fx = 1 / h_over_w
        fy = 1
    else:
        fx = 1
        fy = h_over_w
    seg = cv2.resize(output, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    bones = np.asarray(bones)
    bones[:, 0] = bones[:, 0] * fx
    bones[:, 1] = bones[:, 1] * fy
    bones = bones.astype(int)

    final_output = np.copy(seg)
    for bone in bones:
        final_output = cv2.circle(final_output, (bone[0], bone[1]), 2, (255, 255, 255), thickness=2, lineType=8, shift=0)
    if test: misc.imsave("output_images/bone_output.png", final_output)
    seg = seg.astype(int)

    # box: 4
    # box: hxw
    # box: bx2
    return [box, bones, seg]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to the input image', required=True)
    parser.add_argument('--one_request', action='store_true')
    parser.add_argument('--animal', help='name of the animal', required=True)
    parser.add_argument('--crf', action="store_true", help='whether to use CRF')
    args = parser.parse_args()

    predict_funcs = initialize("./")

    img = cv2.imread(args.input)
    animals = ["sealion", "fox"]
    if args.animal in animals:
        animal_idx = animals.index(args.animal)
    else:
        animal_idx = 0


    if args.one_request:
        predict(predict_funcs, animal_idx, img, crf=args.crf, test=True)
    else:
        detect_input = cv2.resize(img, (416, 416))
        xmin, ymin, xmax, ymax = detect(predict_funcs, img)
        detect_output = detect_input[ymin:ymax, xmin:xmax]

        # save detect result as image
        segment_input = cv2.resize(detect_output, (224, 224))
        cv2.imwrite("output_images/detect_output.png", segment_input)
        bones, seg = segment_and_bone(predict_funcs, animal_idx, segment_input, crf=True)

        # save segment and bone result as images
        misc.imsave("output_images/seg_output.png", seg)
        final_output = np.copy(seg)
        for bone in bones:
            final_output = cv2.circle(final_output, (bone[0], bone[1]), 2, (255, 255, 255), thickness=2, lineType=8, shift=0)
        cv2.imwrite("output_images/bone_output.png", final_output)



    # for i in range(100):
    #     predict(predict_funcs, img)
    #     if i % 10 == 0:
    #         cur_time = strftime("%a, %d %b %Y %X +0000", gmtime())
    #         print(str(i) + ": " + cur_time)
