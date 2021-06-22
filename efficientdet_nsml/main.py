import os
import datetime

import numpy as np

import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import argparse

from data_loader import feed_infer
from data_local_loader import data_loader, data_loader_local
from evaluation import evaluation_metrics

#########
import time
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
from resnext import resnet50, resnet101
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label
#########

import nsml
from nsml import IS_ON_NSML

if IS_ON_NSML:
    VAL_DATASET_PATH = None
else:
    VAL_DATASET_PATH = os.path.join('/home/data/iitp_2020_fallen_final/nsml_dataset/test/test_data')
    VAL_LABEL_PATH = os.path.join('/home/data/iitp_2020_fallen_final/nsml_dataset/test/test_label')

IMG_WIDTH = 960
IMG_HEIGHT = 540

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.13
iou_threshold = 0.2


def _infer(model, model_a, root_path, test_loader=None):
    """Inference function for NSML infer.
    Args:
        model: Trained model instance for inference
        root_path: Set proper path for local evaluation.
        test_loader: Data loader is defined in `data_local_loader.py`.
    Returns:
        results: tuple of (image_names, outputs)
            image_names: list of file names (size: N)
                        (ex: ['aaaa.jpg', 'bbbb.jpg', ... ])
            outputs: numpy array of bounding boxes (size: N x 4)
                        (ex: [[x1,y1,x2,y2],[x1,y1,x2,y2],...])
    """
    if test_loader is None:
        # Eval using local dataset loader
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test',
            batch_size=1)

    base = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    model.requires_grad_(False)
    model.eval()
    model_a.requires_grad_(False)
    model_a.eval()
    
    outputs = []
    image_names = []

    prior_bbox = np.array([])

    for idx, (img_path, x) in enumerate(test_loader):
        now_name = img_path[0].split('/')[-1]
        now_name = '_'.join(now_name.split('_')[:-1])

        if idx == 0 :
            pre_name = now_name
            ch_toogle = (pre_name != now_name)
        else:
            ch_toogle = (pre_name != now_name) 
            if ch_toogle:
                pre_name = now_name

        if (idx) % 5 != 0 and not ch_toogle:
            if prior_bbox.shape[0] != 0:
                outputs.extend(prior_bbox[0].astype(np.int16))
                image_names.append(img_path[0].split('/')[-1])
            continue

        x = x.cuda()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                        anchors, regression, classification[:,:,:1],
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

        out = invert_affine(out)
        bbox = out[0]['rois']

        if len(bbox) !=0:
            tmp = bbox
            tmp_wh = tmp[:,2:4] - tmp[:,:2]
            keep= (tmp_wh[:,0] > tmp_wh[:,1]* 0.9) & (tmp_wh[:,0] >32) & (tmp_wh[:,1] >32)
            out[0]['rois'] = np.array(out[0]['rois'][keep])
            
        bbox = out[0]['rois']
        
        #bbox가 아무것도 없는 경우(잘 일어나지는 않음 그러나 이 부분이 오류를 유발 할 수 도 있음)
        if len(bbox) == 0:
            if prior_bbox.shape[0] != 0:
                outputs.extend(prior_bbox[0].astype(np.int16))
                image_names.append(img_path[0].split('/')[-1])
            continue

        img = Image.open(img_path[0])
        crop_tensor = list()
        for b in bbox:
            img = img.crop(b)    
            crop_tensor.append(base(img).unsqueeze(0))
        
        data = torch.cat(crop_tensor, dim=0)

        with torch.no_grad():
            data = data.cuda()
            out2 = model_a(data)
        prob = F.softmax(out2, dim=1)

        index = prob[:,1].cpu().argmax()
        masked_bbox = np.array([bbox[index]])
        prior_bbox = masked_bbox

        if prior_bbox.shape[0] != 0:
            outputs.extend(prior_bbox[0].astype(np.int16))
            image_names.append(img_path[0].split('/')[-1])

    outputs = np.array(outputs).reshape(-1,4)
    print(outputs.shape)
    print(outputs)

    print(len(image_names))
    print(image_names)
    results = (image_names, outputs)
    return results

def local_eval(model, test_loader=None, test_label_file=None):
    """Local debugging function.
    You can use this function for debugging.
    """
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, model_a, root_path, test_loader=test_loader))
    if not test_label_file:
        test_label_file = os.path.join(VAL_DATASET_PATH, 'test_label')
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file
    )
    print('[Eval result] recall: {:.2f}'.format(metric_result))
    return metric_result


def bind_nsml(model, model_a):
    """NSML binding function.
    This function is used for internal process in NSML. Do not change.
    """

    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'model_a' : model_a.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        model_a.load_state_dict(state['model_a'])
        print('loaded')

    def infer(root_path):
        return _infer(model, model_a, root_path)

    nsml.bind(save=save, load=load, infer=infer)


def load_weight(model, model_a, weight_file, a_weight_file):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file) and os.path.isfile(a_weight_file):
        # model.load_state_dict(torch.load(weight_file)['model'])
        model.load_state_dict(torch.load(weight_file))
        model_a.load_state_dict(torch.load(a_weight_file))
        print('load weight from {}.'.format(weight_file))
        print('load weight from {}.'.format(a_weight_file))
    else:
        print('weight file {} is not exist.'.format(weight_file))
        print('=> random initialized model will be used.')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--local_eval", default=False, action='store_true')
    args.add_argument("--weight_file", type=str, default='weights/efficientdet-d5.pth')
    args.add_argument("--a_weight_file", type=str, default='weights/resnet50epoch_all_100.pth')
    # These arguments are reserved for nsml. Do not change.
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    # model building
    model = EfficientDetBackbone(compound_coef=5, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
    model_a = resnet50(pretrained=False, num_classes=2)
    #model_a = resnet101(pretrained=False, num_classes=2)
    # load trained weight
    load_weight(model, model_a, config.weight_file, config.a_weight_file)

    if config.cuda:
        model = model.cuda()
        model_a = model_a.cuda()

    print("bind model")
    bind_nsml(model, model_a)
    nsml.save('model')

    if config.pause:
        nsml.paused(scope=locals())

    print("IS_ON_NSML", IS_ON_NSML)
    print("local_eval", config.local_eval)
    print("VAL_DATASET_PATH", VAL_DATASET_PATH)

    if config.mode == 'train':
        val_loader = data_loader_local(root=VAL_DATASET_PATH)
        time_ = datetime.datetime.now()
        if not IS_ON_NSML and config.local_eval:
            # Local debugging block.
            start_time = time.time()
            local_eval(model, val_loader, VAL_LABEL_PATH)
            print('{} sec'.format(time.time() - start_time))