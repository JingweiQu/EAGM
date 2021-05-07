# ========================================================================================
# Generate edge semantic features for PascalVOC
# Author: Jingwei Qu
# Date: 2020/10/25 21:34
# ========================================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import numpy as np
import os
import cv2
import random
from skimage.draw import line
from pathlib import Path
import pickle
import xml.dom.minidom
import argparse

from transformation import GeometricTnf

CACHE_PATH = "../data/cache"
VOC_IMAGE_ROOT = "../data/PascalVOC/JPEGImages"
VOC_ANNOTATION_ROOT = "../data/PascalVOC/annotations"
VOC_FEATURE_ROOT = "../data/PascalVOC/features"
VOC_CATEGORIES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                  "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                  "tvmonitor"]
IMG_SIZE = 256
affineTnf = GeometricTnf(geometric_model='affine', out_h=IMG_SIZE, out_w=IMG_SIZE, use_cuda=False)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Pooling for edge semantic features
pool_feat = nn.AdaptiveMaxPool1d(1)


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # gpu
        # torch.cuda.manual_seed_all(seed) # all gpu
    rand = np.random.RandomState(seed=seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = True  # Accelerate

    return rand


def get_image(img_name, bounds):
    image = cv2.imread(img_name)
    # If the image just has two channels, add one channel
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)

    # Get image size, (H, W, C)
    im_size = np.asarray(image.shape)

    # Clip bounding box
    bounds[0] = bounds[0].clip(0, im_size[1] - 1)
    bounds[1] = bounds[1].clip(0, im_size[0] - 1)
    bounds[2] = bounds[2].clip(0, im_size[1] - 1)
    bounds[3] = bounds[3].clip(0, im_size[0] - 1)

    image = image[int(bounds[1]):int(bounds[3]), int(bounds[0]):int(bounds[2]), :]

    # Resize image
    image = image[:, :, ::-1]
    image = torch.Tensor(image.astype(np.float32))
    image = image.permute(2, 0, 1)
    image.requires_grad = False
    image = affineTnf(image_batch=image.unsqueeze(0)).squeeze()
    image = image_normalize(image).unsqueeze(0)
    image = image.cuda()

    return image, bounds, im_size


def image_normalize(image):
    image /= 255.0
    image = normalize(image)

    return image


def feature_name(file):
    iLen = len(file)
    raw_file = file[:iLen - 4]
    edge_feat_file = raw_file + "_EF.pt"
    return edge_feat_file


def _load_annotation(file):
    dom = xml.dom.minidom.parse(file)
    root = dom.documentElement

    image = root.getElementsByTagName('image')[0].firstChild.data

    keypoints = root.getElementsByTagName('keypoints')[0]
    kps = keypoints.getElementsByTagName('keypoint')

    bounds = root.getElementsByTagName('visible_bounds')[0]
    xmin = float(bounds.getAttribute('xmin'))
    ymin = float(bounds.getAttribute('ymin'))
    xmax = xmin + float(bounds.getAttribute('width'))
    ymax = ymin + float(bounds.getAttribute('height'))

    annoName = []
    annoPts = []
    for kp in kps:
        x = float(kp.getAttribute('x'))
        y = float(kp.getAttribute('y'))
        name = kp.getAttribute('name')
        annoName.append(name)
        annoPts.append([x, y])

    annoName = np.array(annoName)
    annoPts = np.array(annoPts)

    if annoPts.shape[0] >= 3:
        offset = 4
        xmin = min(xmin, np.floor(min(annoPts[:, 0]))) - offset
        ymin = min(ymin, np.floor(min(annoPts[:, 1]))) - offset
        xmax = max(xmax, np.ceil(max(annoPts[:, 0]))) + offset
        ymax = max(ymax, np.ceil(max(annoPts[:, 1]))) + offset

    bounds = np.array([xmin, ymin, xmax, ymax])

    return image, annoName, annoPts, bounds


def relocate_pts(pts_prime, bounds):
    pts_prime[:, 0] = (pts_prime[:, 0] - bounds[0]) * float(IMG_SIZE) / (bounds[2] - bounds[0])
    pts_prime[:, 1] = (pts_prime[:, 1] - bounds[1]) * float(IMG_SIZE) / (bounds[3] - bounds[1])
    pts_prime = np.around(pts_prime).astype(np.long).clip(0, IMG_SIZE - 1)

    return pts_prime


def gen_edge_feature(pts_prime, im_feat, node_feat):
    """
    Generate edge semantic features
    :param pts_prime: coordinates of nodes
    :param im_feat: upsampled feature maps
    :param node_feat: node semantic features
    :return: edge semantic features
    """
    num_node = pts_prime.shape[0]
    edge_feat = torch.zeros(num_node * num_node, im_feat.shape[1])
    for i in range(num_node):
        cor_tail = pts_prime[i]
        for j in range(num_node):
            if i == j:
                edge_feat[i * num_node + j] = node_feat[i]
            else:
                cor_head = pts_prime[j]
                # Compute pixel points passed by current edge (including its tail and head)
                discrete_line = list(zip(*line(*cor_tail, *cor_head)))
                discrete_line = np.array(discrete_line, np.int)
                # Aggregate semantic features of these pixel points
                feat = im_feat[:, :, discrete_line[:, 1], discrete_line[:, 0]].to('cpu')
                edge_feat[i * num_node + j] = pool_feat(feat).squeeze()

    # L2 normalize by channel-wise
    edge_feat = featureL2norm(edge_feat)

    return edge_feat


def featureL2norm(feature):
    """
    Compute L2 normalized feature by channel-wise
    """
    norm = torch.norm(feature, dim=1, keepdim=True).expand_as(feature)
    return torch.div(feature, norm)


def parse_args():
    """
    Parse input arguments
    :return: parser
    """
    parser = argparse.ArgumentParser(description='GraphMatching Arguments')
    parser.add_argument('--split', dest='split', type=str, default='test',
                        help='Split of PascalVOC to use for generating edge semantic features')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    seed = 0
    rand = set_seed(seed)

    ''' Set vgg16 for extracting features '''
    vgg16_outputs = []


    def hook(module, x, y):
        vgg16_outputs.append(y)


    vgg16 = models.vgg16(pretrained=True)
    if torch.cuda.is_available():
        vgg16.cuda()
    vgg16.features[20].register_forward_hook(hook)  # relu4_2
    vgg16.features[25].register_forward_hook(hook)  # relu5_1

    # Splits of PascalVOC
    args = parse_args()
    sets = args.split
    cache_name = 'voc_db_{}.pkl'.format(sets)
    cache_path = Path(CACHE_PATH)
    cache_file = cache_path / cache_name
    if cache_file.exists():
        with cache_file.open(mode='rb') as f:
            xml_list = pickle.load(f)
        print('xml list loaded from {}'.format(cache_file))

    print('Initializing edge semantic features ...')
    vgg16.eval()
    with torch.no_grad():
        for category_id in range(0, len(VOC_CATEGORIES)):
            if not os.path.exists(os.path.join(VOC_FEATURE_ROOT, VOC_CATEGORIES[category_id])):
                os.makedirs(os.path.join(VOC_FEATURE_ROOT, VOC_CATEGORIES[category_id]))
            for xml_name in xml_list[category_id]:
                edgeFeatFile = feature_name(xml_name)
                if os.path.exists(os.path.join(VOC_FEATURE_ROOT, edgeFeatFile)):
                    continue

                xml_file = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_name)
                img_name, anno_names, anno_pts, bounds = _load_annotation(xml_file)
                if anno_pts.shape[0] >= 3:
                    pts = anno_pts.copy()
                    names = anno_names.copy()

                    img_name = os.path.join(VOC_IMAGE_ROOT, img_name + '.jpg')
                    img, bounds, im_size = get_image(img_name, bounds)
                    # Relocate keypoints in IMG_SIZE*IMG_SIZE
                    pts = relocate_pts(pts, bounds)

                    vgg16_outputs.clear()
                    vgg16(img)
                    im_feat_0 = F.interpolate(vgg16_outputs[0], IMG_SIZE, mode='bilinear', align_corners=False)
                    im_feat_1 = F.interpolate(vgg16_outputs[1], IMG_SIZE, mode='bilinear', align_corners=False)
                    im_feat = torch.cat([im_feat_0, im_feat_1], dim=1)

                    # Semantic features of nodes
                    node_feat = im_feat[0, :, pts[:, 1], pts[:, 0]].t().to('cpu')

                    # Semantic features of edges
                    edge_feat = gen_edge_feature(pts, im_feat, node_feat)
                    # Save features
                    torch.save(edge_feat, os.path.join(VOC_FEATURE_ROOT, edgeFeatFile))
                    # print('{} is done'.format(edgeFeatFile))
                # else:
                #     print('node number is less than 3!')

    print('Done!')