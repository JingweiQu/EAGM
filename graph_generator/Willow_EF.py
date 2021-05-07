# ========================================================================================
# Generate edge semantic features for Willow
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
from scipy import io
from skimage.draw import line

from transformation import GeometricTnf


WILLOW_FILE_ROOT = "Willow"
CATEGORIES = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]
IMG_SIZE = 256
affineTnf = GeometricTnf(geometric_model='affine', out_h=IMG_SIZE, out_w=IMG_SIZE, use_cuda=False)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
pool_feat = nn.AdaptiveMaxPool1d(1)


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # gpu
        # torch.cuda.manual_seed_all(seed) # all gpu
    rand = np.random.RandomState(seed=seed) # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = True  # Accelerate

    return rand


def _list_images(root, category) :
    imgFiles = []
    path = os.path.join(root, category)
    for file in os.listdir(path) :
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath) :
            if os.path.basename(filepath).endswith('.png') :
                imgFiles.append(filepath)

    return imgFiles


def get_image(img_name):
    image = cv2.imread(img_name)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)

    # Get image size, (H, W, C)
    im_size = np.asarray(image.shape)

    image = image[:, :, ::-1]
    image = torch.Tensor(image.astype(np.float32))
    image = image.permute(2, 0, 1)
    image.requires_grad = False
    image = affineTnf(image_batch=image.unsqueeze(0)).squeeze()
    image = image_normalize(image).unsqueeze(0)
    image = image.cuda()

    return image, im_size


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
    iLen = len(file)
    raw_file = file[:iLen - 4]
    anno_file = raw_file + ".mat"
    anno = io.loadmat(anno_file)
    pts = np.transpose(anno["pts_coord"])

    return pts


def _read_image_features(file):
    anno_pts = _load_annotation(file)

    anno_descs = []
    sift_pts = []
    sift_descs = []

    return anno_pts, anno_descs, sift_pts, sift_descs


def relocate_pts(pts_prime, im_size):
    pts_prime[:, 0] = pts_prime[:, 0] * float(IMG_SIZE) / im_size[1]
    pts_prime[:, 1] = pts_prime[:, 1] * float(IMG_SIZE) / im_size[0]
    pts_prime = np.around(pts_prime).astype(np.long).clip(0, IMG_SIZE - 1)

    return pts_prime


def gen_edge_feature(pts_prime, im_feat, node_feat):
    num_node = pts_prime.shape[0]
    edge_feat = torch.zeros(num_node*num_node, im_feat.shape[1])
    for i in range(num_node):
        cor_tail = pts_prime[i]
        for j in range(num_node):
            if i == j:
                edge_feat[i * num_node + j] = node_feat[i]
            else:
                cor_head = pts_prime[j]
                discrete_line = list(zip(*line(*cor_tail, *cor_head)))
                discrete_line = np.array(discrete_line, np.int)
                feat = im_feat[:, :, discrete_line[:, 1], discrete_line[:, 0]].to('cpu')
                edge_feat[i * num_node + j] = pool_feat(feat).squeeze()

    edge_feat = featureL2norm(edge_feat)

    return edge_feat


def featureL2norm(feature):
    """
    Compute L2 normalized feature by channel-wise
    """
    norm = torch.norm(feature, dim=1, keepdim=True).expand_as(feature)
    return torch.div(feature, norm)


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

    print('Initializing edge semantic features ...')
    vgg16.eval()
    with torch.no_grad():
        for k in range(len(CATEGORIES)):
            imgFiles = _list_images(os.path.join('../data', WILLOW_FILE_ROOT), CATEGORIES[k])
            for imgFile in imgFiles:
                edgeFeatFile = feature_name(imgFile)
                if os.path.exists(edgeFeatFile):
                    continue

                anno_pts, anno_descs, sift_pts, sift_descs = _read_image_features(imgFile)
                pts = anno_pts
                pts_prime = pts.copy()
                img, im_size = get_image(imgFile)
                pts_prime = relocate_pts(pts_prime, im_size)

                vgg16_outputs.clear()
                vgg16(img)
                im_feat_0 = F.interpolate(vgg16_outputs[0], IMG_SIZE, mode='bilinear', align_corners=False)
                im_feat_1 = F.interpolate(vgg16_outputs[1], IMG_SIZE, mode='bilinear', align_corners=False)
                im_feat = torch.cat([im_feat_0, im_feat_1], dim=1)

                node_feat = im_feat[0, :, pts_prime[:, 1], pts_prime[:, 0]].t().to('cpu')
                edge_feat = gen_edge_feature(pts_prime, im_feat, node_feat)

                torch.save(edge_feat, edgeFeatFile)
                # print('{} is done'.format(edgeFeatFile))

    print('Done!')