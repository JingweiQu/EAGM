# ========================================================================================
# Generate edge semantic features for CMU House
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
import scipy
from scipy import io
from skimage.draw import line

from transformation import GeometricTnf

CMUHouse_IMAGE_ROOT = "../data/CMUHouse/images"
CMUHouse_FEATURE_ROOT = "../data/CMUHouse/features"
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
    rand = np.random.RandomState(seed=seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = True  # Accelerate

    return rand


def _load_data_from_mat(mat_file) :
    data = scipy.io.loadmat(mat_file)
    XTs = data["XTs"]
    num_frames = XTs.shape[0] / 2
    num_points = XTs.shape[1]
    return num_frames, num_points, XTs


def get_image(img_name):
    image = cv2.imread(img_name)
    # If the image just has two channels, add one channel
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


def relocate_pts(pts_prime, im_size):
    pts_prime[:, 0] = pts_prime[:, 0] * float(IMG_SIZE) / im_size[1]
    pts_prime[:, 1] = pts_prime[:, 1] * float(IMG_SIZE) / im_size[0]
    pts_prime = np.around(pts_prime).astype(np.long).clip(0, IMG_SIZE - 1)

    return pts_prime


def gen_edge_feature(pts_prime, im_feat, node_feat):
    num_node = pts_prime.shape[0]
    edge_feat = torch.zeros(num_node * num_node, im_feat.shape[1])
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

    mat_file = "cmuHouse.mat"
    num_frames, num_points, XTs = _load_data_from_mat(mat_file)
    all_edge_feat = torch.zeros(int(num_frames), 900, 1024)

    print('Initializing edge semantic features ...')
    if not os.path.exists(CMUHouse_FEATURE_ROOT):
        os.makedirs(CMUHouse_FEATURE_ROOT)
    vgg16.eval()
    with torch.no_grad():
        for frame_id in range(int(num_frames)):
            edgeFeatFile = 'house.seq{}_EF.pt'.format(frame_id)
            # if os.path.exists(os.path.join(CMUHouse_FEATURE_ROOT, edgeFeatFile)):
            #     continue

            pts = XTs[frame_id * 2: frame_id * 2 + 2][:]
            pts = np.transpose(pts)

            img_name = 'house.seq{}.png'.format(frame_id)
            img, im_size = get_image(os.path.join(CMUHouse_IMAGE_ROOT, img_name))
            pts = relocate_pts(pts, im_size)

            vgg16_outputs.clear()
            vgg16(img)
            im_feat_0 = F.interpolate(vgg16_outputs[0], IMG_SIZE, mode='bilinear', align_corners=False)
            im_feat_1 = F.interpolate(vgg16_outputs[1], IMG_SIZE, mode='bilinear', align_corners=False)
            im_feat = torch.cat([im_feat_0, im_feat_1], dim=1)

            node_feat = im_feat[0, :, pts[:, 1], pts[:, 0]].t().to('cpu')

            edge_feat = gen_edge_feature(pts, im_feat, node_feat)
            all_edge_feat[frame_id] = edge_feat
            # torch.save(edge_feat, os.path.join(CMUHouse_FEATURE_ROOT, edgeFeatFile))
            # print('{} is done'.format(edgeFeatFile))

    torch.save(all_edge_feat, os.path.join(CMUHouse_FEATURE_ROOT, 'feature.pt'))
    print('Done!')