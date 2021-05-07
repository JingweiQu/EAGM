import numpy as np
from scipy import spatial
from scipy.spatial import Delaunay
import os
import math
import torch
import xml.dom.minidom
from graph_generator import pascal_voc

VOC_ANNOTATION_ROOT = "data/PascalVOC/annotations"
VOC_IMAGE_ROOT = "data/PascalVOC/JPEGImages"
VOC_FEATURE_ROOT = "data/PascalVOC/features"
VOC_CATEGORIES = ["aeroplane",   "bicycle", "bird",  "boat",      "bottle", "bus",         "car",   "cat",   "chair", "cow",
                  "diningtable", "dog",     "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",  "train", "tvmonitor"]
VOC_ANNO_LISTS= []


def _load_annotation(file) :
    dom = xml.dom.minidom.parse(file)
    root = dom.documentElement

    image = root.getElementsByTagName('image')[0].firstChild.data

    keypoints = root.getElementsByTagName('keypoints')[0]
    kps = keypoints.getElementsByTagName('keypoint')

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

    return image, annoName, annoPts


def _load_feature(file, valid_idx, ori_node_num):
    iLen = len(file)
    raw_file = file[:iLen - 4]

    # Edge semantic features
    feat_file = os.path.join(VOC_FEATURE_ROOT, raw_file + "_EF.pt")
    edge_feat = torch.load(feat_file, map_location=lambda storage, loc: storage).numpy()

    # Node semantic features
    node_feat = edge_feat[::ori_node_num+1]
    node_feat = node_feat[valid_idx]

    return edge_feat, node_feat


dataset_train = pascal_voc.PascalVOC('train', (256, 256))
dataset_test = pascal_voc.PascalVOC('test', (256, 256))


def _normalize_coordinates(points) :
    # Normalize by center
    center = np.sum(points, axis = 0) / points.shape[0]
    norm_points = np.transpose(points)
    norm_points[0] = norm_points[0] - center[0]
    norm_points[1] = norm_points[1] - center[1]

    # normalized by max_distance
    distance = spatial.distance.cdist(points, points)
    maxDst = np.max(distance) + 1e-6
    norm_points = norm_points / maxDst

    points = np.transpose(norm_points)

    return points


def _build_delaunay_graph(points, edge_feat, valid_idx, ori_node_num, perm=None) :
    A = np.zeros(shape = (points.shape[0], points.shape[0]), dtype = np.float)
    distance = spatial.distance.cdist(points, points)

    if points.shape[0] == 1:
        A[0, 0] = 1
    elif points.shape[0] <= 3:
        A = distance
    else:
        triangles = Delaunay(points).simplices
        for tri in triangles:
            A[tri[0]][tri[1]] = distance[tri[0]][tri[1]]
            A[tri[0]][tri[2]] = distance[tri[0]][tri[2]]
            A[tri[1]][tri[2]] = distance[tri[1]][tri[2]]
            A[tri[1]][tri[0]] = A[tri[0]][tri[1]]
            A[tri[2]][tri[0]] = A[tri[0]][tri[2]]
            A[tri[2]][tri[1]] = A[tri[1]][tri[2]]

    idxs = np.nonzero(A)
    dists = A[idxs]
    tails = idxs[0]
    heads = idxs[1]

    if perm is None:
        tails_ori = valid_idx[tails]
        heads_ori = valid_idx[heads]
    else:
        tails_ori = valid_idx[perm[tails]]
        heads_ori = valid_idx[perm[heads]]
    edge_idx = tails_ori * ori_node_num + heads_ori
    edge_feat = edge_feat[edge_idx]

    edges = points[heads] - points[tails]
    angs = np.zeros(edges.shape[0], dtype = np.float)
    for i in range(edges.shape[0]):
        angs[i] = math.fabs(math.atan(edges[i][1] / (edges[i][0] + 1e-16)))

    return tails, heads, dists, angs, edge_feat


def _gen_features_VOC(pts0, tails0, heads0, e_feat0, n_feat0, pts1, tails1, heads1, e_feat1, n_feat1, gX):
    num_nodes0 = pts0.shape[0]
    num_nodes1 = pts1.shape[0]
    num_edges0 = len(tails0)
    num_edges1 = len(tails1)
    num_matches = num_nodes0 * num_nodes1

    gidx1 = np.zeros(num_matches, np.int)
    gidx2 = np.zeros(num_matches, np.int)
    for i in range(num_matches):
        gidx1[i] = i / num_nodes1
        gidx2[i] = i % num_nodes1

    node_feaLen = 4
    node_fea2Len = n_feat0.shape[1] + n_feat1.shape[1]
    edge_feaLen = 8
    edge_fea2Len = e_feat0.shape[1] + e_feat1.shape[1]
    num_assGraph_nodes = num_matches
    num_assGraph_edges = num_edges0 * num_edges1
    senders = np.zeros(num_assGraph_edges, np.int)
    receivers = np.zeros(num_assGraph_edges, np.int)
    edge_features = np.zeros((num_assGraph_edges, edge_feaLen), np.float)
    edge_features2 = np.zeros((num_assGraph_edges, edge_fea2Len), np.float)
    edge_solution = np.zeros((num_assGraph_edges, 1), np.float)
    node_features = np.zeros((num_assGraph_nodes, node_feaLen), np.float)
    node_features2 = np.zeros((num_assGraph_nodes, node_fea2Len), np.float)

    for i in range(num_matches):
        cor_node0 = pts0[gidx1[i]]
        cor_node1 = pts1[gidx2[i]]
        node_features[i] = np.hstack((cor_node0, cor_node1))

        feat_node0 = n_feat0[gidx1[i]]
        feat_node1 = n_feat1[gidx2[i]]
        node_features2[i] = np.hstack((feat_node0, feat_node1))

    idx = 0
    for i in range(num_edges0):
        cor_tail0 = pts0[tails0[i]]
        cor_head0 = pts0[heads0[i]]
        for k in range(num_edges1):
            cor_tail1 = pts1[tails1[k]]
            cor_head1 = pts1[heads1[k]]

            senders[idx] = tails0[i] * num_nodes1 + tails1[k]
            receivers[idx] = heads0[i] * num_nodes1 + heads1[k]
            edge_features[idx] = np.hstack((cor_tail0, cor_head0, cor_tail1, cor_head1)) # Geometric feature
            edge_features2[idx] = np.hstack((e_feat0[i], e_feat1[k]))  # Semantic feature

            edge_solution[idx] = gX[tails0[i], tails1[k]] * gX[heads0[i], heads1[k]]

            idx = idx + 1

    assignGraph = {"gidx1": gidx1,
                   "gidx2": gidx2,
                   "node_features": node_features,
                   "node_features2": node_features2,
                   "senders": senders,
                   "receivers": receivers,
                   "edge_features": edge_features,
                   "edge_features2": edge_features2,
                   "edge_solution": edge_solution}

    return assignGraph


def _remove_unmatched_points(anno_names0, anno_pts0,
                             anno_names1, anno_pts1):
    valid0 = np.zeros(anno_names0.shape[0], dtype = np.bool)
    valid1 = np.zeros(anno_names1.shape[0], dtype = np.bool)

    for i in range(anno_names0.shape[0]):
        for k in range(anno_names1.shape[0]):
            if anno_names0[i] == anno_names1[k]:
                valid0[i] = True
                valid1[k] = True
                break

    anno_names0 = anno_names0[valid0]
    anno_pts0 = anno_pts0[valid0]
    valid_idx0 = np.where(valid0 == True)[0]

    anno_names1 = anno_names1[valid1]
    anno_pts1 = anno_pts1[valid1]
    valid_idx1 = np.where(valid1 == True)[0]

    return anno_names0, anno_pts0, valid_idx0, anno_names1, anno_pts1, valid_idx1


def _gen_random_graph(rand,
                      use_train_set,
                      category_id,
                      num_outlier_min_max):
    while True:
        if use_train_set:
            xml_files = dataset_train.get_xml_pair(category_id)
        else:
            xml_files = dataset_test.get_xml_pair(category_id)
        xml_file0 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[0])
        xml_file1 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[1])

        image0, anno_names0, anno_pts0 = _load_annotation(xml_file0)
        image1, anno_names1, anno_pts1 = _load_annotation(xml_file1)
        # Original number of nodes
        ori_node_num0 = anno_pts0.shape[0]
        ori_node_num1 = anno_pts1.shape[0]

        # Remove unmatched points
        anno_names0, anno_pts0, valid_idx0, anno_names1, anno_pts1, valid_idx1 = _remove_unmatched_points(
            anno_names0, anno_pts0, anno_names1, anno_pts1)

        if anno_pts0.shape[0] >= 3 and anno_pts1.shape[0] >= 3:
            break

    edge_feat0, node_feat0 = _load_feature(xml_files[0], valid_idx0, ori_node_num0)
    edge_feat1, node_feat1 = _load_feature(xml_files[1], valid_idx1, ori_node_num1)

    pts0 = anno_pts0.copy()
    pts1 = anno_pts1.copy()
    names0 = anno_names0.copy()
    names1 = anno_names1.copy()

    # Randomly re-order
    index0 = np.arange(0, pts0.shape[0])
    # rand.shuffle(index0)
    # pts0 = pts0[index0]
    # names0 = names0[index0]

    index1 = np.arange(0, pts1.shape[0])
    rand.shuffle(index1)
    pts1 = pts1[index1]
    names1 = names1[index1]
    node_feat1 = node_feat1[index1]

    # Normalize point coordinates
    pts0 = _normalize_coordinates(pts0)
    pts1 = _normalize_coordinates(pts1)

    # Record ground-truth matches
    gX = np.eye(pts1.shape[0])
    gX = np.transpose(np.transpose(gX[index0])[index1])

    tails0, heads0, dists0, angs0, edge_feat0 = _build_delaunay_graph(pts0, edge_feat0, valid_idx0, ori_node_num0, None)
    tails1, heads1, dists1, angs1, edge_feat1 = _build_delaunay_graph(pts1, edge_feat1, valid_idx1, ori_node_num1, index1)

    assignGraph = _gen_features_VOC(pts0, tails0, heads0, edge_feat0, node_feat0,
                                    pts1, tails1, heads1, edge_feat1, node_feat1,
                                    gX)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    image = {"category": VOC_CATEGORIES[category_id],
             "image1": xml_file0,
             "image2": xml_file1}

    return assignGraph, image


def gen_random_graphs_VOC(rand,
                          num_examples,
                          num_inner_min_max,
                          num_outlier_min_max,
                          use_train_set = True,
                          category_id = -1):

    graphs = []
    images = []
    for _ in range(num_examples):
        if category_id < 0:
            cid = rand.randint(0, len(VOC_CATEGORIES))
        else:
            cid = category_id
        graph, image = _gen_random_graph(rand,
                                         use_train_set,
                                         cid,
                                         num_outlier_min_max=num_outlier_min_max)
        graphs.append(graph)
        images.append(image)

    return graphs, images