import numpy as np
from scipy import spatial
from scipy.spatial import Delaunay
import scipy.io
import os
import math
import torch

WILLOW_FILE_ROOT = "data/Willow"
WILLOW_TRAIN_NUM = 20
WILLOW_TRAIN_OFFSET = 0


def _list_images(root, category):
    imgFiles = []
    path = os.path.join(root, category)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath):
            if os.path.basename(filepath).endswith('.png'):
                imgFiles.append(filepath)

    return imgFiles


def _load_annotation(file):
    iLen = len(file)
    raw_file = file[:iLen - 4]
    anno_file = raw_file + ".mat"
    anno = scipy.io.loadmat(anno_file)
    pts = np.transpose(anno["pts_coord"])

    # Edge semantic features
    feat_file = raw_file + "_EF.pt"
    edge_feat = torch.load(feat_file, map_location=lambda storage, loc: storage).numpy()

    # Node semantic features
    node_feat = edge_feat[::pts.shape[0] + 1]

    return pts, edge_feat, node_feat


def _read_image_features(file):
    anno_pts, edge_feat, node_feat = _load_annotation(file)

    anno_descs = []
    sift_pts = []
    sift_descs = []

    return anno_pts, anno_descs, sift_pts, sift_descs, edge_feat, node_feat


def _normalize_coordinates(points):
    # Normalize the coordinates
    deviation = np.nanstd(points, axis=0)
    points = np.transpose(points)
    points[0] = points[0] / deviation[0]
    points[1] = points[1] / deviation[1]
    points = np.transpose(points)
    return points


def _build_delaunay_graph(points, edge_feat, node_num, perm):
    A = np.zeros(shape=(points.shape[0], points.shape[0]), dtype=np.float)
    distance = spatial.distance.cdist(points, points)

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

    tails_ori = perm[tails]
    heads_ori = perm[heads]
    edge_idx = tails_ori * node_num + heads_ori
    edge_feat = edge_feat[edge_idx]

    edges = points[heads] - points[tails]
    angs = np.zeros(edges.shape[0], dtype=np.float)
    for i in range(edges.shape[0]):
        angs[i] = math.atan2(edges[i][1], edges[i][0])

    return tails, heads, dists, angs, edge_feat


def _gen_features_Willow(pts0, tails0, heads0, e_feat0, n_feat0, pts1, tails1, heads1, e_feat1, n_feat1, gX):
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
            edge_features[idx] = np.hstack((cor_tail0, cor_head0, cor_tail1, cor_head1)) # Geometric features
            edge_features2[idx] = np.hstack((e_feat0[i], e_feat1[k]))  # Semantic features

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


def _gen_random_graph(rand,
                      category,
                      num_outlier_min_max,
                      use_train_set=True):
    imgFiles = _list_images(WILLOW_FILE_ROOT, category)
    if use_train_set:
        imgIdx = rand.randint(0, WILLOW_TRAIN_NUM, size=2)
    else:
        imgIdx = rand.randint(WILLOW_TRAIN_NUM, len(imgFiles), size=2)

    anno_pts0, anno_descs0, sift_pts0, sift_descs0, edge_feat0, node_feat0 = _read_image_features(imgFiles[imgIdx[0]])
    anno_pts1, anno_descs1, sift_pts1, sift_descs1, edge_feat1, node_feat1 = _read_image_features(imgFiles[imgIdx[1]])

    pts0 = anno_pts0.copy()
    pts1 = anno_pts1.copy()

    # Randomly re-order
    index0 = np.arange(0, pts0.shape[0])
    rand.shuffle(index0)
    pts0 = pts0[index0]
    node_feat0 = node_feat0[index0]

    index1 = np.arange(0, pts1.shape[0])
    rand.shuffle(index1)
    pts1 = pts1[index1]
    node_feat1 = node_feat1[index1]

    # Normalize point coordinates
    pts0 = _normalize_coordinates(pts0)
    pts1 = _normalize_coordinates(pts1)

    # Record ground-truth matches
    gX = np.eye(pts1.shape[0])
    gX = np.transpose(np.transpose(gX[index0])[index1])

    node_num = pts0.shape[0]
    tails0, heads0, dists0, angs0, edge_feat0 = _build_delaunay_graph(pts0, edge_feat0, node_num, index0)
    tails1, heads1, dists1, angs1, edge_feat1 = _build_delaunay_graph(pts1, edge_feat1, node_num, index1)

    assignGraph = _gen_features_Willow(pts0, tails0, heads0, edge_feat0, node_feat0,
                                       pts1, tails1, heads1, edge_feat1, node_feat1,
                                       gX)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)):
        if gX[gidx1[i]][gidx2[i]]:
            solutions[i] = True
    assignGraph["solutions"] = solutions

    image = {"category": category,
             "image1": imgFiles[imgIdx[0]],
             "image2": imgFiles[imgIdx[1]]}

    return assignGraph, image


def gen_random_graphs_Willow(rand,
                             num_examples,
                             num_inner_min_max,
                             num_outlier_min_max,
                             use_train_set=True,
                             category_id=-1):
    categories = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

    graphs = []
    images = []
    for _ in range(num_examples):
        if category_id < 0:
            cid = rand.randint(0, 5)
        else:
            cid = category_id
        graph, image = _gen_random_graph(rand,
                                         categories[cid],
                                         num_outlier_min_max=(0,1),
                                         use_train_set = use_train_set)
        graphs.append(graph)
        images.append(image)

    return graphs, images