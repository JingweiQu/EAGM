import numpy as np
from scipy import spatial
from scipy.spatial import Delaunay
import scipy.io
import math
import os
import torch


CMUHouse_IMAGE_ROOT = "data/CMUHouse/images"
CMUHouse_FEATURE_ROOT = "data/CMUHouse/features"

features = torch.load(os.path.join(CMUHouse_FEATURE_ROOT, 'feature.pt'), map_location=lambda storage, loc: storage).numpy()

def _load_data_from_mat(mat_file) :
    data = scipy.io.loadmat(mat_file)
    XTs = data["XTs"]
    num_frames = XTs.shape[0] / 2
    num_points = XTs.shape[1]
    return num_frames, num_points, XTs


def _build_delaunay_graph(points, edge_feat, valid_idx, ori_node_num) :
    A = np.zeros(shape = (points.shape[0], points.shape[0]), dtype = np.float)
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

    tails_ori = valid_idx[tails]
    heads_ori = valid_idx[heads]
    edge_idx = tails_ori * ori_node_num + heads_ori
    edge_feat = edge_feat[edge_idx]

    edges = points[heads] - points[tails]
    angs = np.zeros(edges.shape[0], dtype = np.float)
    for i in range(edges.shape[0]):
        angs[i] = math.atan2(edges[i][1], edges[i][0])

    return tails, heads, dists, angs, edge_feat


def _normalize_coordinates(points):
    # Normalize the coordinates
    deviation = np.nanstd(points, axis=0)
    points = np.transpose(points)
    points[0] = points[0] / deviation[0]
    points[1] = points[1] / deviation[1]
    points = np.transpose(points)
    return points


def _gen_features_CMU(pts0, tails0, heads0,  e_feat0, n_feat0, pts1, tails1, heads1, e_feat1, n_feat1, gX):
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


def _load_feature(frame_idx, ori_node_num):
    # Edge semantic features
    feat_file = os.path.join(CMUHouse_FEATURE_ROOT, 'house.seq{}_EF.pt'.format(frame_idx))
    edge_feat = torch.load(feat_file, map_location=lambda storage, loc: storage).numpy()

    # Node semantic features
    node_feat = edge_feat[::ori_node_num+1]

    return edge_feat, node_feat


def _gen_random_graph(rand,
                      XTs,
                      frame_indexs,
                      num_inner_min_max) :
    max_nodes = 30
    num_nodes = [rand.randint(num_inner_min_max[0],num_inner_min_max[1]), max_nodes]

    # edge_feat0, node_feat0 = _load_feature(frame_indexs[0], max_nodes)
    # edge_feat1, node_feat1 = _load_feature(frame_indexs[1], max_nodes)

    edge_feat0 = features[frame_indexs[0]]
    node_feat0 = edge_feat0[::max_nodes + 1]

    edge_feat1 = features[frame_indexs[1]]
    node_feat1 = edge_feat1[::max_nodes + 1]

    points0 = XTs[frame_indexs[0] * 2 : frame_indexs[0] * 2 + 2][:]
    points1 = XTs[frame_indexs[1] * 2 : frame_indexs[1] * 2 + 2][:]

    points0 = np.transpose(points0)
    points1 = np.transpose(points1)

    points0 = _normalize_coordinates(points0)
    points1 = _normalize_coordinates(points1)

    # Randomly select nodes from frame points
    index0 = np.arange(0, max_nodes, 1)
    index1 = np.arange(0, max_nodes, 1)
    rand.shuffle(index0)
#    rand.shuffle(index1)
    index0 = index0[0:num_nodes[0]]
    index1 = index1[0:num_nodes[1]]
    points0 = points0[index0]
    points1 = points1[index1]

    node_feat0 = node_feat0[index0]
    node_feat1 = node_feat1[index1]

    # Record ground-truth matches
    gX = np.eye(XTs.shape[1])
    gX = np.transpose(np.transpose(gX[index0])[index1])

    tails0, heads0, dists0, angs0, edge_feat0 = _build_delaunay_graph(points0, edge_feat0, index0, max_nodes)
    tails1, heads1, dists1, angs1, edge_feat1 = _build_delaunay_graph(points1, edge_feat1, index1, max_nodes)
    assignGraph = _gen_features_CMU(points0, tails0, heads0, edge_feat0, node_feat0,
                                    points1, tails1, heads1, edge_feat1, node_feat1,
                                    gX)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    return assignGraph


def gen_random_graphs_CMU(rand,
                          num_examples,
                          num_inner_min_max,
                          num_outlier_min_max):
    mat_file = "graph_generator/cmuHouse.mat"
    num_frames, num_points, XTs =_load_data_from_mat(mat_file)

    # edge_feat = torch.load(os.path.join(CMUHouse_FEATURE_ROOT, 'feature.pt'), map_location=lambda storage, loc: storage).numpy()

    graphs = []
    max_frames = 111
    for _ in range(num_examples):
        gap = rand.randint(10, 101)
        frame1 = rand.randint(0, max_frames - gap)
        frame_indexs = (frame1, frame1 + gap)
        graph = _gen_random_graph(rand,
                                  XTs,
                                  frame_indexs,
                                  num_inner_min_max=num_inner_min_max)

        graphs.append(graph)

    return graphs