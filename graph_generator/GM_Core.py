#@title ##### License
# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


#@title Imports  { form-width: "30%" }

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf

from ctypes import *
import numpy.ctypeslib as npct


###################
# Find the shortest path in a graph\n",
#    "This notebook and the accompanying code demonstrates how to use the Graph Nets library to learn to predict the shortest path between two nodes in graph.\n",
#
#    "The network is trained to label the nodes and edges of the shortest path, given the start and end nodes.\n",
#
#    "After training, the network's prediction ability is illustrated by comparing its output to the true shortest path.
#    Then the network's ability to generalise is tested, by using it to predict the shortest path in similar but larger graphs."
################


#@title ### Install the Graph Nets library on this Colaboratory runtime  { form-width: "60%", run: "auto"}
#@markdown <br>1. Connect to a local or hosted Colaboratory runtime by clicking the **Connect** button at the top-right.<br>2.
# Choose "Yes" below to install the Graph Nets library on the runtime machine with:<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# ```pip install graph_nets```<br> Note, this works both with local and hosted Colaboratory runtimes.

install_graph_nets_library = "No"  #@param ["Yes", "No"]

if install_graph_nets_library.lower() == "yes":
  print("Installing Graph Nets library with:")
  print("  $ pip install graph_nets\n")
  print("Output message from command:\n")
#  !pip install graph_nets
else:
  print("Skipping installation of Graph Nets library")

#  "If you are running this notebook locally (i.e., not through Colaboratory), you will also need to install a few more dependencies.
#  Run the following on the command line to install the graph networks library, as well as a few other dependencies:\n",
#
#  "pip install graph_nets matplotlib scipy\n",
# =================================

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#@title Helper functions  { form-width: "30%" }

# pylint: disable=redefined-outer-name

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.
NODE_ATTRIBUTE_NAME = "node_attr"

MAX_NODES_PER_GRAPH = 100           # maximum of number of nodes in each graph

def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)


def set_diff(seq0, seq1):
  """Return the set difference between 2 sequences as a list."""
  return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
  one_hot = np.eye(max_value)[indices]
  if axis not in (-1, one_hot.ndim):
    one_hot = np.moveaxis(one_hot, -1, axis)
  return one_hot


def get_node_dict(graph, attr):
  """Return a `dict` of node:attribute pairs from a graph."""
  return {k: v[attr] for k, v in graph.node.items()}


def generate_2DPoints(rand,
             nInner = 10,
             nOutlier = 0,
             rho = 0.0,        # deformation range
             noise = 0.0, # node noise range
             angle = 0.0,
             scale = 1.0):
    # parameters
    rgSize = 256 * np.sqrt((nInner + nOutlier) / 10)

    # generate inner points
    kps1 = {}
    kps2 = {}
    kps1["coords"] = rgSize * rand.uniform(size = (nInner, 2)) - (rgSize / 2)
    kps1["weights"] = rand.exponential(1.0, size = nInner)

    # white guassian noise for node weights
    kps2["weights"] = kps1["weights"] + rand.uniform(-noise, noise, size=nInner)

    # deform for inner points coords
    deform = rand.uniform(-rho, rho, size = (nInner, 2))
    kps2["coords"] = kps1["coords"] + deform
    # scaling and rotation
    A = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    kps2["coords"] = np.transpose(np.dot(scale * A, np.transpose(kps2["coords"])))

    # generate outlier points in graph2 only
    if nOutlier > 0 :
        coords2 = scale * (rgSize * rand.uniform(size = (nOutlier, 2)) - (rgSize / 2))
        weights2 = rand.exponential(1.0, size = nOutlier)
        kps2["coords"] = np.vstack((kps2["coords"], coords2))
        kps2["weights"] = np.hstack((kps2["weights"], weights2))

    # randomly re-order generated points
    pos_index = np.arange(0, nInner + nOutlier, 1)
    rand.shuffle(pos_index)
    kps2["coords"] = kps2["coords"][pos_index]
    kps2["weights"] = kps2["weights"][pos_index]

    # set ground-truth
    X = np.zeros((nInner + nOutlier, nInner + nOutlier), np.bool)
    for i in range(nInner + nOutlier) :
        if pos_index[i] < nInner :
            X[pos_index[i], i] = True

    return kps1, kps2, X


def generate_affinity_2DPoints(kps1, kps2, X, noise) :
    n1 = kps1["coords"].shape[0]
    n2 = kps2["coords"].shape[0]

    distance1 = spatial.distance.cdist(kps1["coords"], kps1["coords"])
    distance2 = spatial.distance.cdist(kps2["coords"], kps2["coords"])

    # filter candidate matches, remove matches with large node distance
    w1 = np.tile(kps1["weights"].reshape(n1, 1), n2)
    w2 = np.tile(kps2["weights"], (n1, 1))
    w_dist = abs(w1 - w2)

    # generate group and ground-truth label
    gidx1 = []
    gidx2 = []
    solutions = []
    index = 0
    for i2 in range(n2):
        for i1 in range(n1):
        #    if w_dist[i1, i2] <= noise:
                gidx1.append(i1)
                gidx2.append(i2)
                solutions.append(X[i1, i2])
                index = index + 1

    num_matches = index

    alpha = 0.0

    # count = 0
    # rho_d = 5.0
    # # generate affinity matrix
    # K = np.zeros((num_matches, num_matches), np.float)
    # for row in range(num_matches) :
    #     i1 = gidx1[row]
    #     i2 = gidx2[row]
    #
    #     for col in range(num_matches) :
    #         j1 = gidx1[col]
    #         j2 = gidx2[col]
    #
    #         d1 = distance1[i1, j1]
    #         d2 = distance2[i2, j2]
    #
    #         if abs(d1 - d2) < 3 * rho_d :
    #             node_aff = max(0.0, 1.0 - (w_dist[i1, i2] + w_dist[j1, j2]))
    #             edge_eff = 4.5 - (d1 - d2) * (d1 - d2) / (2 * rho_d * rho_d)
    #             K[row, col] = alpha * node_aff + (1.0 - alpha) * edge_eff
    #             count = count + 1

    lib = npct.load_library("GMBuilder.dll",".")  #引入动态链接库，load_library的用法可参考官网文档
    lib.build_affinity_2DPoints.argtypes = [c_int,
                                            c_int,
                                            npct.ndpointer(dtype = np.float32, ndim = 1, flags = "C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                            c_int,
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            c_float,
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]

    dist1 = np.float32(np.reshape(distance1, n1 * n1, order = 'C'))
    dist2 = np.float32(np.reshape(distance2, n2 * n2, order = 'C'))
    wdist = np.float32(np.reshape(w_dist, n1 * n2, order = 'C'))
    g1 = np.array(gidx1)
    g2 = np.array(gidx2)
    K = np.zeros(num_matches * num_matches, np.float32)
    count = lib.build_affinity_2DPoints(n1, n2, dist1, dist2, wdist, num_matches, g1, g2, alpha, K)
    K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))

    return K,gidx1, gidx2, solutions


def generate_2DPoints_graph(rand,
                   num_inner_min_max,
                   num_outlier_min_max,
                   rho = 0.0,
                   noise = 0.0,
                   angle = 0.0,
                   scale = 1.0):
  """Creates a geographic threshold graph.

  Args:
    rand: A random seed for the graph generator. Default= None.
    num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
    dimensions: (optional) An `int` number of dimensions for the positions.
      Default= 2.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Large values (1000+) make mostly trees. Try
      20-60 for good non-trees. Default=1000.0.
    rate: (optional) A rate parameter for the node weight exponential sampling
      distribution. Default= 1.0.

  Returns:
    The graph.
  """

  # generate 2D points
  nInner = rand.randint(*num_inner_min_max)
  nOutlier = rand.randint(*num_outlier_min_max)

  kps1, kps2, X  = generate_2DPoints(rand, nInner, nOutlier,
                                    rho, noise, angle, scale)

  K, gidx1, gidx2, solutions = generate_affinity_2DPoints(kps1, kps2, X, noise)

  G = generate_graph_by_affinity(K, gidx1, gidx2, solutions)

  return G


def generate_graph_geo(rand,
                       num_nodes_min_max,
                       num_outliers,
                       deform_pos=0.2,
                       deform_weight=0.0,
                       #  deform_weight = 0.2,
                       dimensions=2,
                       theta=1000.0,
                       rate=1.0) :
  #Creates a geographic threshold graph.

  num_nodes1 = rand.randint(*num_nodes_min_max)

  # Create geographic threshold graph.
  pos_array1 = rand.uniform(size=(num_nodes1, dimensions))
  pos1 = dict(enumerate(pos_array1))
  weight_node1 = rand.exponential(rate, size=num_nodes1)
  weight1 = dict(enumerate(weight_node1))
  geo_graph1 = nx.geographical_threshold_graph(
      num_nodes1, theta, pos=pos1, weight=weight1)

  # Put all distance weights into edge attributes.
  distances1 = spatial.distance.squareform(spatial.distance.pdist(pos_array1))
  for i, j in geo_graph1.edges():
    geo_graph1.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME,
                                                  distances1[i, j])


  # generate graph2 by deforming graph1 with noise
  num_nodes2 = num_nodes1 + num_outliers
  pos_deform = pos_array1 + rand.uniform(-deform_pos, deform_pos, size=(num_nodes1, dimensions))
  pos_outliers = rand.uniform(size=(num_outliers, dimensions))
  pos_array2 = np.vstack((pos_deform, pos_outliers))

  weight_deform = weight_node1 + rand.exponential(deform_weight * rate, size = num_nodes1)
  weight_outliers = rand.exponential(rate, size = num_outliers)
  weight_node2 = np.hstack((weight_deform, weight_outliers))

  # randomly shuffle the pos array
  pos_index = np.arange(0, num_nodes2, 1)
  rand.shuffle(pos_index)

  pos_array2 = pos_array2[pos_index]
  pos2 = dict(enumerate(pos_array2))

  weight_node2 = weight_node2[pos_index]
  weight2 = dict(enumerate(weight_node2))

  geo_graph2 = nx.geographical_threshold_graph(
      num_nodes2, theta, pos=pos2, weight=weight2)

  # Put all distance weights into edge attributes.
  distances2 = spatial.distance.squareform(spatial.distance.pdist(pos_array2))
  for i, j in geo_graph2.edges():
    geo_graph2.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME,
                                                  distances2[i, j])


  # filter candidate matches, reserve matches with small node distance
  w1 = np.tile(weight_node1, (num_nodes2, 1))
  w2 = np.tile(weight_node2.reshape(num_nodes2, 1), num_nodes1)
  w_dist = abs(w1 - w2)
  dist = w_dist.copy().reshape(1, num_nodes1 * num_nodes2)
  dist.sort()

  num_matches = int(num_nodes1 * np.sqrt(num_nodes2))
  threshold = dist[0, num_matches - 1]

  group1 = []
  group2 = []
  solutions = []
  index = 0
  for r in range(num_nodes2):
      for c in range(num_nodes1):
          if w_dist[r,c] <= threshold:
              group1.append(c)
              group2.append(r)

              if pos_index[r] == c :
                 solutions.append(True)
              else:
                  solutions.append(False)

              index = index + 1

  num_matches = index


  # generate affinity matrix
  alpha = 0.5
  K = np.zeros((num_matches, num_matches), np.float)
  for mi in range(num_matches):
      node_i1 = group1[mi]
      node_i2 = group2[mi]
      for mj in range(num_matches):
          node_j1 = group1[mj]
          node_j2 = group2[mj]
          if node_i1 == node_j1 and node_i2 == node_j2:
              K[mi, mj] = 1.0 - w_dist[node_i2, node_i1]
          else:
              node_dist = 0.5 * (w_dist[node_i2, node_i1] + w_dist[node_j2, node_j1])
              edge_dist = abs(distances1[node_i1, node_j1] - distances2[node_i2, node_j2])
              match_dist = alpha * node_dist + (1.0 - alpha) * edge_dist
              K[mi, mj] = 1.0 - match_dist

          if K[mi, mj] < 0.0:
              K[mi, mj] = 0.0

  return K, group1, group2, solutions


from sklearn import preprocessing

def generate_graph_by_affinity(K, gidx1, gidx2, solutions):
    # data normalization
    scaler = preprocessing.MinMaxScaler()
    K = scaler.fit_transform(K)

    nRow = K.shape[0]
    nCol = K.shape[1]

    last_time = time.time()

    G = nx.empty_graph()
    for i in range(nRow):
        G.add_node(i, affinity = K[i,i], gidx1 = gidx1[i], gidx2 = gidx2[i], solution = solutions[i])

    t0 = time.time() - last_time
    last_time = time.time()

    # idx = np.nonzero(K)
    # v = K[idx]
    # edge_lists = []
    # for i in range(len(v)) :
    #     edge_lists.append((idx[0][i], idx[1][i], v[i]))
    # G.add_weighted_edges_from(edge_lists, "affinity", solution = False)

    for r in range(nRow):
        for c in range(r + 1, nRow):
            if K[r,c] > 0.0 :
                G.add_edge(r, c, affinity = K[r,c], solution = False)

    t1 = time.time() - last_time
    return G



def graph_to_input_target(graph):
  """Returns 2 graphs with input and target feature vectors for training.

  Args:
    graph: An `nx.DiGraph` instance.

  Returns:
    The input `nx.DiGraph` instance.
    The target `nx.DiGraph` instance.

  Raises:
    ValueError: unknown node type
  """

  def create_feature(attr, fields):
     return np.hstack([np.array(attr[field], dtype=float) for field in fields])

  input_node_fields = ("affinity", "gidx1", "gidx2")
  input_edge_fields = ("affinity",)
  target_node_fields = ("solution",)
  target_edge_fields = ("solution",)

  input_graph = graph.copy()
  target_graph = graph.copy()

  solution_length = 0
  gidx1 = []
  gidx2 = []
  node_affinity = []
  node_solution = []
  for node_index, node_feature in graph.nodes(data=True):
    input_graph.add_node(
        node_index, features=create_feature(node_feature, input_node_fields))

    target_node = create_feature(node_feature, target_node_fields).astype(float)
    #target_node = to_one_hot(
    #    create_feature(node_feature, target_node_fields).astype(int), 2)[0]

    target_graph.add_node(node_index, features=target_node)
    solution_length += int(node_feature["solution"])

    gidx1 = np.hstack((gidx1, node_feature["gidx1"]))
    gidx2 = np.hstack((gidx2, node_feature["gidx2"]))
    node_solution = np.hstack((node_solution, node_feature["solution"]))
    node_affinity = np.hstack((node_affinity, node_feature["affinity"]))

  solution_length /= graph.number_of_nodes()

  for receiver, sender, features in graph.edges(data=True):
    input_graph.add_edge(
        sender, receiver, features=create_feature(features, input_edge_fields))

    target_edge = create_feature(features, target_edge_fields).astype(float)
    #target_edge = to_one_hot(
    #    create_feature(features, target_edge_fields).astype(int), 2)[0]

    target_graph.add_edge(sender, receiver, features=target_edge)


  input_graph.graph["features"] = np.array([0.0])
  target_graph.graph["features"] = np.array([solution_length], dtype=float)

  # generate groups info for target_graph
  n_group1 = np.int(np.max(gidx1) + 1)
  n_group2 = np.int(np.max(gidx2) + 1)

  groups1 = np.zeros((n_group1,1), dtype=np.float)
  groups2 = np.zeros((n_group2,1), dtype=np.float)
  gidx1 = np.int32(gidx1)
  gidx2 = np.int32(gidx2)
  for i in range(len(node_affinity)):
      if node_solution[i] :
        groups1[gidx1[i]] = groups1[gidx1[i]] + node_affinity[i]
        groups2[gidx2[i]] = groups2[gidx2[i]] + node_affinity[i]

  input_graph.graph["groups1"] = np.zeros((n_group1,1), dtype=np.float)
  input_graph.graph["groups2"] = np.zeros((n_group2,1), dtype=np.float)
  target_graph.graph["groups1"] = groups1
  target_graph.graph["groups2"] = groups2

  return input_graph, target_graph


def generate_networkx_graphs(rand,
                             num_examples,
                             num_inner_min_max,
                             num_outlier_min_max):
  """Generate graphs for training.

  Args:
    rand: A random seed (np.RandomState instance).
    num_examples: Total number of graphs to generate.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Default= the number of nodes.

  Returns:
    input_graphs: The list of input graphs.
    target_graphs: The list of output graphs.
    graphs: The list of generated graphs.
  """
  input_graphs = []
  target_graphs = []
  graphs = []
  for _ in range(num_examples):
      rho = rand.randint(0, 10, size = (1))
      noise = rand.uniform(0.2, 0.5, size = (1))
   #   rho = 10
   #   scale = 0.5 + rand.uniform(size = (1))

      graph = generate_2DPoints_graph(rand,
            num_inner_min_max = num_inner_min_max,
            num_outlier_min_max = num_outlier_min_max,
            rho = rho,
            noise = noise)

      input_graph, target_graph = graph_to_input_target(graph)
      input_graphs.append(input_graph)
      target_graphs.append(target_graph)
      graphs.append(graph)


  return input_graphs, target_graphs, graphs


def create_placeholders(rand,
                        batch_size,
                        num_inner_min_max,
                        num_outlier_min_max):
  """Creates placeholders for the model training and evaluation.

  Args:
    rand: A random seed (np.RandomState instance).
    batch_size: Total number of graphs per batch.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: A `float` threshold parameters for the geographic threshold graph's
      threshold. Default= the number of nodes.

  Returns:
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.
  """
  # Create some example data for inspecting the vector sizes.
  input_graphs, target_graphs, _ = generate_networkx_graphs(
      rand, batch_size, num_inner_min_max, num_outlier_min_max)
  input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
  target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
  return input_ph, target_ph


def create_feed_dict(rand,
                     batch_size,
                     num_inner_min_max,
                     num_outlier_min_max,
                     input_ph,
                     target_ph,
                     loss_cof_ph,
                     cof):
  """Creates feed_dict for the model training and evaluation.

  Args:
    rand: A random seed (np.RandomState instance).
    batch_size: Total number of graphs per batch.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: A `float` threshold parameters for the geographic threshold graph's
      threshold. Default= the number of nodes.
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.

  Returns:
    feed_dict: The feed `dict` of input and target placeholders and data.
    raw_graphs: The `dict` of raw networkx graphs.
  """
  inputs, targets, raw_graphs = generate_networkx_graphs(
      rand, batch_size, num_inner_min_max, num_outlier_min_max)
  input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
  target_graphs = utils_np.networkxs_to_graphs_tuple(targets)

  feed_dict = {input_ph: input_graphs,
               target_ph: target_graphs}
#  feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph : cof}
  return feed_dict, raw_graphs


def greedy_mapping(nodes, group_indices):
    x = np.zeros(shape = nodes.shape, dtype = np.int)

    while True:
        idx = np.argmax(nodes)
        if nodes[idx] <= 0.0 :
            break

        nodes[idx] = 0.0
        x[idx] = 1

        gidx = group_indices[idx]

        for i in range(len(nodes)):
            if group_indices[i] == gidx :
                nodes[i] = 0.0

    return x

def compute_accuracy(target, output, use_nodes=True, use_edges=False):
  """Calculate model accuracy.

  Returns the number of correctly predicted shortest path nodes and the number
  of completely solved graphs (100% correct predictions).

  Args:
    target: A `graphs.GraphsTuple` that contains the target graph.
    output: A `graphs.GraphsTuple` that contains the output graph.
    use_nodes: A `bool` indicator of whether to compute node accuracy or not.
    use_edges: A `bool` indicator of whether to compute edge accuracy or not.

  Returns:
    correct: A `float` fraction of correctly labeled nodes/edges.
    solved: A `float` fraction of graphs that are completely correctly labeled.

  Raises:
    ValueError: Nodes or edges (or both) must be used
  """
  if not use_nodes and not use_edges:
    raise ValueError("Nodes or edges (or both) must be used")
  tdds = utils_np.graphs_tuple_to_data_dicts(target)
  odds = utils_np.graphs_tuple_to_data_dicts(output)
  cs_all = []
  ss_all = []
  cs_gt = []
  for td, od in zip(tdds, odds):
    xn = td["nodes"].astype(np.int)
    yn = greedy_mapping(od["nodes"], od["group_indices_1"])

    c_all = (xn == yn)
    s_all = np.all(c_all)
    cs_all.append(c_all)
    ss_all.append(s_all)

    c_gt = 0
    for i in range(len(xn)):
        if xn[i] == 1 and xn[i] == yn[i] :
            c_gt = c_gt + 1
    c_gt = c_gt / np.sum(xn)
    cs_gt.append(c_gt)


  correct_gt = np.mean(np.array(cs_gt))
  correct_all = np.mean(np.concatenate(cs_all, axis=0))
  solved = np.mean(np.stack(ss_all))
  return correct_gt, correct_all, solved


def create_loss_ops(target_op, output_ops, loss_cof_ph):
#  loss_ops = [
#      tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
#      tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
#      for output_op in output_ops
#  ]


#  loss_ops = [
#      tf.losses.softmax_cross_entropy(np.matmul(np.transpose(target_op.nodes), group_op), np.matmul(np.transpose(output_op.nodes), group_op)) +
#      tf.losses.mean_squared_error(target_op.nodes, output_op.nodes)
#      for output_op in output_ops
#  ]

    # loss_ops = []
    # for output_op in output_ops :
    #   t_group = np.matmul(group_op, target_op.nodes)
    #   o_group = np.matmul(group_op, output_op.nodes)
    #   loss_ops.append(tf.losses.softmax_cross_entropy(t_group, o_group))

  # cof = target_op.nodes * 10.0 + 1.0
  # loss_ops = [
  #   tf.losses.mean_squared_error(tf.multiply(target_op.nodes, cof), tf.multiply(output_op.nodes, cof)) +
  #   tf.losses.mean_squared_error(target_op.groups, output_op.groups)
  #   for output_op in output_ops
  # ]

  t = 1e4
  # loss_ops = [
  #   tf.losses.mean_squared_error(target_op.nodes, output_op.nodes) +
  #   tf.losses.mean_squared_error(target_op.groups_1, output_op.groups_1) -
  #   tf.cast(tf.reduce_mean(tf.log(tf.maximum(target_op.groups_2 - output_op.groups_2, 1e-6))) / t, tf.float32)
  #   for output_op in output_ops
  # ]

  output_op = output_ops[9]
  loss_nodes = tf.losses.mean_squared_error(target_op.nodes, output_op.nodes)
#  loss_groups_1 = tf.losses.mean_squared_error(target_op.groups_1, output_op.groups_1)
#  loss_groups_2 = - tf.cast(tf.reduce_mean(tf.log(tf.maximum(target_op.groups_2 - output_op.groups_2, 1e-6))) / t, tf.float32)
#  loss_ops = (1.0 - loss_cof_ph) * loss_nodes + loss_cof_ph * loss_groups_1
  loss_ops = loss_nodes

  return loss_ops


def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


class GraphPlotter(object):

  def __init__(self, ax, graph, pos):
    self._ax = ax
    self._graph = graph
    self._pos = pos
    self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
    self._solution_length = None
    self._nodes = None
    self._edges = None
    self._start_nodes = None
    self._end_nodes = None
    self._solution_nodes = None
    self._intermediate_solution_nodes = None
    self._solution_edges = None
    self._non_solution_nodes = None
    self._non_solution_edges = None
    self._ax.set_axis_off()

  @property
  def solution_length(self):
    if self._solution_length is None:
      self._solution_length = len(self._solution_edges)
    return self._solution_length

  @property
  def nodes(self):
    if self._nodes is None:
      self._nodes = self._graph.nodes()
    return self._nodes

  @property
  def edges(self):
    if self._edges is None:
      self._edges = self._graph.edges()
    return self._edges

  @property
  def start_nodes(self):
    if self._start_nodes is None:
      self._start_nodes = [
          n for n in self.nodes if self._graph.node[n].get("start", False)
      ]
    return self._start_nodes

  @property
  def end_nodes(self):
    if self._end_nodes is None:
      self._end_nodes = [
          n for n in self.nodes if self._graph.node[n].get("end", False)
      ]
    return self._end_nodes

  @property
  def solution_nodes(self):
    if self._solution_nodes is None:
      self._solution_nodes = [
          n for n in self.nodes if self._graph.node[n].get("solution", False)
      ]
    return self._solution_nodes

  @property
  def intermediate_solution_nodes(self):
    if self._intermediate_solution_nodes is None:
      self._intermediate_solution_nodes = [
          n for n in self.nodes
          if self._graph.node[n].get("solution", False) and
          not self._graph.node[n].get("start", False) and
          not self._graph.node[n].get("end", False)
      ]
    return self._intermediate_solution_nodes

  @property
  def solution_edges(self):
    if self._solution_edges is None:
      self._solution_edges = [
          e for e in self.edges
          if self._graph.get_edge_data(e[0], e[1]).get("solution", False)
      ]
    return self._solution_edges

  @property
  def non_solution_nodes(self):
    if self._non_solution_nodes is None:
      self._non_solution_nodes = [
          n for n in self.nodes
          if not self._graph.node[n].get("solution", False)
      ]
    return self._non_solution_nodes

  @property
  def non_solution_edges(self):
    if self._non_solution_edges is None:
      self._non_solution_edges = [
          e for e in self.edges
          if not self._graph.get_edge_data(e[0], e[1]).get("solution", False)
      ]
    return self._non_solution_edges

  def _make_draw_kwargs(self, **kwargs):
    kwargs.update(self._base_draw_kwargs)
    return kwargs

  def _draw(self, draw_function, zorder=None, **kwargs):
    draw_kwargs = self._make_draw_kwargs(**kwargs)
    collection = draw_function(**draw_kwargs)
    if collection is not None and zorder is not None:
      try:
        # This is for compatibility with older matplotlib.
        collection.set_zorder(zorder)
      except AttributeError:
        # This is for compatibility with newer matplotlib.
        collection[0].set_zorder(zorder)
    return collection

  def draw_nodes(self, **kwargs):
    """Useful kwargs: nodelist, node_size, node_color, linewidths."""
    if ("node_color" in kwargs and
        isinstance(kwargs["node_color"], collections.Sequence) and
        len(kwargs["node_color"]) in {3, 4} and
        not isinstance(kwargs["node_color"][0],
                       (collections.Sequence, np.ndarray))):
      num_nodes = len(kwargs.get("nodelist", self.nodes))
      kwargs["node_color"] = np.tile(
          np.array(kwargs["node_color"])[None], [num_nodes, 1])
    return self._draw(nx.draw_networkx_nodes, **kwargs)

  def draw_edges(self, **kwargs):
    """Useful kwargs: edgelist, width."""
    return self._draw(nx.draw_networkx_edges, **kwargs)

  def draw_graph(self,
                 node_size=200,
                 node_color=(0.4, 0.8, 0.4),
                 node_linewidth=1.0,
                 edge_width=1.0):
    # Plot nodes.
    self.draw_nodes(
        nodelist=self.nodes,
        node_size=node_size,
        node_color=node_color,
        linewidths=node_linewidth,
        zorder=20)
    # Plot edges.
    self.draw_edges(edgelist=self.edges, width=edge_width, zorder=10)

  def draw_graph_with_solution(self,
                               node_size=200,
                               node_color=(0.4, 0.8, 0.4),
                               node_linewidth=1.0,
                               edge_width=1.0,
                               start_color="w",
                               end_color="k",
                               solution_node_linewidth=3.0,
                               solution_edge_width=3.0):
    node_border_color = (0.0, 0.0, 0.0, 1.0)
    node_collections = {}
    # Plot start nodes.
    node_collections["start nodes"] = self.draw_nodes(
        nodelist=self.start_nodes,
        node_size=node_size,
        node_color=start_color,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color,
        zorder=100)
    # Plot end nodes.
    node_collections["end nodes"] = self.draw_nodes(
        nodelist=self.end_nodes,
        node_size=node_size,
        node_color=end_color,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color,
        zorder=90)
    # Plot intermediate solution nodes.
    if isinstance(node_color, dict):
      c = [node_color[n] for n in self.intermediate_solution_nodes]
    else:
      c = node_color
    node_collections["intermediate solution nodes"] = self.draw_nodes(
        nodelist=self.intermediate_solution_nodes,
        node_size=node_size,
        node_color=c,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color,
        zorder=80)
    # Plot solution edges.
    node_collections["solution edges"] = self.draw_edges(
        edgelist=self.solution_edges, width=solution_edge_width, zorder=70)
    # Plot non-solution nodes.
    if isinstance(node_color, dict):
      c = [node_color[n] for n in self.non_solution_nodes]
    else:
      c = node_color
    node_collections["non-solution nodes"] = self.draw_nodes(
        nodelist=self.non_solution_nodes,
        node_size=node_size,
        node_color=c,
        linewidths=node_linewidth,
        edgecolors=node_border_color,
        zorder=20)
    # Plot non-solution edges.
    node_collections["non-solution edges"] = self.draw_edges(
        edgelist=self.non_solution_edges, width=edge_width, zorder=10)
    # Set title as solution length.
    self._ax.set_title("Solution length: {}".format(self.solution_length))
    return node_collections



#========================================================================

  # @title Set up model training and evaluation  { form-width: "30%" }

  # The model we explore includes three components:
  # - An "Encoder" graph net, which independently encodes the edge, node, and
  #   global attributes (does not compute relations etc.).
  # - A "Core" graph net, which performs N rounds of processing (message-passing)
  #   steps. The input to the Core is the concatenation of the Encoder's output
  #   and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
  #   the processing step).
  # - A "Decoder" graph net, which independently decodes the edge, node, and
  #   global attributes (does not compute relations etc.), on each
  #   message-passing step.
  #
  #                     Hidden(t)   Hidden(t+1)
  #                        |            ^
  #           *---------*  |  *------*  |  *---------*
  #           |         |  |  |      |  |  |         |
  # Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
  #           |         |---->|      |     |         |
  #           *---------*     *------*     *---------*
  #
  # The model is trained by supervised learning. Input graphs are procedurally
  # generated, and output graphs have the same structure with the nodes and edges
  # of the shortest path labeled (using 2-element 1-hot vectors). We could have
  # predicted the shortest path only by labeling either the nodes or edges, and
  # that does work, but we decided to predict both to demonstrate the flexibility
  # of graph nets' outputs.
  #
  # The training loss is computed on the output of each processing step. The
  # reason for this is to encourage the model to try to solve the problem in as
  # few steps as possible. It also helps make the output of intermediate steps
  # more interpretable.
  #
  # There's no need for a separate evaluate dataset because the inputs are
  # never repeated, so the training loss is the measure of performance on graphs
  # from the input distribution.
  #
  # We also evaluate how well the models generalize to graphs which are up to
  # twice as large as those on which it was trained. The loss is computed only
  # on the final processing step.
  #
  # Variables with the suffix _tr are training parameters, and variables with the
  # suffix _ge are test/generalization parameters.
  #
  # After around 2000-5000 training iterations the model reaches near-perfect
  # performance on graphs with between 8-16 nodes.

