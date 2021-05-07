from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from graph_nets import utils_featured_graph
from graph_nets import utils_tf
import numpy as np
import tensorflow as tf

# from graph_generator import CMUHouse
# from graph_generator import Willow
# from graph_generator import PascalVOC as VOC

install_graph_nets_library = "No"  #@param ["Yes", "No"]

NODE_OUTPUT_SIZE = 1

if install_graph_nets_library.lower() == "yes":
  print("Installing Graph Nets library with:")
  print("  $ pip install graph_nets\n")
  print("Output message from command:\n")
#  !pip install graph_nets
else:
  print("Skipping installation of Graph Nets library")


SEED = 1
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)


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


def featured_graph_to_input_target(graph):
  gidx1 = np.int32(graph["gidx1"])
  gidx2 = np.int32(graph["gidx2"])
  gidx1_t = np.reshape(gidx1, (len(gidx1), 1), order = 'C')
  gidx2_t = np.reshape(gidx2, (len(gidx2), 1), order = 'C')

  attr_dicts = dict(gidx1 = gidx1, gidx2 = gidx2)
  input_attr_dicts =attr_dicts
  target_attr_dicts = attr_dicts

  node_features = np.hstack((graph["node_features"], gidx1_t, gidx2_t))
  node_solution = graph["solutions"]

  input_attr_dicts["features"] = np.array([0.0])
  target_attr_dicts["features"] = np.array([0.0])

  # Generate groups info for target_graph
  n_group1 = np.int(np.max(gidx1) + 1)
  n_group2 = np.int(np.max(gidx2) + 1)

  groups1 = np.zeros((n_group1,1), dtype=np.float)
  groups2 = np.zeros((n_group2,1), dtype=np.float)
  for i in range(len(node_solution)):
      if node_solution[i] :
        groups1[gidx1[i]] = groups1[gidx1[i]] + node_solution[i]
        groups2[gidx2[i]] = groups2[gidx2[i]] + node_solution[i]

  input_attr_dicts["gidx1"] = gidx1
  input_attr_dicts["gidx2"] = gidx2
  input_attr_dicts["groups1"] = np.zeros((n_group1,1), dtype=np.float)
  input_attr_dicts["groups2"] = np.zeros((n_group2,1), dtype=np.float)
  target_attr_dicts["gidx1"] = gidx1
  target_attr_dicts["gidx2"] = gidx2
  target_attr_dicts["groups1"] = groups1
  target_attr_dicts["groups2"] = groups2

  if NODE_OUTPUT_SIZE == 1:
      target_node_features = np.zeros((len(node_solution), 1), dtype = np.float)
      for i in range(len(node_solution)) :
         target_node_features[i][0] = node_solution[i]
      edge_solution = graph["edge_solution"]
  else:
      target_node_features = []
      for i in range(len(node_solution)) :
          feature = to_one_hot(node_solution[i].astype(int), NODE_OUTPUT_SIZE)
          target_node_features.append(feature)
      target_node_features = np.array(target_node_features)

      edge_solution = []
      for i in range(graph["edge_solution"].shape[0]):
          solution = to_one_hot(graph["edge_solution"][i, 0].astype(int), NODE_OUTPUT_SIZE)
          edge_solution.append(solution)
      edge_solution = np.array(edge_solution)

  edge_features = {"senders": graph["senders"],
                   "receivers": graph["receivers"],
                   "features": graph["edge_features"],
                   "features2": graph["edge_features2"]}

  target_edge_features = {"senders": graph["senders"],
                          "receivers": graph["receivers"],
                          "features": edge_solution,
                          "features2": edge_solution}

  input_graph = utils_featured_graph.FeaturedGraph(node_features, graph["node_features2"], edge_features, input_attr_dicts)
  target_graph = utils_featured_graph.FeaturedGraph(target_node_features, target_node_features, target_edge_features, target_attr_dicts)

  return input_graph, target_graph


def generate_featured_graphs(rand,
                             dataset,
                             batch_size,
                             num_inner_min_max,
                             num_outlier_min_max,
                             use_train_set=True) :
    if dataset == "CMUHouse":
        from graph_generator import CMUHouse
        graphs = CMUHouse.gen_random_graphs_CMU(rand,
                                                batch_size,
                                                num_inner_min_max,
                                                num_outlier_min_max)
    elif dataset == "Willow":
        from graph_generator import Willow
        graphs, _ = Willow.gen_random_graphs_Willow(rand,
                                                    batch_size,
                                                    num_inner_min_max,
                                                    num_outlier_min_max,
                                                    use_train_set=use_train_set)
    elif dataset == "PascalVOC":
        from graph_generator import PascalVOC as VOC
        graphs, _ = VOC.gen_random_graphs_VOC(rand,
                                              batch_size,
                                              num_inner_min_max,
                                              num_outlier_min_max,
                                              use_train_set=use_train_set)

    input_graphs = []
    target_graphs = []
    for i in range(batch_size) :
        input, target = featured_graph_to_input_target(graphs[i])
        input_graphs.append(input)
        target_graphs.append(target)

    return input_graphs, target_graphs, graphs


def create_placeholders(rand,
                        dataset,
                        batch_size,
                        num_inner_min_max,
                        num_outlier_min_max,
                        use_train_set=True):
  # Create some example data for inspecting the vector sizes.
  input_graphs, target_graphs, _ = generate_featured_graphs(rand,
                                                            dataset,
                                                            batch_size,
                                                            num_inner_min_max,
                                                            num_outlier_min_max,
                                                            use_train_set=use_train_set)
  input_ph = utils_tf.placeholders_from_weighted_graphs(input_graphs)
  target_ph = utils_tf.placeholders_from_weighted_graphs(target_graphs)
  loss_cof_ph = tf.compat.v1.placeholder(dtype = target_ph.nodes.dtype, shape = target_ph.nodes.shape)
  loss_cof2_ph = tf.compat.v1.placeholder(dtype = target_ph.edges.dtype, shape = target_ph.edges.shape)
  return input_ph, target_ph, loss_cof_ph, loss_cof2_ph


def create_feed_dict_by_graphs(graphs,
                               input_ph,
                               target_ph,
                               loss_cof_ph,
                               loss_cof2_ph):

    inputs = []
    targets = []
    for i in range(len(graphs)):
        input, target = featured_graph_to_input_target(graphs[i])
        inputs.append(input)
        targets.append(target)

    input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
    target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

    if NODE_OUTPUT_SIZE == 1:
        loss_cof = target_graphs.nodes * 5.0 + 1.0
        loss_cof2 = target_graphs.edges * 11.0 + 1.0
    else:
        loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
        loss_cof[:][1] = 5.0

        loss_cof2 = np.ones(shape=target_graphs.edges.shape, dtype=target_graphs.edges.dtype)
        loss_cof2[:][1] = 11.0

    feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof, loss_cof2_ph: loss_cof2}

    return feed_dict, graphs


def create_feed_dict_by_graphs2(graphs,
                                input_ph,
                                target_ph,
                                loss_cof_ph,
                                loss_cof2_ph):

    inputs = []
    targets = []
    for i in range(len(graphs)):
        input, target = featured_graph_to_input_target(graphs[i])
        inputs.append(input)
        targets.append(target)

    input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
    target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

    if NODE_OUTPUT_SIZE == 1:
        loss_cof = target_graphs.nodes * 11.0 + 1.0
        loss_cof2 = target_graphs.edges * 35.0 + 1.0
    else:
        loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
        loss_cof[:][1] = 11.0

        loss_cof2 = np.ones(shape=target_graphs.edges.shape, dtype=target_graphs.edges.dtype)
        loss_cof2[:][1] = 35.0

    feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof, loss_cof2_ph: loss_cof2}

    return feed_dict, graphs


def create_feed_dict(rand,
                     dataset,
                     batch_size,
                     num_inner_min_max,
                     num_outlier_min_max,
                     input_ph,
                     target_ph,
                     loss_cof_ph,
                     loss_cof2_ph):
  inputs, targets, raw_graphs = generate_featured_graphs(
      rand, dataset, batch_size, num_inner_min_max, num_outlier_min_max)
  input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
  target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

  if NODE_OUTPUT_SIZE == 1:
    loss_cof = target_graphs.nodes * 5.0  + 1.0
    loss_cof2 = target_graphs.edges * 11.0  + 1.0
  else:
    loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
    loss_cof[:][1] = 5.0

    loss_cof2 = np.ones(shape=target_graphs.edges.shape, dtype=target_graphs.edges.dtype)
    loss_cof2[:][1] = 11.0

  feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof, loss_cof2_ph: loss_cof2}

  return feed_dict, raw_graphs


def create_feed_dict2(rand,
                      dataset,
                      batch_size,
                      num_inner_min_max,
                      num_outlier_min_max,
                      input_ph,
                      target_ph,
                      loss_cof_ph,
                      loss_cof2_ph):
  inputs, targets, raw_graphs = generate_featured_graphs(
      rand, dataset, batch_size, num_inner_min_max, num_outlier_min_max)
  input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
  target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

  if NODE_OUTPUT_SIZE == 1:
    loss_cof = target_graphs.nodes * 11.0  + 1.0
    loss_cof2 = target_graphs.edges * 35.0  + 1.0
  else:
    loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
    loss_cof[:][1] = 11.0

    loss_cof2 = np.ones(shape=target_graphs.edges.shape, dtype=target_graphs.edges.dtype)
    loss_cof2[:][1] = 35.0

  feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof, loss_cof2_ph: loss_cof2}

  return feed_dict, raw_graphs


def greedy_mapping(nodes, group_indices):
    x = np.zeros(shape = nodes.shape, dtype = np.int)
    count = 0

    while True:
        idx = np.argmax(nodes)
        if nodes[idx] <= 0.0 :
            break

        nodes[idx] = 0.0
        x[idx] = 1
        count = count + 1

        gidx = group_indices[idx]

        for i in range(len(nodes)):
            if group_indices[i] == gidx :
                nodes[i] = 0.0

    return x, count

def compute_accuracy(target, output, use_nodes=True, use_edges=False):
  if not use_nodes and not use_edges:
    raise ValueError("Nodes or edges (or both) must be used")

  tdds = utils_featured_graph.graphs_tuple_to_data_dicts(target)
  odds = utils_featured_graph.graphs_tuple_to_data_dicts(output)

  cs_all = []
  ss_all = []
  cs_gt = []
  num_matches = 0

  if NODE_OUTPUT_SIZE == 1:
      for td, od in zip(tdds, odds):
        xn = td["nodes"].astype(np.int)
        yn, num = greedy_mapping(od["nodes"], od["group_indices_1"])
        num_matches = num_matches + num

        c_all = (xn == yn)
        s_all = np.all(c_all)
        cs_all.append(c_all)
        ss_all.append(s_all)

        c_gt = 0
        for i in range(len(xn)):
            if xn[i] == 1 and xn[i] == yn[i] :
                c_gt = c_gt + 1
        if np.sum(xn) > 0:
            c_gt = c_gt / np.sum(xn)
        else:
            c_gt = 0
        cs_gt.append(c_gt)
  else:
      for td, od in zip(tdds, odds):
        tlabels = np.argmax(td["nodes"], axis = 1)
        olabels = np.argmax(od["nodes"], axis = 1)
        olabels, _ = greedy_mapping(olabels, od["group_indices_1"])
        num_matches = num_matches + np.sum(olabels)

        c_all = (tlabels == olabels)
        s_all = np.all(c_all)
        cs_all.append(c_all)
        ss_all.append(s_all)

        c_gt = tlabels.dot(c_all)
        c_gt = np.sum(c_gt) / np.sum(tlabels)
        cs_gt.append(c_gt)

  correct_gt = np.mean(np.array(cs_gt))
  correct_all = np.mean(np.concatenate(cs_all, axis=0))
  solved = np.mean(np.stack(ss_all))
  return correct_gt, correct_all, solved, num_matches


def create_loss_ops(target_op, output_ops, loss_cof, loss_cof2):
    output_op = output_ops[-1]
    if NODE_OUTPUT_SIZE == 1:
        loss_nodes = tf.compat.v1.losses.mean_squared_error(loss_cof * target_op.nodes, loss_cof * output_op.nodes)
        loss_groups_1 = tf.compat.v1.losses.mean_squared_error(target_op.groups_1, output_op.groups_1)
        loss_edges = tf.compat.v1.losses.mean_squared_error(loss_cof2 * target_op.edges, loss_cof2 * output_op.edges)

        loss_ops = loss_nodes  + 0.1 * loss_groups_1 + 0.1 * loss_edges
    else:
        loss_nodes = tf.losses.softmax_cross_entropy(loss_cof * target_op.nodes, loss_cof * output_op.nodes)
        loss_ops = loss_nodes

    return loss_ops


def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]