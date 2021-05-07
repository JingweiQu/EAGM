import tensorflow as tf
import numpy as np
import time
import argparse

from graph_nets.demos import models

from graph_generator import GM_Core_featured as gmc
from graph_generator import CMUHouse as CMU
from graph_generator import Willow
from graph_generator import PascalVOC as VOC


def load_trained_model(model_file, dataset, num_outlier_min_max):
    tf.compat.v1.reset_default_graph()

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_ge = 10

    # Data.
    # Input and target placeholders.
    batch_size_ge = 10
    num_inner_min_max = (10, 11)

    seed = 0
    rand = np.random.RandomState(seed=seed)

    # Data.
    gmc.NODE_OUTPUT_SIZE = 1
    # Input and target placeholders.
    input_ph, target_ph, loss_cof_ph, loss_cof2_ph = gmc.create_placeholders(rand, dataset, batch_size_ge, num_inner_min_max, num_outlier_min_max, False)
    # Instantiate the model.
    model = models.EncodeProcessDecode2(node_input_size = 6, edge_output_size=1, node_output_size=gmc.NODE_OUTPUT_SIZE, group_output_size=1)

    # A list of outputs, one per processing step.
    output_ops_ge = model(input_ph, num_processing_steps_ge)

    # This cell resets the Tensorflow session, but keeps the same computational graph.
    try:
        sess.close()
    except NameError:
        pass
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Load trained model
    saver = tf.compat.v1.train.Saver(max_to_keep = 1)
    saver.restore(sess, model_file)

    return sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge


def evaluate_CMUHouse(sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge):
    mat_file = "graph_generator/cmuHouse.mat"
    num_frames, num_points, XTs = CMU._load_data_from_mat(mat_file)

    last_time = time.time()

    # Varying gap
    max_frames = 111
    for gaps in range(10, 101, 10):
        correct_gt_ge = []
        for start_frame in range(0, max_frames - gaps, 1):
            graphs = []
            graph = CMU._gen_random_graph(rand,
                                          XTs,
                                          frame_indexs=(start_frame, start_frame + gaps),
                                          num_inner_min_max=(20, 21))
            graphs.append(graph)

            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs2(graphs,
                                                                   input_ph,
                                                                   target_ph,
                                                                   loss_cof_ph,
                                                                   loss_cof2_ph)
            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)
            eval_time = time.time() - t0

            nodes = eval_values["outputs"][-1].nodes.copy()
            group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            x, count = gmc.greedy_mapping(nodes, group_indices)
            correct_gt_ge_batch, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
                eval_values["target"], eval_values["outputs"][-1], use_edges=False)
            correct_gt_ge.append(correct_gt_ge_batch)

        correct_gt_ge = np.mean(np.array(correct_gt_ge))
        print("gaps = {}: {:.4f}".format(gaps, correct_gt_ge))

    cost_time = time.time() - last_time
    print("cost_time = {:.4f}".format(cost_time))

    last_time = time.time()

    # Varying outlier
    gaps = 50
    for inners in range(20, 31, 1):
        correct_gt_ge = []
        for start_frame in range(0, max_frames - gaps, 1):
            graphs = []
            graph = CMU._gen_random_graph(rand,
                                          XTs,
                                          frame_indexs = (start_frame, start_frame + gaps),
                                          num_inner_min_max = (inners, inners + 1))
            graphs.append(graph)

            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs2(graphs,
                                                                   input_ph,
                                                                   target_ph,
                                                                   loss_cof_ph,
                                                                   loss_cof2_ph)
            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)
            eval_time = time.time() - t0

            nodes = eval_values["outputs"][-1].nodes.copy()
            group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            x, count = gmc.greedy_mapping(nodes, group_indices)
            correct_gt_ge_batch, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
                eval_values["target"], eval_values["outputs"][-1], use_edges=False)
            correct_gt_ge.append(correct_gt_ge_batch)

        correct_gt_ge = np.mean(np.array(correct_gt_ge))
        print("inners = {}: {:.4f}".format(inners, correct_gt_ge))

    cost_time = time.time() - last_time

    print("cost_time = {:.4f}".format(cost_time))


def evaluate_Willow(sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge):
    WILLOW_CATEGORIES = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

    num_categories = 5
    num_sample_per_category = 1000
    batch_size = 100
    accuracy_categories = []

    last_time = time.time()

    for i in range(num_categories):
        num_batch = int(num_sample_per_category / batch_size)
        accuracy = 0.0
        for j in range(num_batch):
            graphs, _ = Willow.gen_random_graphs_Willow(rand,
                                                        batch_size,
                                                        num_inner_min_max,
                                                        num_outlier_min_max,
                                                        use_train_set=False,
                                                        category_id=i)

            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
                                                                  input_ph,
                                                                  target_ph,
                                                                  loss_cof_ph,
                                                                  loss_cof2_ph)

            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)
            eval_time = time.time() - t0

            nodes = eval_values["outputs"][-1].nodes.copy()
            group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            x, count = gmc.greedy_mapping(nodes, group_indices)
            correct_gt_ge, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
                eval_values["target"], eval_values["outputs"][-1], use_edges=False)

            print("{}, {}, {:.4f}".format(i, j, correct_gt_ge))

            accuracy = accuracy + correct_gt_ge

        accuracy = accuracy / num_batch
        accuracy_categories.append(accuracy)

    avg_accuracy = 0.0
    for i in range(num_categories):
        avg_accuracy = avg_accuracy + accuracy_categories[i]
        print("{}: {:.4f}".format(WILLOW_CATEGORIES[i], accuracy_categories[i]))

    avg_accuracy = avg_accuracy / num_categories
    print("AVG_ACCURACY: {:.4f}".format(avg_accuracy))

    cost_time = time.time() - last_time
    print("cost_time = {:.4f}".format(cost_time))


def evaluate_VOC(sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge, num_outlier_min_max):
    VOC_CATEGORIES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                      "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                      "tvmonitor"]

    num_categories = 20
    num_sample_per_category = 1000
    batch_size = 25
    accuracy_categories = []

    last_time = time.time()

    for i in range(num_categories):
        num_batch = int(num_sample_per_category / batch_size)
        accuracy = 0.0
        for j in range(num_batch):
            graphs, _ = VOC.gen_random_graphs_VOC(rand,
                                                  batch_size,
                                                  num_inner_min_max,
                                                  num_outlier_min_max,
                                                  use_train_set = False,
                                                  category_id = i)
            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
                                                                  input_ph,
                                                                  target_ph,
                                                                  loss_cof_ph,
                                                                  loss_cof2_ph)

            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)
            eval_time = time.time() - t0

            nodes = eval_values["outputs"][-1].nodes.copy()
            group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            x, count = gmc.greedy_mapping(nodes, group_indices)
            correct_gt_ge, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
                eval_values["target"], eval_values["outputs"][-1], use_edges=False)

            print("{}, {}, {:.4f}".format(i, j, correct_gt_ge))

            accuracy = accuracy + correct_gt_ge

        accuracy = accuracy / num_batch
        accuracy_categories.append(accuracy)

    avg_accuracy = 0.0
    for i in range(num_categories):
        avg_accuracy = avg_accuracy + accuracy_categories[i]
        print("{}: {:.4f}".format(VOC_CATEGORIES[i], accuracy_categories[i]))

    avg_accuracy = avg_accuracy / num_categories
    print("AVG_ACCURACY: {:.4f}".format(avg_accuracy))

    cost_time = time.time() - last_time
    print("cost_time = {:.4f}".format(cost_time))


def parse_args():
    """
    Parse input arguments
    :return: parser
    """
    parser = argparse.ArgumentParser(description='GraphMatching Arguments')
    parser.add_argument('--dataset', dest='dataset', type=str, default='PascalVOC',
                        help='Dataset to use for training and testing: PascalVOC, Willow, or CMUHouse')
    args = parser.parse_args()
    return args


## ***********************************************************************
## Evaluation
## **********************************************************************

args = parse_args()

num_inner_min_max = (10, 11)
num_outlier_min_max = (0, 1)

seed = 0
rand = np.random.RandomState(seed=seed)

# Select dataset for evaluation
dataset = args.dataset

model_file  = "trained_models/EAGM_{:s}".format(dataset)
print('Load model: {}'.format(model_file))
sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge = load_trained_model(model_file, dataset, num_outlier_min_max)

if dataset == 'Willow':
    evaluate_fn = evaluate_Willow
    evaluate_Willow(sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge)
elif dataset == 'PascalVOC':
    outlier = 0
    num_outlier_min_max = (outlier, outlier + 1)
    evaluate_VOC(sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge, num_outlier_min_max)
elif dataset == 'CMUHouse':
    evaluate_CMUHouse(sess, input_ph, target_ph, loss_cof_ph, loss_cof2_ph, output_ops_ge)