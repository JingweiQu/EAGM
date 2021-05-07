from graph_nets.demos import models

import time
import os
import argparse
import tensorflow as tf
import numpy as np
from graph_generator import GM_Core_featured as gmc


def parse_args():
    """
    Parse input arguments
    :return: parser
    """
    parser = argparse.ArgumentParser(description='GraphMatching Arguments')
    parser.add_argument('--dataset', dest='dataset', type=str, default='PascalVOC',
                        help='Dataset to use for training and testing: PascalVOC, Willow, or CMUHouse')
    parser.add_argument('--num_processing_steps_tr', dest='num_processing_steps_tr', type=int, default=10,
                        help='Number of processing (message-passing) steps')
    parser.add_argument('--batch_size_tr', dest='batch_size_tr', type=int, default=32, help='Training batch size')
    args = parser.parse_args()
    return args


args = parse_args()
# Select dataset for training
dataset = args.dataset

tf.compat.v1.reset_default_graph()

seed = 0
rand = np.random.RandomState(seed=seed)

# Model parameters.
# Number of processing (message-passing) steps.
num_processing_steps_tr = 10
num_processing_steps_ge = 10

# Data / training parameters.
batch_size_tr = 32
batch_size_ge = 100
# Number of nodes per graph sampled uniformly from this range.
num_inner_min_max = (10, 11)
num_outlier_min_max = (0, 11)
decay_step = 2000
if dataset == 'Willow':
    num_training_samples = 400 * 5 * 600
elif dataset == 'PascalVOC':
    num_training_samples = 100000 * 20
elif dataset == 'CMUHouse':
    batch_size_tr = 16
    batch_size_ge = 25
    num_training_samples = 300000
    num_inner_min_max = (10, 31)
    decay_step = int(32 / batch_size_tr * decay_step)
num_training_iterations = int(num_training_samples / batch_size_tr)
eval_step = int(1600 / batch_size_tr)


# Data.
gmc.NODE_OUTPUT_SIZE = 1
# gmc.NODE_OUTPUT_SIZE = 2

# Input and target placeholders.
input_ph, target_ph, loss_cof_ph, loss_cof2_ph = gmc.create_placeholders(rand, dataset, batch_size_tr, num_inner_min_max, num_outlier_min_max)

# Connect the data to the model.
# Instantiate the model.
model = models.EncodeProcessDecode2(node_input_size = 6, edge_output_size=1, node_output_size=gmc.NODE_OUTPUT_SIZE, group_output_size=1)

# A list of outputs, one per processing step.
output_ops_tr = model(input_ph, num_processing_steps_tr)
output_ops_ge = model(input_ph, num_processing_steps_ge)

# Training loss.
loss_op_tr = gmc.create_loss_ops(target_ph, output_ops_tr, loss_cof_ph, loss_cof2_ph)

# Test/generalization loss.
loss_op_ge = gmc.create_loss_ops(target_ph, output_ops_ge, loss_cof_ph, loss_cof2_ph)

# Optimizer.
# Learning_rate = 1e-3
global_step = tf.Variable(0, trainable = False)
learning_rate = tf.math.maximum(0.0001, tf.compat.v1.train.exponential_decay(0.001, global_step, decay_step, 0.98, staircase=False))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr, global_step)

# Lets an iterable of TF graphs be output from a session as NP graphs.
input_ph, target_ph = gmc.make_all_runnable_in_session(input_ph, target_ph)


#======================================================================================
# @title Reset session  { form-width: "30%" }

# This cell resets the Tensorflow session, but keeps the same computational graph.

try:
    sess.close()
except NameError:
    pass

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []


#============================================
# @title Run training  { form-width: "30%" }

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.

print("# (iteration number), FT (elapsed feed_dict seconds), TT (elapsed training second),"
       "Ltr (training loss), Lge (test/generalization loss), "
       "C_All (test nodes (for all) labeled correctly), "
       "C_GT (test nodes (for groundtruth) labeled correctly), "
       "Sge (test/generalization fraction examples solved correctly)")

# Saver for model
saver = tf.compat.v1.train.Saver(max_to_keep=1)
min_loss = 1e6

start_time = time.time()
last_log_time = start_time
feed_dict_time = 0.0
training_time = 0.0
eval_time = 0.0

if dataset != 'CMUHouse':
    create_feed_dict = gmc.create_feed_dict
else:
    create_feed_dict = gmc.create_feed_dict2

for iteration in range(last_iteration, num_training_iterations):
    last_iteration = iteration
    last_time = time.time()
    feed_dict, _ = create_feed_dict(rand, dataset, batch_size_tr, num_inner_min_max, num_outlier_min_max,
                                    input_ph, target_ph, loss_cof_ph, loss_cof2_ph)
    feed_dict_time = feed_dict_time + time.time() - last_time

    last_time = time.time()

    train_values = sess.run({
        "step": step_op,
        "target": target_ph,
        "loss": loss_op_tr,
        "outputs": output_ops_tr,
        "learning_rate": learning_rate},
         feed_dict=feed_dict)
    output_nodes = train_values["outputs"][-1].nodes
    training_time = training_time + time.time() - last_time

    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time

    if iteration % eval_step == 0:
        last_time = the_time

        correct_gt_ge = []
        correct_all_ge = []
        solved_ge = []
        matches_ge = 0
        test_loss = []
        batch_num = int(100 / batch_size_ge)
        for eval_iter in range(batch_num):
            feed_dict, raw_graphs = create_feed_dict(rand, dataset, batch_size_ge, num_inner_min_max,
                                                     num_outlier_min_max, input_ph, target_ph,
                                                     loss_cof_ph, loss_cof2_ph)
            test_values = sess.run({
                "target": target_ph,
                "loss": loss_op_ge,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)

            correct_gt_ge_batch, correct_all_ge_batch, solved_ge_batch, matches_ge_batch = gmc.compute_accuracy(
                test_values["target"], test_values["outputs"][-1], use_edges=False)
            correct_gt_ge.append(correct_gt_ge_batch)
            correct_all_ge.append(correct_all_ge_batch)
            solved_ge.append(solved_ge_batch)
            matches_ge += matches_ge_batch
            test_loss.append(test_values["loss"])
        correct_gt_ge = np.mean(np.array(correct_gt_ge))
        correct_all_ge = np.mean(np.array(correct_all_ge))
        solved_ge = np.mean(np.array(solved_ge))
        test_loss = np.mean(np.array(test_loss))
        elapsed = time.time() - start_time
        losses_tr.append(train_values["loss"])
        losses_ge.append(test_loss)
        corrects_ge.append(correct_all_ge)
        solveds_ge.append(solved_ge)
        logged_iterations.append(iteration)

        if test_loss < min_loss:
            file_path = "save_models/LGM_{:s}".format(dataset)
            saver.save(sess, file_path, global_step=iteration)
            min_loss = test_loss

        eval_time = eval_time + time.time() - last_time

        print("# {:05d}, T {:.1f}, FT {:.1f}, TT {:.1f}, ET {:.1f}, Ltr {:.4f}, Lge {:.4f}, "
                " CAge {:.4f}, CGge {:.4f}, NEG {:d}, LR {:.5f}".format(
              iteration, elapsed, feed_dict_time, training_time, eval_time, train_values["loss"],
              test_loss, correct_all_ge, correct_gt_ge, matches_ge, train_values["learning_rate"]))