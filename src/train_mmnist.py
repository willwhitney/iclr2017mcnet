import cv2
import sys
import time
import imageio
import os

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from mcnet import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
# from torchvision import transforms
import socket
import ipdb
from functools import partial


hostname = socket.gethostname()

# def normalize_sequence(s, K=10, dataset=None):
#     ipdb.set_trace()
#     # move color channel to the end
#     s.transpose_(1, 2).transpose_(2, 3)
#     # move sequence channel to the one away from the end
#     s.transpose_(0, 1).transpose_(1, 2)
#     s = s.cpu().numpy()
#     s = transform_from01(s)
#     diff = make_diffs(s, K)
#     # ipdb.set_trace()

#     # inverse_transform will put it back in [0, 1], which weirdly we need
#     # because the tanh saturates at -1 if we give it data in [-1, 1]
#     return inverse_transform(s), diff

def show(sequence):
    seq = sequence.transpose(2, 1).transpose(1, 0)
    seq = seq.transpose(3, 2).transpose(2, 1)
    img = image_tensor(seq)
    img = img.transpose(0, 1).transpose(1, 2)
    img = img.squeeze()
    max_size = 12
    max_input_size = max(img.size(0), img.size(1))
    figsize = (torch.Tensor((img.size(1), img.size(0))) * max_size / max_input_size).ceil()
        
    fig = plt.figure(figsize=list(figsize))
    if img.dim() == 2: 
        plt.gray()

    if img.min() < 0:
        img = inverse_transform(img.numpy())
    else:
        img = img.numpy()
        
    plt.imshow(img, interpolation='bilinear')
    plt.show()
    plt.close(fig)


# def load_data(seq_len, batch_size, image_width, channels, dataset, K=10):
#     if hostname == 'zaan':
#         data_path = '/speedy/data/mmnist'
#     else:
#         data_path = '/misc/vlgscratch4/FergusGroup/wwhitney/mmnist'

#     dataset_name = "channels{}_width{}_seqlen{}.t7".format(
#         channels, image_width, seq_len)
#     dataset_path = os.path.join(data_path, dataset_name)

#     normalizer = partial(normalize_data, dataset, channels, K)
#     train_data = MovingMNIST(train=True,
#                              seq_len=seq_len,
#                              image_size=image_width,
#                              colored=(channels == 3),
#                              transforms=[normalizer])

#     train_loader = DataLoader(train_data,
#                               num_workers=0,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               drop_last=True,
#                               pin_memory=True)
#     return train_loader

def main(name, lr, batch_size, alpha, beta, image_size, K, T, num_iter, gpu,
         nonlinearity, samples_every, gdl, channels, dataset, residual):
    margin = 0.3
    updateD = True
    updateG = True
    iters = 0
    namestr = name if len(name) == 0 else "_" + name
    prefix = (dataset.replace('/', '-')
              + namestr
              + "_image_size=" + str(image_size)
              + "_channels=" + str(channels)
              + "_K=" + str(K)
              + "_T=" + str(T)
              + "_batch_size=" + str(batch_size)
              + "_alpha=" + str(alpha)
              + "_beta=" + str(beta)
              + "_lr=" + str(lr)
              + "_nonlin=" + str(nonlinearity)
              + "_gdl=" + str(gdl))

    print("\n" + prefix + "\n")
    checkpoint_dir = "../models/" + prefix + "/"
    samples_dir = "../samples/" + prefix + "/"
    summary_dir = "../logs/" + prefix + "/"

    normalizer = partial(normalize_data, dataset, channels, K)
    train_data, test_data, num_workers = load_dataset(
        dataset, T+K, image_size, channels, transforms=[normalizer])

    train_loader = DataLoader(train_data,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    # train_loader = load_data(T+K, batch_size, image_size, channels, dataset, K)
    def get_training_batch():
        while True:
            for sequence in train_loader:
                yield sequence
    training_batch_generator = get_training_batch()


    checkpoint_dir = "../models/" + prefix + "/"
    samples_dir = "../samples/" + prefix + "/"
    summary_dir = "../logs/" + prefix + "/"

    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    if not exists(samples_dir):
        makedirs(samples_dir)
    if not exists(summary_dir):
        makedirs(summary_dir)
    
    with tf.device("/gpu:%d" % gpu[0]):
        model = MCNET(image_size=[image_size, image_size], c_dim=channels,
                      K=K, batch_size=batch_size, T=T,
                      checkpoint_dir=checkpoint_dir, nonlinearity=nonlinearity, 
                      gdl_weight=gdl, residual=residual)
        d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
            model.d_loss, var_list=model.d_vars
        )
        g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
            alpha * model.L_img + beta * model.L_GAN, var_list=model.g_vars
        )
        print("GDL: ", model.gdl_weight)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          intra_op_parallelism_threads=3,
                                          inter_op_parallelism_threads=3,
                                          gpu_options=gpu_options)) as sess:

        tf.global_variables_initializer().run()

        if model.load(sess, checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        g_sum = tf.summary.merge([model.L_p_sum,
                                  model.L_gdl_sum, model.loss_sum,
                                  model.L_GAN_sum])
        d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                                  model.d_loss_fake_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters + 1
        start_time = time.time()

        with Parallel(n_jobs=batch_size) as parallel:
            while iters < num_iter:
                for batch_index in range(100000):
                    seq_batch, diff_batch = next(training_batch_generator)
                    # show(seq_batch[0])
                    # show(diff_batch[0])
                    # ipdb.set_trace()
                    seq_batch = seq_batch.numpy()
                    diff_batch = diff_batch.numpy()

                    if updateD:
                        _, summary_str = sess.run([d_optim, d_sum],
                                                    feed_dict={model.diff_in: diff_batch,
                                                                model.xt: seq_batch[:, :, :, K - 1],
                                                                model.target: seq_batch})
                        writer.add_summary(summary_str, counter)

                    if updateG:
                        _, summary_str = sess.run([g_optim, g_sum],
                                                    feed_dict={model.diff_in: diff_batch,
                                                                model.xt: seq_batch[:, :, :, K - 1],
                                                                model.target: seq_batch})
                        writer.add_summary(summary_str, counter)

                    errD_fake = model.d_loss_fake.eval({model.diff_in: diff_batch,
                                                        model.xt: seq_batch[:, :, :, K - 1],
                                                        model.target: seq_batch})
                    errD_real = model.d_loss_real.eval({model.diff_in: diff_batch,
                                                        model.xt: seq_batch[:, :, :, K - 1],
                                                        model.target: seq_batch})
                    errG = model.L_GAN.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:, :, :, K - 1],
                                                model.target: seq_batch})
                    errImage = model.L_img.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:, :, :, K - 1],
                                                model.target: seq_batch})

                    if errD_fake < margin or errD_real < margin:
                        updateD = False
                    if errD_fake > (1. - margin) or errD_real > (1. - margin):
                        updateG = False
                    if not updateD and not updateG:
                        updateD = True
                        updateG = True

                    counter += 1

                    print(
                        "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f, L_img: %.8f"
                        % (iters, time.time() - start_time, errD_fake + errD_real, errG, errImage)
                    )

                    if (counter % samples_every) == 0:
                        samples = sess.run([model.G],
                                            feed_dict={model.diff_in: diff_batch,
                                                        model.xt: seq_batch[:, :, :, K - 1],
                                                        model.target: seq_batch})[0]
                        # ipdb.set_trace()
                        samples_pad = np.array(samples)
                        samples_pad.fill(0)
                        generations = np.concatenate((samples_pad, samples), 3)
                        generations = generations[0].swapaxes(0, 2).swapaxes(1, 2)
                        sbatch = seq_batch[0].swapaxes(
                            0, 2).swapaxes(1, 2)
                        generations = np.concatenate((generations, sbatch), axis=0)
                        print("Saving sample ...")
                        # ipdb.set_trace()
                        save_images(generations, [2, T + K],
                                    samples_dir + "train_%s.png" % (iters))
                    if np.mod(counter, 500) == 2:
                        model.save(sess, checkpoint_dir, counter)

                    iters += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr",
                        default=0.0001, help="Base Learning Rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=8, help="Mini-batch size")
    parser.add_argument("--alpha", type=float, dest="alpha",
                        default=1.0, help="Image loss weight")
    parser.add_argument("--beta", type=float, dest="beta",
                        default=0.02, help="GAN loss weight")
    parser.add_argument("--gdl", type=float,
                        default=1, help="GDL loss weight")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=128, help="Mini-batch size")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=100000, help="Number of iterations")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                        help="GPU device id")
    
    parser.add_argument("--name", default="")
    parser.add_argument("--samples_every", type=int, default=20)
    parser.add_argument("--nonlinearity", default="tanh")
    parser.add_argument("--dataset", default="mmnist")
    parser.add_argument("--channels", type=int,
                        default=3, help="how many colors the images have")
    
    parser.add_argument("--no-residual", dest="residual", action="store_false",
                        help="set the weight of residual skip connections to 0")
    parser.set_defaults(residual=True)

    args = parser.parse_args()
    main(**vars(args))
