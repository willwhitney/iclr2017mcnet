import cv2
import sys
import time
import imageio
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from mcnet import MCNET
from factored_mcnet import FactoredMCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import socket
import ipdb
from functools import partial


hostname = socket.gethostname()


def show(sequence):
    seq = sequence.transpose(2, 1).transpose(1, 0)
    seq = seq.transpose(3, 2).transpose(2, 1)
    img = image_tensor(seq)
    img = img.transpose(0, 1).transpose(1, 2)
    img = img.squeeze()
    max_size = 12
    max_input_size = max(img.size(0), img.size(1))
    figsize = (torch.Tensor((img.size(1), img.size(0)))
               * max_size / max_input_size).ceil()

    fig = plt.figure(figsize=list(figsize))
    if img.dim() == 2:
        plt.gray()

    if img.min() < 0:
        img = inverse_transform(img.numpy())
    else:
        img = img.numpy()

    plt.imshow(img[:, :, ::-1], interpolation='bilinear')
    plt.show()
    plt.close(fig)


def main(name, lr, batch_size, alpha, beta, image_size, K, T, num_iter, gpu,
         nonlinearity, samples_every, gdl, channels, dataset, residual, gamma, 
         latents, planes):
    margin = 0.3
    updateD = True
    updateG = True
    iters = 0
    namestr = name if len(name) == 0 else "_" + name
    prefix = ("FAC_" + dataset.replace('/', '-')
              + namestr
              + "_latents=" + str(latents)
              + "_planes=" + str(planes)
              + "_beta=" + str(beta)
              + "_lr=" + str(lr)
              + "_nonlin=" + str(nonlinearity)
              + "_res=" + str(residual)
              + "_gamma=" + str(gamma)
              + "_gdl=" + str(gdl))

    print("\n" + prefix + "\n")
    checkpoint_dir = "../models/" + prefix + "/"
    samples_dir = "../samples/" + prefix + "/"
    summary_dir = "../logs/" + prefix + "/"

    normalizer = partial(normalize_data, dataset, channels, K)
    train_data, test_data, num_workers = load_dataset(
        dataset, T + K, image_size, channels, transforms=[normalizer])

    train_loader = DataLoader(train_data,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

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
        model = FactoredMCNET(image_size=[image_size, image_size], c_dim=channels,
                              K=K, batch_size=batch_size, T=T,
                              checkpoint_dir=checkpoint_dir, nonlinearity=nonlinearity,
                              gdl_weight=gdl, residual=residual,
                              n_latents=latents, planes=planes)
        d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
            model.d_loss, var_list=model.d_vars
        )
        facd_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
            model.facd_loss, var_list=model.facd_vars
        )
        g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
            alpha * model.L_img + beta * model.L_GAN + gamma * model.L_FAC,
            var_list=model.g_vars
        )
        print("GDL: ", model.gdl_weight)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          intra_op_parallelism_threads=3,
                                          inter_op_parallelism_threads=3,
                                          gpu_options=gpu_options))
    
    with sess.as_default():
        tf.global_variables_initializer().run()

        if model.load(sess, checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        g_sum = tf.summary.merge([model.L_p_sum,
                                    model.L_gdl_sum, model.loss_sum,
                                    model.L_GAN_sum, model.L_FAC_sum])
        d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                                    model.d_loss_fake_sum])
        facd_sum = tf.summary.merge([model.facd_loss_real_sum, model.facd_loss_sum,
                                    model.facd_loss_fake_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters + 1
        start_time = time.time()

        while iters < num_iter:
            for batch_index in range(100000):
                seq_batch, diff_batch = next(training_batch_generator)
                # show(seq_batch[0])
                # show(diff_batch[0])
                # ipdb.set_trace()
                seq_batch = seq_batch.numpy()
                diff_batch = diff_batch.numpy()
                step_input = {model.diff_in: diff_batch,
                                model.xt: seq_batch[:, :, :, K - 1],
                                model.target: seq_batch}

                if updateD:
                    _, summary_str = sess.run([d_optim, d_sum],
                                                feed_dict=step_input)
                    writer.add_summary(summary_str, counter)

                if updateG:
                    _, summary_str, _, facd_summary_str = sess.run(
                        [g_optim, g_sum, facd_optim, facd_sum],
                        feed_dict=step_input)

                    writer.add_summary(summary_str, counter)
                    writer.add_summary(facd_summary_str, counter)
                    # _, summary_str = sess.run([facd_optim, facd_sum],
                    #                             feed_dict={model.diff_in: diff_batch,
                    #                                         model.xt: seq_batch[:, :, :, K - 1],
                    #                                         model.target: seq_batch})
                    # writer.add_summary(summary_str, counter)

                things_to_run = [
                    model.d_loss_fake,
                    model.d_loss_real,
                    model.facd_loss_fake,
                    model.facd_loss_real,
                    model.FacD,
                    model.FacD_,
                    model.L_GAN,
                    model.L_img,
                ]
                (errD_fake, errD_real, err_facD_fake, err_facD_real,
                    facD_real_outputs, facD_fake_outputs, errG, errImage) = sess.run(
                        things_to_run, feed_dict=step_input)
                # errD_fake = model.d_loss_fake.eval({model.diff_in: diff_batch,
                #                                     model.xt: seq_batch[:, :, :, K - 1],
                #                                     model.target: seq_batch})
                # errD_real = model.d_loss_real.eval({model.diff_in: diff_batch,
                #                                     model.xt: seq_batch[:, :, :, K - 1],
                #                                     model.target: seq_batch})
                # err_facD_fake = model.facd_loss_fake.eval({model.diff_in: diff_batch,
                #                                     model.xt: seq_batch[:, :, :, K - 1],
                #                                     model.target: seq_batch})
                # err_facD_real = model.facd_loss_real.eval({model.diff_in: diff_batch,
                #                                     model.xt: seq_batch[:, :, :, K - 1],
                #                                     model.target: seq_batch})
                # facD_real_outputs = model.FacD.eval({model.diff_in: diff_batch,
                #                                     model.xt: seq_batch[:, :, :, K - 1],
                #                                     model.target: seq_batch})
                # facD_fake_outputs = model.FacD_.eval({model.diff_in: diff_batch,
                #                                     model.xt: seq_batch[:, :, :, K - 1],
                #                                     model.target: seq_batch})
                # errG = model.L_GAN.eval({model.diff_in: diff_batch,
                #                             model.xt: seq_batch[:, :, :, K - 1],
                #                             model.target: seq_batch})
                # errImage = model.L_img.eval({model.diff_in: diff_batch,
                #                                 model.xt: seq_batch[:, :, :, K - 1],
                #                                 model.target: seq_batch})

                if errD_fake < margin or errD_real < margin:
                    updateD = False
                if errD_fake > (1. - margin) or errD_real > (1. - margin):
                    updateG = False
                if not updateD and not updateG:
                    updateD = True
                    updateG = True

                counter += 1

                print(("Iters: [%2d] time: %4.4f, d_loss: %3.4f, "
                        "L_GAN: %3.4f, L_img: %3.4f, "
                        "facd_loss: %3.4f, facd_real_output: %3.4f, facd_fake_output: %3.4f")
                        % (iters, time.time() - start_time, errD_fake + errD_real,
                            errG, errImage, err_facD_fake + err_facD_real,
                            facD_real_outputs.mean(), facD_fake_outputs.mean())
                        )

                if (counter % samples_every) == 0:
                    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                    samples, frozengens, diffs, crosses, crossgens = sess.run(
                        [model.G, model.frozengens, model.diffs, model.crosses, model.crossgens],
                        feed_dict={model.diff_in: diff_batch,
                                    model.xt: seq_batch[:, :, :, K - 1],
                                    model.target: seq_batch})
                    # ipdb.set_trace()

                    frozen_expanded = frozengens.reshape(batch_size, 128, 128, T, 3, 3)
                    print("Mean difference between pixels in frozen and gen: ",
                            (frozen_expanded[:, :, :, :, 2] - samples).mean())
                    # ipdb.set_trace()
                    print("Mean difference between pixels in crossover and gen: ",
                            (np.concatenate(crossgens[1], 3) - samples).mean())
                    print("Saving sample ...")

                    for l in range(1, latents+1):
                        # bsize x imwidth x imheight x T x channels
                        crossover_batch = np.concatenate(crossgens[l], 3)

                        # bsize x T x imwidth x imheight x channels
                        crossover_batch = crossover_batch.swapaxes(3, 1).swapaxes(2, 3)

                        # squash for the image saving function
                        # (bsize*T) x imwidth x imheight x channels
                        crossover_batch = crossover_batch.reshape(batch_size * T, 128, 128, 3)

                        # each row should be a sequence
                        # each column should be different sequences at the same timestep, all
                        # generated using the same z_l
                        save_images(crossover_batch[:, :, :, ::-1], [batch_size, T],
                                    samples_dir + "cross_%s_%s.png" % (iters, l))

                    for s in range(4):
                        # batchsize x img_width x img_height x T x n_latents + 1 x channels
                        save_images(frozen_expanded[s].swapaxes(2, 3)
                                    .reshape(128, 128, 30, 3).swapaxes(0, 2)
                                    .swapaxes(1, 2)[:, :, :, ::-1], [3, 10], 
                                    samples_dir + "frozen_%s_%s.png" % (iters, s))
                        # ipdb.set_trace()


                        # f_reshaped = frozengens.reshape(8, 128, 128, 10, 3, 3)
                        samples_pad = np.array(samples)
                        samples_pad.fill(0)
                        generations = np.concatenate((samples_pad, samples), 3)

                        # gives 10x128x128x3
                        gen = generations[s].swapaxes(
                            0, 2).swapaxes(1, 2)
                        sbatch = seq_batch[s].swapaxes(
                            0, 2).swapaxes(1, 2)

                        # gives 20x128x128x3
                        gen = np.concatenate(
                            (gen, sbatch), axis=0)
                        # ipdb.set_trace()
                        save_images(gen[:, :, :, ::-1], [2, T + K],
                                    samples_dir + "train_%s_%s.png" % (iters, s))
                if np.mod(counter, 500) == 2:
                    model.save(sess, checkpoint_dir, counter)

                iters += 1
    sess.close()


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

    parser.add_argument("--planes", type=int,
                        default=256, help="the number of planes in h_dyn")
    parser.add_argument("--latents", type=int,
                        default=2, help="the number of latent factors")
    parser.add_argument("--gamma", type=float,
                        default=0.1, help="factor GAN loss weight")

    parser.add_argument("--no-residual", dest="residual", action="store_false",
                        help="set the weight of residual skip connections to 0")
    parser.set_defaults(residual=True)

    args = parser.parse_args()
    main(**vars(args))
