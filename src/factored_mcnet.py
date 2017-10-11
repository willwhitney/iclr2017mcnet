import os
import tensorflow as tf

from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *

import ipdb


class FactoredMCNET(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=10, T=10, checkpoint_dir=None, is_train=True, nonlinearity="tanh",
                 gdl_weight=1.0, residual=True, n_latents=2, planes=256):

        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.nonlinearity = nonlinearity
        self.gdl_weight = gdl_weight
        self.n_latents = n_latents
        self.convlstm_output_planes = planes
        self.latent_dim = self.convlstm_output_planes // n_latents
        self.res_weight = 1.0 if residual else 0.0

        self.gf_dim = 64
        self.df_dim = 64

        self.c_dim = c_dim
        self.K = K
        self.T = T
        self.diff_shape = [batch_size, self.image_size[0],
                           self.image_size[1], K - 1, 1]
        self.xt_shape = [batch_size, self.image_size[0],
                         self.image_size[1], c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             K + T, c_dim]

        self.build_model()
    
    def batch_select_one(self, latent, i):
        start_loc = i * self.latent_dim
        end_loc = (i + 1) * self.latent_dim
        return latent[:, :, :, start_loc: end_loc]
    
    def batch_select(self, latent, start=None, end=None):
        start_loc = None if start is None else start * self.latent_dim
        end_loc = None if end is None else (end + 1) * self.latent_dim
        # ipdb.set_trace()
        return latent[:, :, :, start_loc: end_loc]

    def batch_swap(self, a, b, i):
        before = self.batch_select(a, end=i-1)
        after = self.batch_select(a, start=i+1)
        swapped = self.batch_select_one(b, i)
        # ipdb.set_trace()

        return tf.concat([before, swapped, after], 3)

    

    def build_model(self):
        self.diff_in = tf.placeholder(
            tf.float32, self.diff_shape, name='diff_in')
        self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
        self.target = tf.placeholder(
            tf.float32, self.target_shape, name='target')

        cell = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                 [3, 3], self.convlstm_output_planes)
        pred, latents, frozengens = self.forward(self.diff_in, self.xt, cell)
        self.frozengens = tf.concat(frozengens, 3)

        # latents_Einput = tf.concat(latents, 0)
        # latents_Etarget = tf.ones(latents_Einput.shape) / 2

        # true_latents, false_latents = [], []
        # for latent in latents:
        #     true_latents.append(latents[:self.batch_size//2])
        #     z1 = latents[self.batch_size//2:, :latent.shape[1]//2]
        #     z2 = latents[self.batch_size//2:, latent.shape[1]//2:]
        #     z2 = tf.random_shuffle(z2)
        #     false_latent = tf.concat([z1, z2], 1)
        #     false_latents.append(false_latent)
        # true_latents = tf.concat(true_latents, 0)
        # false_latents = tf.concat(false_latents, 0)

        true_latents = tf.concat(latents, 0)
        latents = tf.stack(latents)
        false_latents = []
        for timestep in range(latents.shape[0]):
            latent = latents[timestep]
            fac_dim = 3

            # for each latent vector in the batch, substitute in one
            # independent factor from the next latent vector

            for i in range(self.batch_size):
                # swapped_index = tf.random_uniform(
                #     [1], minval=0, maxval=self.n_latents, dtype=tf.int32)

                # I wanted to do this by picking a random one, but that's
                # horrifyingly hard in tensorflow
                swapped_index = (timestep * self.T + i) % self.n_latents

                other = (i+1) % self.batch_size
                a = tf.expand_dims(latent[i], 0)
                b = tf.expand_dims(latent[other], 0)
                
                false_latent = self.batch_swap(a, b, swapped_index)
                false_latents.append(false_latent)


            # shuffle the z1s and z2s within each timestep to get false examples
            # z1 = latent[:, :, :, :latent.shape[fac_dim]//2]
            # z2 = latent[:, :, :, latent.shape[fac_dim]//2:]

            # shuffle which batch elements are in which locations
            # z2 = tf.random_shuffle(z2)
            # false_latent = tf.concat([z1, z2], fac_dim)
            # false_latents.append(false_latent)

        false_latents = tf.concat(false_latents, 0)
        # latents_Dinput = tf.concat((true_latents, false_latents), 0)
        # latents_Dtarget = tf.concat((tf.ones(true_latents.shape),
        #                             tf.zeros(false_latents.shape)), 0)

        self.G = tf.concat(axis=3, values=pred)
        if self.is_train:
            true_sim = inverse_transform(self.target[:, :, :, self.K:, :])
            if self.c_dim == 1:
                true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])
            true_sim = tf.reshape(tf.transpose(true_sim, [0, 3, 1, 2, 4]),
                                  [-1, self.image_size[0],
                                   self.image_size[1], 3])
            gen_sim = inverse_transform(self.G)
            if self.c_dim == 1:
                gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
            gen_sim = tf.reshape(tf.transpose(gen_sim, [0, 3, 1, 2, 4]),
                                 [-1, self.image_size[0],
                                  self.image_size[1], 3])
            binput = tf.reshape(self.target[:, :, :, :self.K, :],
                                [self.batch_size, self.image_size[0],
                                 self.image_size[1], -1])
            btarget = tf.reshape(self.target[:, :, :, self.K:, :],
                                 [self.batch_size, self.image_size[0],
                                  self.image_size[1], -1])
            bgen = tf.reshape(self.G, [self.batch_size,
                                       self.image_size[0],
                                       self.image_size[1], -1])

            good_data = tf.concat(axis=3, values=[binput, btarget])
            gen_data = tf.concat(axis=3, values=[binput, bgen])

            with tf.variable_scope("DIS", reuse=False):
                self.D, self.D_logits = self.discriminator(good_data)

            with tf.variable_scope("DIS", reuse=True):
                self.D_, self.D_logits_ = self.discriminator(gen_data)

            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits, labels=tf.ones_like(self.D)
                )
            )
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.zeros_like(self.D_)
                )
            )

            with tf.variable_scope("FAC", reuse=False):
                self.FacD, self.FacD_logits = self.factor_discriminator(true_latents)

            with tf.variable_scope("FAC", reuse=True):
                self.FacD_, self.FacD_logits_ = self.factor_discriminator(false_latents)

            self.facd_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.FacD_logits, labels=tf.ones_like(self.FacD)
                )
            )
            self.facd_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.FacD_logits_, labels=tf.zeros_like(self.FacD_)
                )
            )

            self.L_p = tf.reduce_mean(
                tf.square(self.G - self.target[:, :, :, self.K:, :])
            )
            self.L_gdl = gdl(gen_sim, true_sim, 1.)
            self.L_img = self.L_p + self.gdl_weight * self.L_gdl

            self.facd_loss = self.facd_loss_real + self.facd_loss_fake

            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.L_GAN = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.ones_like(self.D_)
                )
            )

            self.L_FAC = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.FacD_logits, labels=(tf.ones_like(self.D_)/2)
                )
            )

            self.loss_sum = tf.summary.scalar("L_img", self.L_img)
            self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
            self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
            self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
            self.L_FAC_sum = tf.summary.scalar("L_FAC", self.L_FAC)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.facd_loss_sum = tf.summary.scalar("d_loss", self.facd_loss)
            self.d_loss_real_sum = tf.summary.scalar(
                "d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar(
                "d_loss_fake", self.d_loss_fake)
            self.facd_loss_real_sum = tf.summary.scalar(
                "facd_loss_real", self.facd_loss_real)
            self.facd_loss_fake_sum = tf.summary.scalar(
                "facd_loss_fake", self.facd_loss_fake)

            self.t_vars = tf.trainable_variables()
            self.g_vars = [var for var in self.t_vars
                           if 'DIS' not in var.name and 'FAC' not in var.name]
            self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
            self.facd_vars = [var for var in self.t_vars if 'FAC' in var.name]
            num_param = 0.0
            for var in self.g_vars:
                num_param += int(np.prod(var.get_shape()))
            print("Number of parameters: %d" % num_param)

            # content_inputs = 
            # self.analogies = 


        self.saver = tf.train.Saver(max_to_keep=10)

    def reshape_image(self, tensor):
        return tf.reshape(tensor, [self.batch_size, self.image_size[0],
                                   self.image_size[1], 1, self.c_dim])

    def forward(self, diff_in, xt, cell):
        # Initial state
        state = tf.zeros([self.batch_size, self.image_size[0] / 8,
                          self.image_size[1] / 8, 2 * self.convlstm_output_planes])
        reuse = False
        # Encoder
        for t in range(self.K - 1):
            enc_h, res_m = self.motion_enc(diff_in[:, :, :, t, :], reuse=reuse)
            h_dyn, state = cell(enc_h, state, scope='lstm', reuse=reuse)
            reuse = True

        pred = []
        latents = []
        frozengens = []

        # Decoder
        for t in range(self.T):
            # for t=0 we can use the encoding of the motion we already have
            # for t>0 we need to run cell forward on one more image
            if t == 0:
                h_cont, res_c = self.content_enc(xt, reuse=False)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse=False)
                res_connect = [self.res_weight * r for r in
                               self.residual(res_m, res_c, reuse=False)]
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse=False)
            else:
                enc_h, res_m = self.motion_enc(diff_in, reuse=True)
                h_dyn, state = cell(enc_h, state, scope='lstm', reuse=True)
                h_cont, res_c = self.content_enc(xt, reuse=reuse)
                h_tp1 = self.comb_layers(h_dyn, h_cont, reuse=True)
                res_connect = [self.res_weight * r for r in
                               self.residual(res_m, res_c, reuse=True)]
                x_hat = self.dec_cnn(h_tp1, res_connect, reuse=True)
            latents.append(h_dyn)

            # frozengens.append(self.reshape_image(xt))
            frozengens.append(self.reshape_image(x_hat))
            for l in range(self.n_latents):
                new_input = self.batch_swap(latents[0], h_dyn, l)
                frozen_h_tp1 = self.comb_layers(new_input, h_cont, reuse=True)
                frozen_gen = self.dec_cnn(
                    frozen_h_tp1, res_connect, reuse=True)
                frozengens.append(self.reshape_image(frozen_gen))
            # frozengens.append(frozengen)



            if self.c_dim == 3:
                # Network outputs are BGR so they need to be reversed to use
                # rgb_to_grayscale
                x_hat_rgb = tf.concat(axis=3,
                                      values=[x_hat[:, :, :, 2:3], x_hat[:, :, :, 1:2],
                                              x_hat[:, :, :, 0:1]])
                xt_rgb = tf.concat(axis=3,
                                   values=[xt[:, :, :, 2:3], xt[:, :, :, 1:2],
                                           xt[:, :, :, 0:1]])

                x_hat_gray = 1. / 255. * tf.image.rgb_to_grayscale(
                    inverse_transform(x_hat_rgb) * 255.
                )
                xt_gray = 1. / 255. * tf.image.rgb_to_grayscale(
                    inverse_transform(xt_rgb) * 255.
                )
            else:
                x_hat_gray = inverse_transform(x_hat)
                xt_gray = inverse_transform(xt)

            diff_in = x_hat_gray - xt_gray
            xt = x_hat
            pred.append(tf.reshape(x_hat, [self.batch_size, self.image_size[0],
                                           self.image_size[1], 1, self.c_dim]))

        return pred, latents, frozengens

    def motion_enc(self, diff_in, reuse):
        res_in = []
        conv1 = relu(conv2d(diff_in, output_dim=self.gf_dim, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='dyn_conv1', reuse=reuse))
        res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2, 2])

        conv2 = relu(conv2d(pool1, output_dim=self.gf_dim * 2, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='dyn_conv2', reuse=reuse))
        res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2, 2])

        conv3 = relu(conv2d(pool2, output_dim=self.gf_dim * 4, k_h=7, k_w=7,
                            d_h=1, d_w=1, name='dyn_conv3', reuse=reuse))
        res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2, 2])
        return pool3, res_in

    def content_enc(self, xt, reuse):
        res_in = []
        conv1_1 = relu(conv2d(xt, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv1_1', reuse=reuse))
        conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv1_2', reuse=reuse))
        res_in.append(conv1_2)
        pool1 = MaxPooling(conv1_2, [2, 2])

        conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv2_1', reuse=reuse))
        conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv2_2', reuse=reuse))
        res_in.append(conv2_2)
        pool2 = MaxPooling(conv2_2, [2, 2])

        conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv3_1', reuse=reuse))
        conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv3_2', reuse=reuse))
        conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv3_3', reuse=reuse))
        res_in.append(conv3_3)
        pool3 = MaxPooling(conv3_3, [2, 2])
        return pool3, res_in

    def comb_layers(self, h_dyn, h_cont, reuse=False):
        comb1 = relu(conv2d(tf.concat(axis=3, values=[h_dyn, h_cont]),
                            output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                            d_h=1, d_w=1, name='comb1', reuse=reuse))
        comb2 = relu(conv2d(comb1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                            d_h=1, d_w=1, name='comb2', reuse=reuse))
        h_comb = relu(conv2d(comb2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='h_comb', reuse=reuse))
        return h_comb

    def residual(self, input_dyn, input_cont, reuse=False):
        n_layers = len(input_dyn)
        res_out = []
        for l in range(n_layers):
            input_ = tf.concat(axis=3, values=[input_dyn[l], input_cont[l]])
            out_dim = input_cont[l].get_shape()[3]
            res1 = relu(conv2d(input_, output_dim=out_dim,
                               k_h=3, k_w=3, d_h=1, d_w=1,
                               name='res' + str(l) + '_1', reuse=reuse))
            res2 = conv2d(res1, output_dim=out_dim, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='res' + str(l) + '_2', reuse=reuse)
            res_out.append(res2)
        return res_out

    def dec_cnn(self, h_comb, res_connect, reuse=False):
        shapel3 = [self.batch_size, self.image_size[0] / 4,
                   self.image_size[1] / 4, self.gf_dim * 4]
        shapeout3 = [self.batch_size, self.image_size[0] / 4,
                     self.image_size[1] / 4, self.gf_dim * 2]
        depool3 = FixedUnPooling(h_comb, [2, 2])
        deconv3_3 = relu(deconv2d(relu(tf.add(depool3, res_connect[2])),
                                  output_shape=shapel3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_3', reuse=reuse))
        deconv3_2 = relu(deconv2d(deconv3_3, output_shape=shapel3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_2', reuse=reuse))
        deconv3_1 = relu(deconv2d(deconv3_2, output_shape=shapeout3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_1', reuse=reuse))

        shapel2 = [self.batch_size, self.image_size[0] / 2,
                   self.image_size[1] / 2, self.gf_dim * 2]
        shapeout3 = [self.batch_size, self.image_size[0] / 2,
                     self.image_size[1] / 2, self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        deconv2_2 = relu(deconv2d(relu(tf.add(depool2, res_connect[1])),
                                  output_shape=shapel2, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv2_2', reuse=reuse))
        deconv2_1 = relu(deconv2d(deconv2_2, output_shape=shapeout3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv2_1', reuse=reuse))

        shapel1 = [self.batch_size, self.image_size[0],
                   self.image_size[1], self.gf_dim]
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        deconv1_2 = relu(deconv2d(relu(tf.add(depool1, res_connect[0])),
                                  output_shape=shapel1, k_h=3, k_w=3, d_h=1, d_w=1,
                                  name='dec_deconv1_2', reuse=reuse))

        xtp1 = deconv2d(deconv1_2, output_shape=shapeout1, k_h=3, k_w=3,
                        d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse)
        if self.nonlinearity == 'tanh':
            xtp1 = tanh(xtp1)
        elif self.nonlinearity == 'sigmoid':
            xtp1 = tf.nn.sigmoid(xtp1)
        return xtp1

    def discriminator(self, image):
        h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'),
                              "bn1"))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'),
                              "bn2"))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'),
                              "bn3"))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h

    def factor_discriminator(self, latents):
        # h0 = lrelu(linear(latents, 128, name='fac_dis_h0_lin'))
        # h1 = lrelu(linear(h0, 128, name='fac_dis_h1_lin'))
        # # h2 = lrelu(linear(h1, 128, name='fac_dis_h2_lin'))
        # out = linear(h1, 1, name='fac_dis_out_lin')
        # return tf.nn.sigmoid(out), out
        # ipdb.set_trace()
        h0 = lrelu(conv2d(latents, self.df_dim, name='dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'),
                              "bn1"))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'),
                              "bn2"))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'),
                              "bn3"))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h


    def save(self, sess, checkpoint_dir, step):
        model_name = "MCNET.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None:
                model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: " + str(model_name))
            return True, model_name
        else:
            return False, None
