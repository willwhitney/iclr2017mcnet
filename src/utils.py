"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
import scipy.misc
import numpy as np
import ipdb
import torch
import socket
import os
from moving_mnist.dataset import MovingMNIST
from video_dataset import *

def transform(image):
    return image / 127.5 - 1.


def transform_from01(image):
    return image * 2 - 1


def inverse_transform(images):
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    return imsave(inverse_transform(images) * 255., size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """ 
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
    if img.shape[2] == 1:
        img = np.repeat(img, [3], axis=2)

    if is_input:
        img[:2, :, 0] = img[:2, :, 2] = 0
        img[:, :2, 0] = img[:, :2, 2] = 0
        img[-2:, :, 0] = img[-2:, :, 2] = 0
        img[:, -2:, 0] = img[:, -2:, 2] = 0
        img[:2, :, 1] = 255
        img[:, :2, 1] = 255
        img[-2:, :, 1] = 255
        img[:, -2:, 1] = 255
    else:
        img[:2, :, 0] = img[:2, :, 1] = 0
        img[:, :2, 0] = img[:, :2, 2] = 0
        img[-2:, :, 0] = img[-2:, :, 1] = 0
        img[:, -2:, 0] = img[:, -2:, 1] = 0
        img[:2, :, 2] = 255
        img[:, :2, 2] = 255
        img[-2:, :, 2] = 255
        img[:, -2:, 2] = 255

    return img


def load_kth_data(f_name, data_path, image_size, K, T):
    flip = np.random.binomial(1, .5, 1)[0]
    tokens = f_name.split()
    vid_path = data_path + tokens[0] + "_uncomp.avi"
    vid = imageio.get_reader(vid_path, "ffmpeg")
    low = int(tokens[1])
    high = np.min([int(tokens[2]), vid.get_length()]) - K - T + 1
    if low == high:
        stidx = 0
    else:
        if low >= high:
            print(vid_path)
        stidx = np.random.randint(low=low, high=high)
    seq = np.zeros((image_size, image_size, K + T, 1), dtype="float32")
    for t in range(K + T):
        img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t),
                                      (image_size, image_size)),
                           cv2.COLOR_RGB2GRAY)
        seq[:, :, t] = transform(img[:, :, None])

    if flip == 1:
        seq = seq[:, ::-1]

    diff = np.zeros((image_size, image_size, K - 1, 1), dtype="float32")
    for t in range(1, K):
        prev = inverse_transform(seq[:, :, t - 1])
        next = inverse_transform(seq[:, :, t])
        diff[:, :, t - 1] = next.astype("float32") - prev.astype("float32")

    return seq, diff

def make_diffs(seq, K):
    # ipdb.set_trace()
    diff = np.zeros((seq.shape[0], seq.shape[1],
                     K - 1, 1), dtype="float32")
    for t in range(1, K):
        if seq.shape[3] == 1:
            prev = inverse_transform(seq[:, :, t - 1])
            nex = inverse_transform(seq[:, :, t])
            diff[:, :, t - 1] = nex.astype("float32") - prev.astype("float32")
        else:
            prev = inverse_transform(seq[:, :, t - 1]) * 255
            prev = cv2.cvtColor(prev.astype("uint8"), cv2.COLOR_BGR2GRAY)
            nex = inverse_transform(seq[:, :, t]) * 255
            nex = cv2.cvtColor(nex.astype("uint8"), cv2.COLOR_BGR2GRAY)
            diff[:, :, t - 1, 0] = (nex.astype("float32") -
                                    prev.astype("float32")) / 255.

    return diff


def load_s1m_data(f_name, data_path, trainlist, K, T):
    flip = np.random.binomial(1, .5, 1)[0]
    vid_path = data_path + f_name
    img_size = [240, 320]

    while True:
        try:
            vid = imageio.get_reader(vid_path, "ffmpeg")
            low = 1
            high = vid.get_length() - K - T + 1
            if low == high:
                stidx = 0
            else:
                stidx = np.random.randint(low=low, high=high)
            seq = np.zeros((img_size[0], img_size[1], K + T, 3),
                           dtype="float32")
            for t in range(K + T):
                img = cv2.resize(vid.get_data(stidx + t),
                                 (img_size[1], img_size[0]))[:, :, ::-1]
                seq[:, :, t] = transform(img)

            if flip == 1:
                seq = seq[:, ::-1]

            diff = np.zeros((img_size[0], img_size[1], K - 1, 1),
                            dtype="float32")
            for t in range(1, K):
                prev = inverse_transform(seq[:, :, t - 1]) * 255
                prev = cv2.cvtColor(prev.astype("uint8"), cv2.COLOR_BGR2GRAY)
                next = inverse_transform(seq[:, :, t]) * 255
                next = cv2.cvtColor(next.astype("uint8"), cv2.COLOR_BGR2GRAY)
                diff[:, :, t - 1, 0] = (next.astype("float32") -
                                        prev.astype("float32")) / 255.
            break
        except Exception:
            # In case the current video is bad load a random one
            rep_idx = np.random.randint(low=0, high=len(trainlist))
            f_name = trainlist[rep_idx]
            vid_path = data_path + f_name

    return seq, diff


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
             hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images) - 1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding:
                   (i + 1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images) - 1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding:
                   (i + 1) * y_dim + i * padding].copy_(image)
        return result


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    return scipy.misc.toimage(tensor.numpy(),
                              high=255 * tensor.max(),
                              channel_axis=0)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)


def save_gif(filename, inputs, bounce=False, duration=0.2):
    images = []
    for tensor in inputs:
        tensor = tensor.cpu()
        tensor = tensor.transpose(0, 1).transpose(1, 2).clamp(0, 1)
        images.append(tensor.cpu().numpy())
    if bounce:
        images = images + list(reversed(images[1:-1]))
    imageio.mimsave(filename, images, duration=duration)


def ensure_path_exists(fn):
    """
    A decorator which, given a function that has a path as its first argument,
    ensures that the directory containing that path exists,
    creating it if necessary.
    """
    def wrapper(path, *args, **kwargs):
        try:
            return fn(path, *args, **kwargs)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return fn(path, *args, **kwargs)
    return wrapper


def load_dataset(data, seq_len, image_width, channels, transforms=[]):
    if socket.gethostname() == 'zaan':
        data_path = '/speedy/data/' + data
        # data_path = '/speedy/data/urban/5th_ave'
    else:
        data_path = '/misc/vlgscratch4/FergusGroup/wwhitney/' + data
        # data_path = '/misc/vlgscratch3/FergusGroup/wwhitney/urban/5th_ave'

    if data == 'sample':
        train_data, test_data = make_split_datasets(
            '.', 5, 4, image_width=image_width, chunk_length=50)
        load_workers = 0
    # 'urban' datasets are in-memory stores
    elif data_path.find('urban') >= 0:
        if socket.gethostname() == 'zaan':
            data_path = '/home/will/data/' + data
        if not data_path[-3:] == '.t7':
            data_path = data_path + '/dataset.t7'

        print("Loading stored dataset from {}".format(data_path))
        data_checkpoint = torch.load(data_path)
        train_data = data_checkpoint['train_data']
        test_data = data_checkpoint['test_data']

        train_data.transforms = transforms
        test_data.transforms = transforms

        if train_data.image_size[1] != image_width:
            train_data.resize_([train_data.image_size[0],
                                image_width,
                                image_width])
            test_data.resize_([test_data.image_size[0],
                                image_width,
                                image_width])
        train_data.seq_len = seq_len
        test_data.seq_len = seq_len

        load_workers = 0

    # elif data == 'atari':
    #     train_data = AtariData(
    #         opt.game, 'train', seq_len, image_width)
    #     test_data = AtariData(
    #         opt.game, 'test', seq_len, image_width)
    #     load_workers = 0

    # elif data == 'mnist':
    #     train_data = datasets.MNIST('../data', train=True, download=True,
    #                                 transform=transforms.Compose([
    #                                     transforms.Scale(image_width),
    #                                     transforms.ToTensor()]))
    #     test_data = datasets.MNIST('../data', train=False,
    #                                 transform=transforms.Compose([
    #                                     transforms.Scale(image_width),
    #                                     transforms.ToTensor()]))
    #     load_workers = 1

    elif data == 'mmnist':
        dataset_name = "channels{}_width{}_seqlen{}.t7".format(
            channels, image_width, seq_len)
        dataset_path = os.path.join(data_path, dataset_name)

        if os.path.exists(dataset_path):
            data_cp = torch.load(dataset_path)
            train_data = data_cp['train']
            test_data = data_cp['test']
        else:
            train_data = MovingMNIST(train=True,
                                     seq_len=seq_len,
                                     image_size=image_width,
                                     colored=(channels == 3),
                                     transforms=transforms)
            test_data = MovingMNIST(train=False,
                                    seq_len=seq_len,
                                    image_size=image_width,
                                    colored=(channels == 3),
                                    transforms=transforms)
            # pdb.set_trace()
            # torch.save({
            #     'train': train_data,
            #     'test': test_data,
            # }, dataset_path)
        load_workers = 1

    # other video datasets are big and stored as chunks
    else:
        if hostname != 'zaan':
            scratch_path = '/scratch/wwhitney/' + data
            vlg_path = '/misc/vlgscratch4/FergusGroup/wwhitney/' + data

            data_path = vlg_path
            # if os.path.exists(scratch_path):
            #     data_path = scratch_path
            # else:
            #     data_path = vlg_path

        print("Loading stored dataset from {}".format(data_path))
        train_data, test_data = load_disk_backed_data(data_path)

        load_workers = 4

        train_data.transforms = transforms
        test_data.transforms = transforms

        train_data.framerate = opt.fps
        test_data.framerate = opt.fps

        train_data.seq_len = opt.seq_len
        test_data.seq_len = opt.seq_len
    return train_data, test_data, load_workers


def normalize_data(data, channels, K, sequence):
    if data == 'mmnist':
        # color channel is already at the end
        if channels == 1:
            # make a color channel since there won't be one
            sequence.unsqueeze_(3)
    else:
        # move color channel to the end
        # ipdb.set_trace()
        sequence.transpose_(1, 2).transpose_(2, 3)
    # move sequence channel to one away from the end
    sequence.transpose_(0, 1).transpose_(1, 2)
    sequence = sequence.cpu().numpy()
    sequence = transform_from01(sequence)
    # ipdb.set_trace()
    diff = make_diffs(sequence, K)

    if data == 'mmnist':
        # since mmnist is so near zero it saturates at -1; 
        # put the data back to [0, 1] so the tanh can't saturate
        sequence = inverse_transform(sequence)
    
    return sequence, diff
