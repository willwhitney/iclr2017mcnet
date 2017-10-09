import torch.utils.data as data
from moving_mnist import data_handler as dh
import progressbar as pb
import torch
import ipdb

class MovingMNIST(data.Dataset):
    def __init__(self, train=True, seq_len=20, image_size=64, colored=False,
                 transforms=[]):
        self.transforms = transforms
        self.colored = colored
        if colored:
            handler = dh.ColoredBouncingMNISTDataHandler
        else:
            handler = dh.BouncingMNISTDataHandler

        # tiny = True
        tiny = False
        if train:
            self.data_handler = handler(seq_len, image_size)
            self.data_size = 64 * pow(2, 2) if tiny else 64 * pow(2, 12)
        else:
            self.data_handler = handler(seq_len, image_size)
            self.data_size = 64 * pow(2, 2) if tiny else 64 * pow(2, 5)

        # pbar = pb.ProgressBar()
        # self.data = []
        # print("Generating dataset:")
        # for i in pbar(range(self.data_size)):
        #     self.data.append(

        #             )

    def __getitem__(self, index):
        item = torch.from_numpy(self.data_handler.GetItem())
        item /= 255
        # ipdb.set_trace()
        # if not self.colored:
        #     item = item.unsqueeze(1)
        # else: 
        #     item = item.transpose(3, 2).transpose(2, 1)
        for transform in self.transforms:
            item = transform(item)
        return item

    def __len__(self):
        return self.data_size
