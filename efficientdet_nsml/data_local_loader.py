""" data_local_loader.py
"""

from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import os
import torch

from utils.utils import preprocess

use_float16 = False

def get_transform():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)

class CustomDataset(data.Dataset):
    def __init__(self, root, transform, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader

        self.images = []
        dir_list = sorted(os.listdir(self.root)) # 300
        
        for idx, file_path in enumerate(dir_list):
            if file_path.endswith('.jpg'):
                self.images.append(os.path.join(self.root, file_path))
                
        self.images = natsort.natsorted(self.images)
        print("data len", len(self.images))
        #self.images = self.images[:15]
        #print("data len", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #print(self.images[index])
        ori_img, framed_img = preprocess(self.images[index], max_size=1280)
        #x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        x = torch.from_numpy(framed_img)
        #x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(2, 0, 1)
        # images = self.loader(self.images[index])
        #image_name = self.images[index].split('/')[-1]
        # inputs = self.transform(images)
        return self.images[index], x


def data_loader(root, phase='train', batch_size=1):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError
    input_transform = get_transform()
    dataset = CustomDataset(root, input_transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)


def data_loader_local(root, batch_size=1):
    input_transform = get_transform()
    dataset = CustomDataset(root, input_transform)
    val_loader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    return val_loader