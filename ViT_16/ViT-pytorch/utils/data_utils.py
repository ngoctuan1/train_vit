import logging
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
from torch.utils.data import Dataset

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.io import read_image
from PIL import Image


logger = logging.getLogger(__name__)

class Track4_Dataset(Dataset):
  def __init__(self, df, transform = None, target_transform = None):
    self.df = df
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, index):
    row = self.df.loc[index]
    img = Image.open(row.image_path)
    label = int(row.class_id) - 1
    if self.transform is not None:
      img = self.transform(img)

    # if self.target_transform is not None:
    #   label = self.target_transform(label)

    return img, label

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "track4":
        lst_imgs = glob(f'{args.img_path}/*.jpg')
        df = pd.DataFrame(lst_imgs, columns=['image_path'])
        df_train, df_test = train_test_split(df, test_size = 0.2, random_state=42)

        df_train['class_id'] = df_train.apply(lambda x: int(x.image_path.split("/")[-1].split("_")[0]), axis = 1)
        df_test['class_id'] = df_test.apply(lambda x: int(x.image_path.split("/")[-1].split("_")[0]), axis = 1)

        df_train.reset_index(inplace=True)
        df_test.reset_index(inplace=True)

        trainset = Track4_Dataset(df_train, transform=transform_train, target_transform=target_transform)
        testset = Track4_Dataset(df_test, transform=transform_test, target_transform=target_transform)
        

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
