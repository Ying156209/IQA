import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.io as sio
import random
from PIL import Image
import pickle
import csv
import os
import os.path
from matplotlib.pyplot import *


def make_dataset(dir):
    train_list=[]
    train_mos=[]
    train_std=[]

    val_list=[]
    val_mos=[]
    val_std=[]

    length = int(0.8 * 25)
    tmp=[i for i in range(1, 26)]
    random.shuffle(tmp)
    train_num = tmp[:length]
    val_num = tmp[length:]

    f1=open(os.path.join(dir,'mos_with_names.txt'),'rb')
    f_std=open(os.path.join(dir,'mos_std.txt'),'rb')
    std=f_std.read().splitlines()

    for index, line in enumerate(f1.read().splitlines()):
        line_split=line.split()
        name = line_split[1].split('\\')[0]
        name_num=int((name.split('_')[0])[1:])
        if name_num in train_num:
            train_list.append(name)
            train_mos.append(float(line_split[0]))
            train_std.append(float(std[index]))
        else:
            val_list.append(name)
            val_mos.append(float(line_split[0]))
            val_std.append(float(std[index]))
        # print(float(line_split[0]))

    # for line in file.read().splitlines():
    #     # print(float(line))
    #     img_std.append(float(line))


    # for aug_img in os.listdir(os.path.join(dir,'Images_Aug')):
    #     li=aug_img.split('_')
    #     img_zmos.append(float(li[1]))
    #     img_std.append(float(li[2]))
    #     img_list.append(aug_img)

    return train_list,train_mos,train_std, val_list,val_mos,val_std

# def default_loader(path):
#     return Image.open(path) #

class Reg_Multi_Loader(data.Dataset):

    def __init__(self,root, stage, transform=None, target_transform=None):
        # use the same split train_val (for testing all the hypeparam)  or  new split
        self.root='../TID2013/TID_data/'
        self.stage = stage
        self.transform = transform
        self.target_transform = target_transform

        # pkl_path=os.path.join(args.plot)

        if not os.path.exists('./split/split_TID.pkl'):
            self.train_name, self.train_mos, self.train_std, self.val_name, self.val_mos, self.val_std = make_dataset(self.root)

            self.train_list=zip(self.train_name, self.train_mos, self.train_std)
            self.val_list=zip(self.val_name, self.val_mos, self.val_std)
            split_dict={'train': self.train_list,
                        'val':self.val_list}

            output=open('./split/split_TID.pkl', 'w')
            pickle.dump(split_dict,output)
            output.close()

        else:
            # data = sio.loadmat('split.mat')
            input=open('./split/split_TID.pkl', 'r')
            print('%s,load pkl'%stage)
            data=pickle.load(input)
            self.train_list = data['train']
            self.val_list = data['val']
            input.close()


    def __getitem__(self, index):
        if self.stage=='train':
            itemlist=self.train_list
        else:
            itemlist = self.val_list
        item=itemlist[index]
        path = item[0]
        mos = float(item[1]/8)
        std = float(item[2]/8)
        # target_vec=torch.gt(torch.Tensor([mos]), torch.Tensor([float(r) / self.classes for r in range(self.classes + 1)])).float()
        # weight_vec = get_weight_vec(mos, self.classes, std, self.weight)

        # if '_' not in path:    # some training sample in Aug
        #     img_path=os.path.join(self.root,'Images',path)
        # else:
        #     img_path = os.path.join(self.root, 'Images_Aug', path)
        # # s=Image.open(img_path)

        img_path = os.path.join(self.root, 'Images', path)
        sample = Image.open(img_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, mos, std


    def __len__(self):
        if self.stage == 'train':
            return len(self.train_list)
        else:
            return len(self.val_list)



# def get_weight_vec(target, classes, std):
#     vec = torch.ones(1,classes+1).squeeze(0).float()
#     # claculate the bins width: mos & std
#     mos = int(target * classes)
#     std = int(std * classes)
#     for i in range(mos - std, mos + std):
#         if i < 0 or i > classes:
#             continue
#         prob = 0.5 + 0.5/std*abs(i-mos)
#         vec[i]=prob
#     return vec



def get_weight_vec(target, classes, std, is_weight):
    vec = torch.ones(1,classes+1).squeeze(0).float()
    if is_weight==1.0:
        return vec
    else:
    # claculate the bins width: mos & std
        mos = int(target * classes)
        std = int(std * classes)
        for i in range(mos - std, mos + std):
            if i < 0 or i > classes:
                continue
            prob = is_weight + (1-is_weight)/std*abs(i-mos)
            vec[i]=prob
        return vec



# def get_target_vec(target, classes, std):
#     target_vec = torch.gt(target, torch.Tensor([float(r) / classes for r in range(classes + 1)])).float()
#     mos = int(target * classes)
#     std = int(std * classes)
#     for i in range(mos - std, mos + std):
#         if i < 0 or i > classes:
#             continue
#         prob = 1 - float(i - mos + std) / float(2 * std)
#     return target_vec



if __name__ == '__main__':
    root_dir='../TID_data/'
    # use the dataloader to load img from your own dataset
    train_loader = torch.utils.data.DataLoader(
        Reg_Multi_Loader(root_dir, 'train', transforms.Compose([
            transforms.RandomCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=4, shuffle=True,
        num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Reg_Multi_Loader(root_dir, 'val', transforms.Compose([
            transforms.RandomCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=4, shuffle=True,
        num_workers=0, pin_memory=True)


    for i, (img, mos,std) in enumerate(val_loader):
        print(i,mos,std)