import torch
import shutil
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         print('save the best model for now')
#         shutil.copyfile(filename, 'model_best.pth.tar')

def save_checkpoint(state, is_best, dir, filename='model_best.pth.tar'):
    if is_best:
        print('save the best model for now')
        torch.save(state, os.path.join(dir,filename))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lambda(use_weight):
    if use_weight:
        input = open('split.pkl', 'r')
        data = pickle.load(input)
        train_score = list(zip(*data['train'])[1])
        train_score.sort()

        # base: avoid 0:1000  ->  base+0: base+1000  ,  add two base

        hist, _ = np.histogram(train_score, bins=100, range=(0,100))
        mount=max(hist)
        hist=hist.astype('float32')

        base=1.5

        weights=(base-hist/mount)/base

    else:
        weights = np.ones(100)

    # plt.ylim(0,3)
    # plt.plot(weights)
    # plt.show()

    return weights

    # left_weights=np.ones(class_num)





def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div(255.0)
    return (batch - mean) / std



if __name__ == '__main__':
    get_lambda(100)
