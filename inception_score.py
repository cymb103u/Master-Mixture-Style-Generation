"""
Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.:
Improved techniques for training gans. In: NIPS. (2016)
code is  modified  from https://github.com/sbarratt/inception-score-pytorch
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

import torchvision.datasets as dset
import torchvision.transforms as transforms

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

if __name__ == '__main__':


    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))



# # calculate inception score in numpy
# from numpy import asarray
# from numpy import expand_dims
# from numpy import log
# from numpy import mean
# from numpy import exp
 
# # calculate the inception score for p(y|x)
# def calculate_inception_score(p_yx, eps=1E-16):
# 	# calculate p(y)
# 	p_y = expand_dims(p_yx.mean(axis=0), 0)
# 	# kl divergence for each image
# 	kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
# 	# sum over classes
# 	sum_kl_d = kl_d.sum(axis=1)
# 	# average over images
# 	avg_kl_d = mean(sum_kl_d)
# 	# undo the logs
# 	is_score = exp(avg_kl_d)
# 	return is_score
 
# # conditional probabilities for high quality images
# p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# score = calculate_inception_score(p_yx)
# print(score)