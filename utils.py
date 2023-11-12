import errno
import os
import random

from torchvision.transforms import transforms

import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from PIL import Image
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn.manifold import TSNE
import seaborn as sns


class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root = root
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        # read the iterable image
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        # label
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)


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


def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def str2bool(str):
    return True if str.lower() == 'true' else False


def extract_attr_w2v_CUB():
    model = KeyedVectors.load_word2vec_format(datapath(f'/home/c402/data/XL/GoogleNews-vectors-negative300.bin.gz'), binary=True)
    dim_w2v = 300
    replace_word = [('spatulate', 'broad'), ('upperparts', 'upper parts'), ('grey', 'gray')]
    path = '/home/c402/data/Dataset/CUB_200_2011/attributes.txt'
    df = pd.read_csv(path, sep=' ', header=None, names=['idx', 'des'])
    des = df['des'].values
    new_des = [' '.join(i.split('_')) for i in des]
    new_des = [' '.join(i.split('-')) for i in new_des]
    new_des = [' '.join(i.split('::')) for i in new_des]
    new_des = [i.split('(')[0] for i in new_des]
    new_des = [i[4:] for i in new_des]
    for pair in replace_word:
        for idx, s in enumerate(new_des):
            new_des[idx] = s.replace(pair[0], pair[1])
    df['new_des'] = new_des
    df.to_csv('../attribute/CUB/new_des.csv')
    all_w2v = []
    for s in new_des:
        words = s.split(' ')
        if words[-1] == '':  # remove empty element
            words = words[:-1]
        w2v = np.zeros(dim_w2v)
        for w in words:
            try:
                w2v += model[w]
            except Exception as e:
                print(e)
        all_w2v.append(w2v[np.newaxis, :])
    all_w2v = np.concatenate(all_w2v, axis=0)
    return all_w2v


def extract_attr_w2v_AWA2():
    model = KeyedVectors.load_word2vec_format(datapath(f'/home/c402/data/XL/GoogleNews-vectors-negative300.bin.gz'), binary=True)
    dim_w2v = 300
    replace_word = [('newworld', 'new world'), ('oldworld', 'old world'), ('nestspot', 'nest spot'),
                    ('toughskin', 'tough skin'),
                    ('longleg', 'long leg'), ('chewteeth', 'chew teeth'), ('meatteeth', 'meat teeth'),
                    ('strainteeth', 'strain teeth'),
                    ('quadrapedal', 'quadrupedal')]
    dataset = 'AWA2'
    path = '/home/c402/data/Dataset/Animals_with_Attributes2/predicates.txt'
    df = pd.read_csv(path, sep='\t', header=None, names=['idx', 'des'])
    des = df['des'].values
    for pair in replace_word:
        for idx, s in enumerate(des):
            des[idx] = s.replace(pair[0], pair[1])
    df['new_des'] = des
    df.to_csv('../attribute/{}/new_des.csv'.format(dataset))
    counter_err = 0
    all_w2v = []
    for s in des:
        words = s.split(' ')
        if words[-1] == '':  # remove empty element
            words = words[:-1]
        w2v = np.zeros(dim_w2v)
        for w in words:
            try:
                w2v += model[w]
            except Exception as e:
                print(e)
                counter_err += 1
        all_w2v.append(w2v[np.newaxis, :])
    all_w2v = np.concatenate(all_w2v, axis=0)
    return all_w2v


def extract_attr_w2v_SUN():
    model = KeyedVectors.load_word2vec_format(datapath(f'/home/c402/data/XL/GoogleNews-vectors-negative300.bin.gz'), binary=True)
    dim_w2v = 300
    print('Done loading model')
    replace_word = [('rockstone', 'rock stone'), ('dirtsoil', 'dirt soil'), ('man-made', 'man-made'),
                    ('sunsunny', 'sun sunny'),
                    ('electricindoor', 'electric indoor'), ('semi-enclosed', 'semi enclosed'), ('far-away', 'faraway')]
    file_path = '/home/c402/data/Dataset/SUN/SUNAttributeDB/attributes.mat'
    matcontent = sio.loadmat(file_path)
    des = matcontent['attributes'].flatten()
    df = pd.DataFrame()
    new_des = [''.join(i.item().split('/')) for i in des]
    for pair in replace_word:
        for idx, s in enumerate(new_des):
            new_des[idx] = s.replace(pair[0], pair[1])
    df['new_des'] = new_des
    df.to_csv('../attribute/{}/new_des.csv'.format('SUN'))
    all_w2v = []
    for s in new_des:
        words = s.split(' ')
        if words[-1] == '':  # remove empty element
            words = words[:-1]
        w2v = np.zeros(dim_w2v)
        for w in words:
            try:
                w2v += model[w]
            except Exception as e:
                print(e)
        all_w2v.append(w2v[np.newaxis, :])
    all_w2v = np.concatenate(all_w2v, axis=0)
    return all_w2v


def tsne_attribute(model, dataloader):
    features = []
    colors = []

    # all_attr_num = 102
    # all_attr_num = 85
    all_attr_num = 312
    sample_attr_num = 20
    sample_num_per_attr = 200
    # attr_names = get_attr_name(f'/home/c402/data/Dataset/SUN/SUN_attr.txt')
    attr_names = get_attr_name(f'/home/c402/data/Dataset/CUB_200_2011/attributes.txt')
    # attr_names = get_attr_name(f'/home/c402/data/Dataset/Animals_with_Attributes2/predicates.txt')

    random_attr = np.array(random.sample(list(range(all_attr_num)), sample_attr_num))

    palette = sns.color_palette("hls", sample_attr_num)

    root = dataloader.dataset.root
    random_samples = random.sample(list(dataloader.dataset.image_files), sample_num_per_attr)
    for i in range(sample_num_per_attr):
        colors.append(list(random_attr))
        print(random_samples[i])
    colors = np.concatenate(colors, 0)

    print("Selected samples : ", random_samples)

    with torch.no_grad():
        for image_file in random_samples:
            img = Image.open(os.path.join(root, image_file))
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            img_tensor = transform(img).unsqueeze(dim=0).cuda()
            feature = model(img_tensor)['attr_v_feature'][-1, random_attr, :]
            features.append(feature)

    features = torch.cat(features, dim=0)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(features.data.cpu().numpy())

    plt.figure()

    for i, label in enumerate(random_attr):
        indices = np.where(colors == label)[0]
        label_name = str(label) + '_' + str(attr_names[label])[:-1]
        label_name = label_name.split()[0].split('_')[1]
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=np.array(palette[i]).reshape(1, -1), label=label_name)

    # add labels
    # for idx, label in enumerate(colors):
    #     plt.annotate(label, (X_tsne[idx, 0], X_tsne[idx, 1]))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def tsne_class(model, dataloader):
    features = []
    colors = []
    root = dataloader.dataset.root

    class_num = 15
    sample_num_per_class = 20
    class_names = []

    palette = sns.color_palette("hls", class_num)

    labels = dataloader.dataset.labels
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    selected_labels = np.random.choice(unique_labels, size=class_num, replace=False)
    selected_samples = []
    for label in selected_labels:
        samples_with_label = np.where(labels == label)[0]
        selected_samples_with_label = np.random.choice(samples_with_label, size=sample_num_per_class, replace=False)
        selected_samples.extend(selected_samples_with_label.tolist())
        class_name = dataloader.dataset.image_files[selected_samples_with_label[0]].split('/')[1]
        class_names.append(class_name)
        for i in range(sample_num_per_class):
            colors.append(label)

    print("Selected labels : ", selected_labels)

    with torch.no_grad():
        for idx in selected_samples:
            image_file = dataloader.dataset.image_files[idx]
            img = Image.open(os.path.join(root, image_file))
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            img_tensor = transform(img).unsqueeze(dim=0).cuda()
            feature = model(img_tensor)['global_feature']
            features.append(feature)

    features = torch.cat(features, dim=0)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(features.data.cpu().numpy())

    plt.figure(figsize=(10, 8))

    for i, label in enumerate(selected_labels):
        indices = np.where(colors == label)[0]
        label_name = str(label) + '_' + str(class_names[i])
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=np.array(palette[i]).reshape(1, -1), label=label_name)

    # add labels
    for idx, label in enumerate(colors):
        plt.annotate(label, (X_tsne[idx, 0], X_tsne[idx, 1]))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def s2v_tsne(model):
    colors = np.arange(102)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    s2v = model(torch.rand(1, 3, 448, 448).cuda())['s2v_embedding']
    X_tsne = tsne.fit_transform(s2v.data.cpu().numpy())
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='jet')

    # add labels
    for idx, label in enumerate(colors):
        plt.annotate(label, (X_tsne[idx, 0], X_tsne[idx, 1]))

    plt.show()


def get_attr_name(attr_file):
    lines = []
    with open(attr_file, 'r') as file:
        for line in file:
            lines.append(line)
    return lines

if __name__ == '__main__':
    w2v = extract_attr_w2v_SUN()
    colors = np.arange(102)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(w2v)
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='jet')

    # add labels
    for idx, label in enumerate(colors):
        plt.annotate(label, (X_tsne[idx, 0], X_tsne[idx, 1]))

    plt.show()
