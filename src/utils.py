# -*- coding: utf-8 -*-
"""
@File    : utils.py
@Time    : 8/3/2021 3:38 PM
@Author  : Mengfei Yuan
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import random


def load_data(data_dir, data_type, is_extra_normalization=False):
    """"""
    if data_type == 'mnist': # classification
        x, y = load_mnist(path=data_dir + 'mnist.npz')
    elif data_type == 'random_shape_28': # classification
        x, y = load_random_shape(path=data_dir + 'random_shape_c7_s2000d_diffused.csv')
    elif data_type == 'random_shape_28_manim':  # regression
        x, y = load_random_shape_manim(path=data_dir + 'manim_shapeData_10515frames.csv')
    elif data_type == 'cube_slices_28': # classification
        x, y = load_cube(path=data_dir + 'cubeSliceData_28px_centered_minArea_rotated.csv')
    elif data_type == 'cube_slices_128_v2': # classification
        x, y = load_cube(path=data_dir + 'cubeSliceData_128px_centered_minArea_noRotate.csv')
    if is_extra_normalization:
        x = (x - 0.5) / 0.5  # from [0, 1] map to [-1, 1]s

    n_cluster = len(np.unique(y))
    print("x.shape: %s, y.shape: %s" % (x.shape, y.shape))
    print("number of class: %s" % (n_cluster))
    print("min and max of x: %.2f, %.2f" % (x.min(), x.max()))
    return x, y


def load_mnist(path):
    """"""
    data = np.load(path)
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

    x = np.concatenate((x_test, x_train))
    y = np.concatenate((y_test, y_train))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y

def load_random_shape(path):
    """"""
    img = pd.read_csv(path, header=None)
    x = img.iloc[:, 1:].values
    y = img.iloc[:, :1].values
    return x, y

def load_cube(path):
    
    img = pd.read_csv(path, header=None)
    print(img.shape)

    x = img.iloc[:, 1:].values
    y = img.iloc[:, :1].values

    x = np.divide(x, 255)

    return x, y


def load_label2params(data_path='../../data/vtk2D_disc28.csv',
                      label2params_path='../../data/label2params.json'):

    """original data are start with true ic and kappa parameters, map them to classes"""

    if not os.path.exists(label2params_path):
        spinodal_img = pd.read_csv(data_path)
        unique_ic = spinodal_img.iloc[:, 0].unique().tolist()
        unique_kappa = spinodal_img.iloc[:, 1].unique().tolist()

        label2params, params2label = dict(), dict()
        cnt = 0
        for ic in unique_ic:
            for kappa in unique_kappa:
                label2params[cnt] = tuple((ic, kappa))
                # params2label[ic][kappa] = cnt
                cnt += 1
        with open(label2params_path, 'w', encoding='utf-8') as f:
            json.dump(label2params, f, indent=4, ensure_ascii=False)
    else:
        label2params = json.load(open(label2params_path, 'r', encoding='utf-8'))

    return label2params


def load_spinodal_cls(path):

    spinodal_img = pd.read_csv(path, header=None)
    x = spinodal_img.iloc[:, 2:].values
    y_params = spinodal_img.iloc[:, :2].values

    label2params = load_label2params(data_path=path,
                                     label2params_path='../../data/label2params.json')
    params2label = dict()
    for k, v in label2params.items():
        params2label[tuple(v)] = k

    y = []
    for ic, kappa in y_params:
        y.append(int(params2label[(ic, kappa)]))
    y = np.array(y)

    return x, y


def load_random_shape_manim(path):
    """"""
    df = pd.read_csv(path, header=None)
    df.iloc[:, 1:] = df.iloc[:, 1:] / 255
    x = df.iloc[:, 1:].values
    y = df.iloc[:, 0:1].values
    x = x.reshape(-1, 28 * 28)
    return x, y


def load_spinodal_reg(path):

    spinodal_img = pd.read_csv(path, header=None)
    x = spinodal_img.iloc[:, 2:].values
    y = spinodal_img.iloc[:, 0:2].values

    # reshape to 28 * 28 to adopt into the existing networks
    # x = x.reshape(-1, 32, 32)
    # x = x[:, :28, :28]
    x = x.reshape(-1, 28 * 28)
    #
    # x = (x - x.min()) / (x.max() - x.min())
    # x[x <= 0.01] = 0
    # x[x >= 0.99] = 1
    return x, y


def train_test_split_balanced(x, y, test_size=0.2, img_size=128, is_flatten=True):

    if y.shape[0] == 1:  # y is single label
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        x_train, x_test, y_train, y_test = [], [], [], []

        for train_ind, test_ind in sss.split(x, y):
            x_train, x_test = x[train_ind], x[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]
    else:
        ind = [i for i in range(len(x))]
        random.shuffle(ind)
        train_ind = ind[:int((1-test_size)*len(ind))]
        test_ind = ind[int((1-test_size)*len(ind)):]
        x_train, x_test = x[train_ind], x[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

    if is_flatten == False:
        x_train = x_train.reshape(x_train.shape[0], img_size, img_size).astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0], img_size, img_size).astype(np.float32)

    if x_train.shape[-1] != 1:
        # output size should be (n_samples, flatten_dim, 1) or (n_samples, img_size, img_size, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    print("x_train.shape: %s, x_test.shape: %s, y_train.shape: %s, y_test.shape: %s\n" % (
        x_train.shape, x_test.shape, y_train.shape, y_test.shape))
    return x_train, x_test, y_train, y_test


def plot_random_sample(x, y, size=1):
    """plot a random image in dataset"""
    # x: (n_samples, flatten_dim, 1)
    idx = np.random.randint(0, x.shape[0], size=size)
    idx = idx[0]

    cmap = matplotlib.colors.ListColormap(['grey', 'white'], name="from_list", N=None)
    img = x[idx].squeeze()
    img_size = int(len(img)**0.5)
    img = img.reshape(img_size, img_size)
    plt.imshow(img, cmap=cmap)
    plt.title('class %s' %(y[idx].squeeze()))
    # plt.show()

def plot_one_sample(x, y, save_path=None):
    """x: (28,28), y is label"""
    plt.figure()
    plt.imshow(x, cmap=plt.cm.Greys)
    plt.title('class %s' %(str(int(y))))
    plt.colorbar()
    if save_path != None:
        plt.savefig(save_path)


def plot_fig_recons(x_test, x_test_pred, config):
    for _ in range(10):
        plt.figure(figsize=(8,4))
        J = np.random.randint(x_test.shape[0], size=1)
        plt.subplot(121)
        img_temp = x_test[J, :]
        img_temp = np.array(img_temp).reshape(config["img_size"], config["img_size"])
        plt.imshow(img_temp, cmap=plt.cm.Greys)
        plt.title('True image')
        plt.subplot(122)
        img_temp = x_test_pred[J, :]
        img_temp = np.array(img_temp).reshape(config["img_size"], config["img_size"])
        plt.imshow(img_temp, cmap=plt.cm.Greys)
        plt.title('Predicted image')
        plt.savefig(config["fig_recons"][:-4] + '_' + str(J[0]) + '.png')
    # plt.show()


def get_classification_report(y_true, y_pred):
    """y_pred is the output after sigmoid/softmax"""
    y_pred_01 = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))], dtype='uint8')

    report = classification_report(y_true, y_pred_01)
    acc = accuracy_score(y_true, y_pred_01)

    print('classification_report:\n', report)
    return y_pred_01, acc, report


def plot_zmeans(z_means, y_test, filename):

    print(len(z_means), len(y_test))
    print(z_means[0])
    plt.figure(figsize=(12, 10))
    plt.scatter(z_means[:, 0], z_means[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.xlabel("z[1]")
    plt.savefig(filename)
    # plt.show()



def plot_fig_loss(history, config):
    # plot all types of loss
    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('vae loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(1,3,2)
    plt.plot(history.history['kl_loss'])
    plt.plot(history.history['val_kl_loss'])
    plt.title('kl loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(1,3,3)
    plt.plot(history.history['recon_loss'])
    plt.plot(history.history['val_recon_loss'])
    plt.title('recon loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper right')

    plt.savefig(config['fig_loss'])

def plot_2d_variation(model, n, config):

    encoder, decoder = model
    img_size = config["img_size"]

    figure = np.zeros((img_size*n, img_size*n))
    grid_x = np.linspace(-8,8,n)
    grid_y = np.linspace(-8,8,n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            img_decoded = decoder.predict(z_sample)
            img = img_decoded.reshape(img_size, img_size)
            figure[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size] = img

    plt.figure(figsize=(10,10))
    start = img_size //2
    end = (n-1)*img_size + start + 1
    pixel_range = np.arange(start, end, img_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.imshow(figure, cmap=plt.cm.Greys)
    plt.savefig(config['fig_2d_variation'])



def plot_true_prediction(y_pred, y_true, filename):
    c_pred = [i[0] for i in y_pred]
    k_pred = [i[1] for i in y_pred]
    c_true = [i[0] for i in y_true]
    k_true = [i[1] for i in y_true]

    plt.figure(figsize=(12,10))
    plt.plot(c_true, c_true, 'b-')
    plt.plot(c_true, c_pred, 'r*')
    plt.xlabel("true initial composition")
    plt.ylabel("predicted initial composition")
    plt.savefig(filename + '.init_comp.png')

    plt.figure(figsize=(12,10))
    plt.plot(k_true, k_true, 'b-')
    plt.plot(k_true, k_pred, 'r*')
    plt.xlabel("true interfacial energy")
    plt.ylabel("predicted interfacial energy")
    plt.savefig(filename + '.interf_energy.png')

    with open(filename + 'true_pred_data.txt', 'w') as f:
        for i in range(len(c_pred)):
            f.write(str(c_true[i])+'\t'+str(c_pred[i])+'\t'+str(k_true[i])+'\t'+str(k_pred[i])+'\n')

