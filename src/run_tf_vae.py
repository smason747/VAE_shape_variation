# -*- coding: utf-8 -*-
"""
@File    : run_tf_vae_cls.py
@Time    : 12/3/2021 3:38 PM
@Author  : Mengfei Yuan
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tf_vae import VAE
from utils import load_data, train_test_split_balanced
from utils import plot_fig_recons, plot_zmeans, plot_fig_loss
from utils import plot_2d_variation



# setting tf gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus: ', gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# configs

config = {
    "model_type": "vae",
    "data_type": 'cube_slices_128',  # mnist, random_shape_28, cube_slices_28
    "nn_type": "cnn",  # dense, cnn
    "img_size": 128,
    "img_channels": 1,
    "batch_size": 128,
    "epochs": 40,

    "data_dir": "../data/",
    "model_dir": "../save_model_vae/tf/",

    "recon_loss_type": 'bce',  # bce, mse
    "is_bn": False,  # add batch_normalization or not
    "is_load_pretrain": False,

    "z_dim": 2,

    "select_seed": 2036
}


tf.random.set_seed(config["select_seed"])

# adding configs for saving plots and weights
config['img_dir'] = "%s/%s/%s_img_%s_%s_%s_bn%s_seed%s/" %(
    config['model_dir'], config['data_type'], config["model_type"],
    config['data_type'], config['nn_type'], config['recon_loss_type'], config['is_bn'],
    config["select_seed"])
os.makedirs(config["img_dir"], exist_ok=True)

config["fig_loss"] = "%s/loss.png" % (config["img_dir"])
config["fig_recons"] = "%s/recon_example.png" % (config["img_dir"])
config["fig_zmeans"] = "%s/param_predictions.png" % (config["img_dir"])
config["fig_2d_variation"] = "%s/fig_2d_variation.png" % (config["img_dir"])
config["cls_report"] = "%s/cls_report.txt" % (config["img_dir"])
config["fig_reg_true_prediction"] = "%s/reg_true_prediction" % (config["img_dir"])

config["load_model_weights"] = "%s/%s_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])
config["load_enc_weights"] = "%s/%s_enc_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])
config["load_dec_weights"] = "%s/%s_dec_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])

if __name__ == "__main__":
    print('config:\n', config)

    # network type: dense for flatten img, cnn for normal img
    if config["nn_type"] == "dense":
        is_data_flatten = True
    elif config["nn_type"] == "cnn":
        is_data_flatten = False

    # load data
    x, y = load_data(data_dir=config["data_dir"], data_type=config["data_type"], is_extra_normalization=False)
    x_train, x_test, y_train, y_test = train_test_split_balanced(x, y, test_size=0.2, img_size=config["img_size"], is_flatten=is_data_flatten)

    # build and compile vae model
    vae = VAE(img_size=config["img_size"], channels=config["img_channels"], config=config)
    if config["nn_type"] == 'dense':
        model, encoder, decoder = vae.build_dense_vae()
    elif config["nn_type"] == 'cnn':
        model, encoder, decoder = vae.build_cnn_vae()

    # train
    if config['is_load_pretrain']:
        model.load_weights(config["load_model_weights"])
        encoder.load_weights(config["load_enc_weights"])
        decoder.load_weights(config["load_dec_weights"])
    else:
        es = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')
        rp = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=0, verbose=0, mode='auto',
                               min_delta=0.0001, cooldown=0, min_lr=0)
        history = model.fit(x_train, x_train, epochs=config["epochs"], batch_size=config['batch_size'],
                            validation_data=(x_test, x_test), callbacks=[es, rp])
        model.save_weights(config["load_model_weights"])
        encoder.save_weights(config["load_enc_weights"])
        decoder.save_weights(config["load_dec_weights"])

    # evaluation
    x_test_pred = model.predict(x_test)
    config["L1_norm_recon"] = str(np.mean(np.abs(
        x_test_pred.reshape(-1, config['img_size'] ** 2) - x_test.reshape(-1, config['img_size'] ** 2))))
    plot_fig_recons(x_test, x_test_pred, config)
    if not config['is_load_pretrain']:
        plot_fig_loss(history, config)
    plot_zmeans(z_means=encoder.predict(x_test)[0], y_test=y_test, filename=config['fig_zmeans'])
    plot_2d_variation(model=[encoder, decoder], n=16, config=config)

