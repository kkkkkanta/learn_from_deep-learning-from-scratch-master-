import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from single_convnet import SingleConvNet
from common.trainer import Trainer

#データ読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = SingleConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=5, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("single_convnet_params.pkl")
print("Saved Network Parameters!")
