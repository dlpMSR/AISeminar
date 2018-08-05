import chainer
from chainer import datasets
from chainer import functions as F 
from chainer import links as L 
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from PIL import Image
import numpy as np

import os
import glob
from itertools import chain


class HotDogDetecter(chainer.Chain):
    def __init__(self):
        super(HotDogDetecter, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=8, ksize=3)
            self.conv2 = L.Convolution2D(None, out_channels=16, ksize=3)
            self.fc1 = L.Linear(None, 256)
            self.fc2 = L.Linear(256, 2)
        

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, ksize=3, stride=1)
        h = F.sigmoid(self.fc1(h))
        h = F.softmax(self.fc2(h))
        return h


def mnist_train():
    #データセットの取得
    train_full, test_full = datasets.get_mnist(ndim=3)
    train = datasets.SubDataset(train_full, 0, 1000)
    test = datasets.SubDataset(test_full, 0, 1000)

    #Set up a iterator
    batchsize = 100
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    model = L.Classifier(HotDogDetecter())
    opt = chainer.optimizers.Adam()
    opt.setup(model)

    epoch = 100

    updater = training.StandardUpdater(train_iter, opt, device=0)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    #trainer.extend(extensions.Snapshot((10, 'epoch')))

    trainer.run()


def hotdog_train():
    #データセットの取得
    dataset = load_images()
    train, test = datasets.split_dataset_random(dataset, int(len(dataset) * 0.8))

    #Set up a iterator
    batchsize = 1
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    model = L.Classifier(HotDogDetecter())
    opt = chainer.optimizers.Adam()
    opt.setup(model)

    epoch = 20

    updater = training.StandardUpdater(train_iter, opt, device=-1)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    #trainer.extend(extensions.Snapshot((10, 'epoch')))

    trainer.run()


def load_images():
    IMG_DIR = './data/'
    dir_names = glob.glob('{}/*'.format(IMG_DIR))
    file_names = [glob.glob('{}/*.jpg'.format(dir)) for dir in dir_names]
    file_names = list(chain.from_iterable(file_names))
    
    labels = [os.path.basename(os.path.dirname(file)) for file in file_names]
    dir_names = [os.path.basename(dir) for dir in dir_names]
    labels = [dir_names.index(label) for label in labels]

    d = datasets.LabeledImageDataset(list(zip(file_names, labels)))

    def resize(img):
        width, height = 256, 256
        img = Image.fromarray(img.transpose(1, 2, 0))
        img = img.resize((width, height), Image.BICUBIC)
        return np.asarray(img).transpose(2, 0, 1)
    
    def transform(inputs):
        img, label = inputs
        img = img[:3, ...]
        img = resize(img.astype(np.uint8))
        img = img.astype(np.float32)
        return img, label
    
    transformed_d = datasets.TransformDataset(d, transform)

    return transformed_d


def main():
    hotdog_train()
    #mnist_train()


if __name__ == '__main__':
    main()