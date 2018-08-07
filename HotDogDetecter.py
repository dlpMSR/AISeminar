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


class Normal(chainer.Chain):
    def __init__(self):
        super(Normal, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=8, ksize=3)
            self.conv2 = L.Convolution2D(None, out_channels=16, ksize=3)
            self.fc1 = L.Linear(None, 256)
            self.fc2 = L.Linear(None, 2)
        
    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, ksize=3, stride=1)
        h = F.sigmoid(self.fc1(h))
        h = F.softmax(self.fc2(h))
        return h


class Normalize(chainer.Chain):
    def __init__(self):
        super(Normalize, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=8, ksize=3)
            self.bn1 = L.BatchNormalization(8)
            self.conv2 = L.Convolution2D(None, out_channels=16, ksize=3)
            self.fc1 = L.Linear(None, 256)
            self.bn2 = L.BatchNormalization(256)
            self.fc2 = L.Linear(None, 3)
    
    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, ksize=3, stride=1)
        h = F.sigmoid(self.fc1(h))
        h = self.bn2(h)
        h = F.softmax(self.fc2(h))
        return h


def cifar10_train():
    #データセットの取得
    train_full, test_full = datasets.get_cifar10()
    train = datasets.SubDataset(train_full, 0, 1000)
    test = datasets.SubDataset(test_full, 0, 1000)

    #Set up a iterator
    batchsize = 60
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    model = L.Classifier(Normal())
    opt = chainer.optimizers.Adam()
    opt.setup(model)

    epoch = 200

    updater = training.StandardUpdater(train_iter, opt, device=0)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    #trainer.extend(extensions.Snapshot((10, 'epoch')))

    trainer.run()


def hotdog_train():
    train = load_images('./data/train')
    test = load_images('./data/test')
    #dataset = load_images()
    #train, test = datasets.split_dataset_random(dataset, int(len(dataset) * 0.9))

    batchsize = 16
    epoch = 20
    gpu_id = 0

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    model = L.Classifier(Normal(), lossfun=F.softmax_cross_entropy)
    opt = chainer.optimizers.SGD(lr=0.01)
    opt.setup(model)

    updater = training.StandardUpdater(train_iter, opt, device=gpu_id)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                         x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.run()


def load_images(IMG_DIR):
    dir_names = glob.glob('{}/*'.format(IMG_DIR))
    file_names = [glob.glob('{}/*.jpg'.format(dir)) for dir in dir_names]
    file_names = list(chain.from_iterable(file_names))
    labels = [os.path.basename(os.path.dirname(file)) for file in file_names]
    dir_names = [os.path.basename(dir) for dir in dir_names]
    labels = [dir_names.index(label) for label in labels]

    d = datasets.LabeledImageDataset(list(zip(file_names, labels)))
    
    def resize(img):
        width, height = 128, 128
        img = Image.fromarray(img.transpose(1, 2, 0))
        img = img.resize((width, height), Image.BICUBIC)
        return np.asarray(img).transpose(2, 0, 1)
    
    def transform(inputs):
        img, label = inputs
        img = img[:3, ...]
        img = resize(img.astype(np.uint8))
        img = img.astype(np.float32)
        img = img / 255
        return img, label
    
    transformed_d = datasets.TransformDataset(d, transform)
    return transformed_d
    

def main():
    hotdog_train()
    #cifar10_train()


if __name__ == '__main__':
    main()
