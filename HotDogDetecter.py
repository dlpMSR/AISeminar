import chainer
from chainer import datasets
from chainer import functions as F 
from chainer import links as L 
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import Variable

from PIL import Image
import numpy as np

import os
import glob
from itertools import chain

import cv2

import models.VGG
import models.Pt1_Normal
import models.Pt2_Normalize
import models.VGG_FORNO


def hotdog_train():
    #train = load_images('./data/train')
    #test = load_images('./data/test')
    dataset = load_images('./data/')
    train, test = datasets.split_dataset_random(dataset, int(len(dataset) * 0.7))

    batchsize = 16
    epoch = 500
    gpu_id = 0
    class_labels = 2

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    #model = L.Classifier(models.Pt1_Normal.Normal(), lossfun=F.softmax_cross_entropy)
    model = L.Classifier(models.VGG.VGG(class_labels))
    #opt = chainer.optimizers.SGD(lr=0.05)
    opt = chainer.optimizers.Adam()
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

    model.to_cpu()
    serializers.save_npz('my_hotdog_VGG.model', model)


def load_images(IMG_DIR):
    dir_names = glob.glob('{}/*'.format(IMG_DIR))
    file_names = [glob.glob('{}/*.jpg'.format(dir)) for dir in dir_names]
    file_names = list(chain.from_iterable(file_names))
    labels = [os.path.basename(os.path.dirname(file)) for file in file_names]
    dir_names = [os.path.basename(dir) for dir in dir_names]
    labels = [dir_names.index(label) for label in labels]

    d = datasets.LabeledImageDataset(list(zip(file_names, labels)))
    
    def resize(img):
        width, height = 224, 224
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
    

def inference():
    model_demo = L.Classifier(models.Pt1_Normal.Normal())
    serializers.load_npz('my_hotdog.model', model_demo)

    gpu_id = -1
    if gpu_id >= 0:
        model_demo.to_gpu(gpu_id)
    
    img_cv =cv2.imread('./test3.jpg')
    img = img_cv[:,:,::-1].copy()
    img_224 = cv2.resize(img, (128, 128))

    X = img_224.transpose(2, 0, 1)
    x = X.astype(np.float32)
    x = x/255

    result = model_demo.predictor(Variable(np.array([x])))
    print(result)
    

def main():
    #hotdog_train()
    inference()


if __name__ == '__main__':
    main()
