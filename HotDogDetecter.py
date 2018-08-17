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


def hotdog_train():
    #train = load_images('./data/train')
    #test = load_images('./data/test')
    dataset = load_images('./data/sushi_hotdog')
    train, test = datasets.split_dataset_random(dataset, int(len(dataset) * 0.8))

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
    serializers.save_npz('sushi_hotdog.model', model)


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
    class_labels = 2
    gpu_id = -1
    chainer.config.train = False
    model = L.Classifier(models.VGG.VGG(class_labels))
    serializers.load_npz('my_hotdog_VGG.model', model)
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    def load_image():
        img = cv2.imread('./test.jpg')
        return img

    def transform(img_cv):
        img = img_cv[:,:,::-1].copy()
        img_resized = cv2.resize(img, (224, 224))
        X = img_resized.transpose(2, 0, 1)
        x = X.astype(np.float32)
        x = x/255
        return x
    
    #img_cv = load_image()
    img_cv = capture_camera()
    x = transform(img_cv)
    result = model.predictor(Variable(np.array([x])))
    print(result)


def capture_camera(mirror=True, size=None):
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if mirror == True:
            frame = frame[:,::-1]
        cv2.imshow('webcam', frame)
        key = cv2.waitKey(1)
        
        if key == ord('c'):
            return frame
            break
    cap.release()
    cv2.destroyAllWindows()        


def main():
    #hotdog_train()
    inference()
    #capture_camera()


if __name__ == '__main__':
    main()
