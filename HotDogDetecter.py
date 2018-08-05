import chainer
from chainer import datasets
from chainer import functions as F 
from chainer import links as L 
from chainer import optimizers
from chainer import training
from chainer.training import extensions


class HotDogDetecter(chainer.Chain):
    def __init__(self):
        super(HotDogDetecter, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_channels=8, ksize=3)
            self.conv2 = L.Convolution2D(None, out_channels=16, ksize=3)
            self.fc1 = L.Linear(None, 256)
            self.fc2 = L.Linear(256, 10)
        

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

    updater = training.StandardUpdater(train_iter, opt, device=-1)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                           'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    #trainer.extend(extensions.Snapshot((10, 'epoch')))

    trainer.run()    


def main():
    mnist_train()


if __name__ == '__main__':
    main()