import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers

class HotDogDetecter(chainer.chain):
    def __init__(self):
        super(HotDogDetecter, self).__init__(
            conv1 = L.Convolution2D(None, out_channels=8, ksize=3),
            conv2 = L.Convolution2D(None, out_channels=16, ksize=3),
            fc1 = L.Linear(None, 256)
            fc2 = L.Linear(256, 10)
        )

        def __call__(self, x):
            h = self.conv1(x)
            h = self.conv2(h)
            h = F.max_pooling_2d(h, ksize=3, stride=1)
            h = F.sigmoid(fc1(h))
            h = F.softmax(fc2(h))   
            return h


def mnist_train()):
    train_, test_ = chainer.datasets.get_mnist()
    train_data, train_label = train_._datasets
    test_data,  test_label  = test_._datasets
    train_data = train_data.reshape((len(train_data)), 1, 28, 28)
    test_data  = test_data.reshape((len(test_data)), 1, 28, 28)

    hdd_cnn = HotDogDetecter()
    hdd_cls = L.Classifier(hdd_cnn, lossfun=F.softmax_cross_entropy)

    cuda.get_device(-1).use()
    hdd_cls.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(hdd_cls)

    batchsize = 1
    train_datasize = 60000
    valid_datasize = 10000

    for epoch in range(20):
    print('epoch', epoch)
    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)


    



if __name__ == '__main__':
    main()