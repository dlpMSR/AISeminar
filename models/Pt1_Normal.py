import chainer
import chainer.functions as F
import chainer.links as L

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
