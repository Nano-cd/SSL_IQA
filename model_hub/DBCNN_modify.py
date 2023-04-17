import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39
        #        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(48,48,3,2,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(48,64,3,1,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(64,64,3,1,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(64,128,3,1,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(128,128,3,1,1),nn.ReLU(inplace=True),
        #                                      nn.Conv2d(128,128,3,2,1),nn.ReLU(inplace=True))
        self.features = nn.Sequential(nn.Conv2d(3, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                                      nn.Conv2d(48, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                                      nn.Conv2d(48, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14, 1)
        self.projection = nn.Sequential(nn.Conv2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)
        self.classifier = nn.Linear(256, self.num_class)
        weight_init(self.classifier)

    def forward(self, X):
        #        return X
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 128, 14, 14)
        X = self.pooling(X)
        assert X.size() == (N, 128, 1, 1)
        X = self.projection(X)
        X = X.view(X.size(0), -1)
        X = self.classifier(X)
        assert X.size() == (N, self.num_class)
        return X


class DBCNN(torch.nn.Module):

    def __init__(self, scnn_root, num_classes):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features1 = torchvision.models.vgg16(pretrained=True).features
        self.features1 = nn.Sequential(*list(self.features1.children())
        [:-1])
        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()

        scnn.load_state_dict(torch.load(scnn_root,map_location=torch.device('cpu')))
        self.features2 = scnn.module.features
        # Linear classifier.
        self.fc = nn.Sequential(torch.nn.Linear(512 * 128, num_classes),
                                torch.nn.Softmax(dim=1))

        # if options['fc'] == True:
        #     # Freeze all previous layers.
        #     for param in self.features1.parameters():
        #         param.requires_grad = False
        #     for param in self.features2.parameters():
        #         param.requires_grad = False
        #     # Initialize the fc layers.
        #     nn.init.kaiming_normal_(self.fc.weight.data)
        #     if self.fc.bias is not None:
        #         nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]
        X1 = self.features1(X)
        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 512
        X2 = self.features2(X)
        H2 = X2.size()[2]
        W2 = X2.size()[3]
        assert X2.size()[1] == 128

        if (H != H2) | (W != W2):
            X2 = F.upsample_bilinear(X2, (H, W))

        X1 = X1.view(N, 512, H * W)
        X2 = X2.view(N, 128, H * W)
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H * W)  # Bilinear
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512 * 128)
        activation = X
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 5)
        return X, activation


# a = torch.rand([1, 3, 244, 244])
# model = DBCNN('D:\chengdi\YIDA_3\scnn.pkl', 5)
# print(model(a))
