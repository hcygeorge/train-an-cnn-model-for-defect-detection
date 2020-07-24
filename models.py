import torch
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F

#%% Define model structure
# VGG 
class VGG(nn.Module):
    '''VGG model structure but with only one fully connnected layer.
    
    Args:
        dataset (str): Name of the dataset, which indicates the num_classes and feature_channels of model
        pretrained (bool): Whether to build and pretrained VGG16 model.
        cfg (list): A list of integers and string 'M' represent the structure of model.
        batch_norm (bool): Whether to insert batch normalization layer between convolutional layers
        
    Attributes:
        feature (torch.nn.Sequential): The feature extractor of model,
            usually contains several convolutional layers and pooling layers
        classifier (torch.nn.Sequential): The classifier which output a numerical vector as prediction.
        
    '''
    def __init__(self, dataset=None, pretrained=True, cfg=None, batch_norm=True):
        super(VGG, self).__init__()
        # Set number of model output and feature channel according to your dataset
        if dataset == 'aoi':
            self.in_channels = 3
            num_classes = 6
            feature_channels = 512*7*7
        

        # Define model structure according to cfg or using pretrained model
        if pretrained:
            print('Use pretrained VGG feature extractor')
            if batch_norm:
                self.feature = torchvision.models.vgg16_bn(pretrained=True).features
                self.feature = nn.Sequential(*list(self.feature.children())[:-1])  # Remove pool5
            else:
                self.feature = torchvision.models.vgg16(pretrained=True).features
                self.feature = nn.Sequential(*list(self.feature.children())[:-1])  # Remove pool5

            self.classifier = nn.Linear(feature_channels, num_classes)
        else:
            if cfg is None:
                cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
            self.feature = self.make_layers(cfg, batch_norm=batch_norm)
            self.classifier = nn.Linear(feature_channels, num_classes)
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                self.in_channels = v
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        
        return y

    def _initialize_weights(self):
        print('Initial model parameters...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()    
# LeNet5 model
class LeNet5(nn.Module):
    def __init__(self, dataset):
        super(LeNet5, self).__init__()
        if dataset == 'aoi':
            in_channels = 1
            num_classes = 6
            self.feature_channels = 120*50*50
        # C1 Convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # S2 Max-pooling
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        # C3 Convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # S4 Max-pooling
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        # C5 Convolution
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=True)
        # F6 Fully connected layer
        self.fc1 = nn.Linear(self.feature_channels, 84)
        # Output layer
        self.fc2 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.max_pool_1(x))
        x = self.conv2(x)
        x = F.relu(self.max_pool_2(x))
        x = self.conv3(x)
        x = x.view(-1, self.feature_channels)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#%%
if __name__ == '__main__':
    # Check model output size
    net = LeNet5(dataset='aoi')
    x = torch.FloatTensor(1, 1, 224, 224).cuda()
    nn.AvgPool2d(2)(net.feature(x)).shape  # 512*7*7

    
