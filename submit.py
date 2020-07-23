#%%
import math, time
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from load_data import CreateList, CustomDataset
from models import VGG, LeNet5, BCNN

#%% Paths
dir_img_test = 'C:/Dataset/AOI/test_images/'
path_label_test = 'C:/Dataset/AOI/test.csv'
path_model = './model/bestmodel0721_vgg_pre_bn01.pkl'
save_submit = './submit/{}_submit.csv'.format(path_model.split('/')[-1].replace('.pkl', ''))

#%% Parameters
cuda = True
workers = 2
batch_size = 128
#%% Load the Model
net = VGG('aoi', True)
save = torch.load(path_model)
save['best_prec1']
net.load_state_dict(save['state_dict'])
net.eval()

# Send model into gpu memory
if cuda:
    net.cuda()
#%% Prepare the data
test_list = CreateList(dir_img_test, path_label_test, shuffle=False, train=False)

transform = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

fake_list = [i for i in range(len(test_list.img))]

test_dataset = CustomDataset(test_list.img,
                             label_list=fake_list,
                             transform=transform['test'])

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=workers,
                                          pin_memory=True)

#%% Predict test images
# Collect prediction values
test_predict = []
net.eval()
with torch.no_grad():
    for images, _ in tqdm(test_loader):
        images = images.cuda()
        
        out = net(images)  # forward
        _, pred = torch.max(out.data, 1)
        test_predict += pred.cpu().numpy().tolist()
# Check number of class of predicitons
len(set(test_predict))
# Check whether the number of predictions match test images
len(test_predict) == len(test_list.filename)
    
#%% Create submit data
df_submit = pd.DataFrame({'ID': test_list.filename,
                            'Label': test_predict})

df_submit.to_csv(save_submit,
                header=True,
                sep=',',
                encoding='utf-8',
                index=False)