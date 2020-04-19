# -*- coding: utf-8 -*-
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image 
from torchvision import models, transforms
import torch.nn as nn
import shutil
import glob
data_dir = './img'
features_dir = './features'
shutil.copytree(data_dir, os.path.join(features_dir, data_dir[2:]))
 
 
def extractor(img_path, saved_path, net, use_gpu):
  transform = transforms.Compose([
      #transforms.Scale(256),
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor()  ]
  )
  
  img = Image.open(img_path)
  img = transform(img)
  
 
 
  x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
  if use_gpu:
    x = x.cuda()
    net = net.cuda()
  y = net(x).cpu()
  y = y.data.numpy()
  np.savetxt(saved_path, y, delimiter=',')
  #print(saved_path)

  
if __name__ == '__main__':
  extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    
  files_list = []
  sub_dirs = [x[0] for x in os.walk(data_dir) ]
  sub_dirs = sub_dirs[1:]
  for sub_dir in sub_dirs:
    for extention in extensions:
      file_glob = os.path.join(sub_dir, '*.' + extention)
      files_list.extend(glob.glob(file_glob))
    
  resnet50_feature_extractor = models.resnet50(pretrained = True)
  resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
  torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)
  for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False  
    
  use_gpu = torch.cuda.is_available()
 
  for x_path in files_list:
    print(x_path)
    fx_path = os.path.join(features_dir, x_path[2:] + '.txt')
    extractor(x_path, fx_path, resnet50_feature_extractor, use_gpu)
    #Remove IMG in feature may have no use
    fx_path = os.path.join(features_dir, x_path[2:])
    os.remove(fx_path)
