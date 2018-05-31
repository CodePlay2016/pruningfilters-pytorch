import numpy as np
import torch
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
from finetune import ModifiedVGG16Model
from PIL import Image

model = torch.load("model").cuda()
print("model load success: ",model)

filelist = glob.glob("/home/hfq/model_compress/prune/pytorch-pruning/test/*")
filelist.sort()
filelist = filelist[:1000]

filepath = filelist[0]
img = Image.open(filepath)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize(256),
                                transforms.ToTensor(),
                                normalize])
img = transform(img)
print(np.array(img))
