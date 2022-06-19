import torch
import torch.nn as nn
from torchvision import models
from Evison import Display, show_network

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_feature_from_model(model):
    feat = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
    return feat

def load_checkpoint(filepath, extract=False):
   checkpoint = torch.load(filepath, map_location='cpu')
   model = checkpoint['model']  
   model.load_state_dict(checkpoint['model_state_dict']) 
   if isinstance(model, torch.nn.DataParallel):
       model = model.module
   if extract:
       feat  = extract_feature_from_model(model)
   return feat if extract else model

# 生成我们需要可视化的网络(可以使用自己设计的网络)
# network = models.resnet18(pretrained=True)
network = load_checkpoint(r'cipucnet_final.pth', False)

# 使用show_network这个辅助函数来看看有什么网络层(layers)
show_network(network)

# 构建visualization的对象 以及 制定可视化的网络层
visualized_layer = 'layer4.1.bn2'
display = Display(network, visualized_layer, img_size=(224, 224))  # img_size的参数指的是输入图片的大小

# 加载我们想要可视化的图片
from PIL import Image
image = Image.open('../image_20.jpg').resize((224, 224))

# 将想要可视化的图片送入display中，然后进行保存
display.save(image)
