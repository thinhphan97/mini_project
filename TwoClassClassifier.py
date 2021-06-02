import torch
from torch import nn
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class valid_efficientnet(nn.Module):
    def __init__(self, name_model, num_class, device='cpu'):
        super(valid_efficientnet, self).__init__()
        self.name_model = name_model
        self.num_class = num_class
        self.device = device
        
    def create_model_(self):
        self.model = create_model(self.name_model, pretrained= False, num_classes=self.num_class)
        
    def load_model(self, path_model):
        
        self.model.load_state_dict(torch.load(path_model))
    def forward(self, img):
        config = resolve_data_config({}, model= self.model)
        transform = create_transform(**config)
        # img = Image.open(path_image).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        tensor = tensor.to(self.device)
        model = self.model.to(self.device)
        with torch.no_grad():
            out = model(tensor)
        probabilities = torch.sigmoid(out)
        return probabilities


def predict(model_name, weights_path, image) :
    """
    Loads pretrained weights from weights path and predict image belong to which class.
    Args:
        weights_path : Path to weights file *.pth.
        model_name (str): Model name of efficientnet.
        weights_path (str): Path to pretrained weights file on the local disk.
        img_path (str): Path to image
    """
    model = valid_efficientnet(model_name, 1)
    model.create_model_()
    model.load_model(weights_path)
    probabilities = model(image)
    probabilities = probabilities.cpu().numpy()
    if probabilities > 0.9 : 
        name = 'Normal'
    else :
        name = 'Abnormal'
        probabilities = 1 - probabilities
    return name, probabilities

# path_image ='C:/Users/Admin/Downloads/media_images_Bounding Box Debugger_Images_89_8.png'
# img = Image.open(path_image).convert('RGB')
# name, probabilities = predict(model_name='efficientnet_b2_pruned', weights_path="model_step1.pth", image = img)
# print('name = ', name, '\t', 'probabilities = ', probabilities)