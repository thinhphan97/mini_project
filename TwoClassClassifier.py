import torch
from torch import nn
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import torch
from torchvision import transforms
import base64
from io import BytesIO


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tfms = transforms.Compose([
    transforms.Resize((260,260)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

class valid_efficientnet(nn.Module):
    def __init__(self, name_model, num_class, device='cpu'):
        super(valid_efficientnet, self).__init__()
        self.name_model = name_model
        self.num_class = num_class
        self.device = device
        
    def create_model_(self):
        self.model = create_model(self.name_model, pretrained= False, num_classes=self.num_class)
        
    def load_model(self, path_model):
        
        self.model.load_state_dict(torch.load(path_model, map_location='cpu'))
    def forward(self, img):
        # config = resolve_data_config({}, model= self.model)
        transform = tfms
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

# path_image ='/home/thuyngan/Downloads/images/val/0a8d69d1c45bece901929db269a5e6cf.png'
# img = Image.open(path_image).convert('RGB')
# name, probabilities = predict(model_name='efficientnet_b2_pruned', weights_path="model_step1.pth", image = img)
# print('name = ', name, '\t', 'probabilities = ', probabilities)
def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format='JPEG')
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str