# loading dependencies

import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet50

# speed up trick
torch.backends.cudnn.benchmark = True

# create GPU connection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading model weights and seeting it in inference mode

model = resnet50(weights = None) # init. an empty model
model.fc = nn.Linear(2048, 45) # replacing the classification head

state_dict = torch.load('95.6_val_acc.pth', map_location = device)
model.load_state_dict(state_dict)

model.to(device) # moving comps. to GPU
model.eval()

# prepare data for inference

inf_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]   
    )
])

def infer_from_resnet50(img):
    img_tensor = inf_transforms(img).unsqueeze(0)

    img_tensor = img_tensor.to(device)
    # inference

    with torch.no_grad():
        output = model(img_tensor)

    # obtain class index

    pred_class = torch.argmax(output, dim = 1).item()

    return pred_class