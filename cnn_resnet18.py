import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EmbedderNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()
        self.backbone = base
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def load_embeder_from_pt(path):
    model = EmbedderNet(emb_dim=128)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def extract_feature_from_bbox(image, bbox, model):
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]
    # Return None if crop is invalid
    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None
    img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_pil).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().cpu().numpy()
