import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_saliency(model, input, target_class):
    input.requires_grad_()
    output = model(input)
    score = output[0, target_class]
    model.zero_grad()
    score.backward()

    saliency,_ = torch.max(input.grad.data.abs(), dim=1)
    saliency = saliency[0].cpu().numpy()
    return saliency

def saliency_map(model, image_path, target_class):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    ori_image = transforms.Resize((224, 224))(image)
    ori_image = np.array(ori_image)
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2GRAY)

    input_tensor = preprocess(image).unsqueeze(0)
    
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    saliency = get_saliency(model, input_tensor, target_class)
    
    saliency = np.maximum(saliency, 0)
    saliency = saliency / np.max(saliency)
    saliency = (saliency * 255).astype(np.uint8)
    
    return saliency, ori_image

if __name__ == "__main__":
    model = models.vgg16(pretrained=True).to(device)
    class_id = 'n01443537'
    image_folder = f'data/tiny-imagenet-200/train/{class_id}/images'
    # target class is the index of class_id in the train folder
    target_class = list(os.listdir("data/tiny-imagenet-200/train")).index(class_id)
    image_paths = os.listdir(image_folder)

    output_folder = 'code/output'
    for image in image_paths:
        image_path = os.path.join(image_folder, image)
        saliency, ori_image = saliency_map(model, image_path, target_class)

        # concatenate the saliency map and original image
        combined = np.hstack((ori_image, saliency))
        output_path = os.path.join(output_folder, 'saliency_' + image)
        cv2.imwrite(output_path, combined)
        print(f"Saliency map for {image} is Done")




