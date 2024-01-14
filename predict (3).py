import argparse
import torch
from torchvision import models
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import json

def load_checkpoint(filepath):
    if not os.path.isfile(filepath):
        print(f"Checkpoint file {filepath} does not exist.")
        exit()

    checkpoint = torch.load(filepath)
    arch = checkpoint.get('architecture', 'vgg16')  # Default to 'vgg16' if not found

    if arch == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier = checkpoint.get('classifier')
    elif arch == 'resnet18':
        model = model.resnet18(pretrained=False)
        model.fc = checkpoint.get('classifier')
    else:
        print(f"Architecture {arch} not supported. Please use 'vgg16'.")
        exit()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    # Image processing steps remain the same
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((10000, 256) if img.size[0] > img.size[1] else (256, 10000))
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    img = img.crop((left_margin, bottom_margin, left_margin+224, bottom_margin+224))
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)

def predict(image_path, model, device, topk=5):
    img = process_image(image_path).unsqueeze(0)
    model.to(device)
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(img)
    ps = F.softmax(logits, dim=1)
    topk_probs, topk_indices = ps.topk(topk)
    topk_probs = topk_probs.cpu().numpy()[0]
    topk_indices = topk_indices.cpu().numpy()[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    topk_classes = [idx_to_class[i] for i in topk_indices]
    return topk_probs, topk_classes

def main():
    parser = argparse.ArgumentParser(description='Predict top K classes of an image using VGG16.')
    parser.add_argument('input_image', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint to use when predicting')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, default=None, help='JSON file for mapping class to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("GPU not available. Using CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    model = load_checkpoint(args.checkpoint)

    if args.category_names:
        if not os.path.isfile(args.category_names):
            print(f"JSON file {args.category_names} does not exist.")
            exit()
        else:
            with open(args.category_names, 'r') as f:
                class_to_name = json.load(f, strict=False) 
                
    topk_probs, topk_classes = predict(args.input_image, model, device, args.top_k)

    if args.category_names:
        topk_names = [class_to_name[i] for i in topk_classes]
        print('Top K class names: ', topk_names)

    print('Top K classes: ', topk_classes)
    print('Top K probabilities: ', topk_probs)

if __name__ == '__main__':
    main()
