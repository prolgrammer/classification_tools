import torch
from PIL import Image
from torchvision import transforms
import logging


def predict_image(image_path, model, classes, device):
    logging.basicConfig(filename='logs/inference.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_name = classes[predicted.item()]

    logging.info(f'Image: {image_path}, Predicted: {class_name}, Confidence: {confidence.item():.4f}')
    return class_name, confidence.item()