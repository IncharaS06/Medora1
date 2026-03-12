import base64
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def generate_gradcam(model, image_path: str, target_class: int | None = None):

    model.eval()
    device = next(model.parameters()).device

    # load image
    pil_image = Image.open(image_path).convert("RGB")
    original = cv2.imread(image_path)

    if original is None:
        raise Exception("Image could not be loaded")

    h, w = original.shape[:2]

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # EfficientNet last conv layer
    target_layer = model.m.features[-1]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax()

    model.zero_grad()
    output[0, target_class].backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()

    weights = torch.mean(grads, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = cam.cpu().numpy()
    cam = np.maximum(cam, 0)

    if cam.max() != 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (w,h))

    heatmap = np.uint8(255*cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    success, buffer = cv2.imencode(".jpg", overlay)

    fh.remove()
    bh.remove()

    if not success:
        raise Exception("GradCAM encoding failed")

    return base64.b64encode(buffer).decode()