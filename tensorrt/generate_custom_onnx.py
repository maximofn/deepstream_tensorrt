import torch
from torch import nn
import torchvision
# import cv2
# import numpy as np
# import time
# from thread import InputThread
# from udp_socket import udp_socket
# from video import video
from torch.onnx import OperatorExportTypes
# import tensorrt as trt

# Create custom model
print("Creating model without cuda")
class Resnet50custom(nn.Module):
    def __init__(self):
        super(Resnet50custom, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).cuda()

    def forward(self, img):
        # Input image is a numpy array with shape (H, W, C)
        # Preprocess image
        # img = img.cuda()                  # Move to GPU
        img = torch.flip(img, dims=[2])     # RGB to BGR
        img = img / 255.0                   # 0-255 to 0-1
        img = img.permute(2, 0, 1)          # HWC to CHW
        img = img.unsqueeze(0)              # Add batch dimension
        # Output image is a torch tensor with shape (1, C, H, W)

        # Inference
        probs = self.resnet50(img)
        return probs

        # Postprocess
        probs = torch.nn.functional.softmax(probs, dim=1)
        probs = probs.squeeze(0)
        prob, idx = torch.max(probs, dim=0)
        # # prob = prob.item()
        # # idx = int(idx.item())
        return prob, idx
model = Resnet50custom()

# Export model to ONNX
print("Exporting model to ONNX...", end="")
onnx_file_without_cuda = f"resnet50custom.onnx"
dummy_input=torch.randn(1080, 1920, 3)
output = torch.onnx.export(
    model=model,
    args=dummy_input,
    f=onnx_file_without_cuda,
    export_params=True,
    verbose=False,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=["input"],
    output_names=["output"],
    operator_export_type=OperatorExportTypes.ONNX)
print("\tdone exporting to ONNX")

# Create custom model
print("Creating model with cuda")
class Resnet50custom(nn.Module):
    def __init__(self):
        super(Resnet50custom, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).cuda()

    def forward(self, img):
        # Preprocess image
        img = img.cuda()                    # Move to GPU
        img = torch.flip(img, dims=[2])     # RGB to BGR
        img = img / 255.0                   # 0-255 to 0-1
        img = img.permute(2, 0, 1)          # HWC to CHW
        img = img.unsqueeze(0)              # Add batch dimension

        # Inference
        probs = self.resnet50(img)
        return probs

        # Postprocess
        probs = torch.nn.functional.softmax(probs, dim=1)
        probs = probs.squeeze(0)
        prob, idx = torch.max(probs, dim=0)
        # prob = prob.item()
        # idx = int(idx.item())
        return prob, idx
model = Resnet50custom()

# Export model to ONNX
print("Exporting model to ONNX...", end="")
onnx_file_with_cuda = f"resnet50customCuda.onnx"
dummy_input=torch.randn(1080, 1920, 3)
output = torch.onnx.export(
    model=model,
    args=dummy_input,
    f=onnx_file_with_cuda,
    export_params=True,
    verbose=False,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=["input"],
    output_names=["output"],
    operator_export_type=OperatorExportTypes.ONNX)
print("\tdone exporting to ONNX")
