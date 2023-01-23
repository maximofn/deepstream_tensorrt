import torch
from torch import nn
import torchvision
from torch.onnx import OperatorExportTypes
import segmentation_models_pytorch as smp

# Custom model
print("Creating model")
class segmentation(nn.Module):
    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet', number_of_channels=3, classes=1000, activation=None):
        super(segmentation, self).__init__()
        self.encoder = smp.Unet(encoder_name, encoder_weights=encoder_weights, in_channels=number_of_channels, classes=classes, activation=activation)

    def forward(self, img):
        # Input image is a numpy array with shape (H, W, C)
        # Preprocess image
        img = torch.flip(img, dims=[2])         # RGB to BGR
        # img = self.resize(img)                 # Resize
        img = img.permute(2, 0, 1)              # HWC to CHW
        img = (img / 255.0).type(torch.float32) # 0-255 to 0-1
        img = img.unsqueeze(0)                  # Add batch dimension
        # Output image is a torch tensor with shape (1, C, H, W)

        # Inference
        mask = self.encoder(img)
        # Mask is a torch tensor with shape (1, C, H, W)

        # Postprocess mask
        mask = torch.argmax(mask, axis=1)
        mask = mask.squeeze(0).type(torch.uint8)
        # Mask is a torch tensor with shape (H, W)

        # Postprocess mask
        img = img.squeeze(0).permute(1, 2, 0) * 255         # CHW to HWC
        img = torch.flip(img, dims=[2]).type(torch.uint8)   # RGB to BGR
        # Image is a torch tensor with shape (H, W, C)

        return mask, img
model = segmentation()

# Export model to ONNX
print("Exporting model to ONNX...", end="")
onnx_file = f"segmentation_custom.onnx"
dummy_input=torch.randn(1088, 1920, 3)
output = torch.onnx.export(
    model=model,
    args=dummy_input,
    f=onnx_file,
    export_params=True,
    verbose=False,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=["input"],
    output_names=["output"],
    operator_export_type=OperatorExportTypes.ONNX)
print("\tdone exporting to ONNX")