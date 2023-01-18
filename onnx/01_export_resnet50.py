import torchvision
import torch
from torch.onnx import OperatorExportTypes

# Create model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).cuda()

# Export to ONNX
BATCH_SIZE=1
onnx_file = f"resnet50_BS{BATCH_SIZE}.onnx"
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224, device="cuda")


###
# torch.onnx.export(
#   model,                                                  # Model to export
#   args,                                                   # Input to the model
#   f,                                                      # File to export to
#   export_params=True,                                     # Export model parameters
#   verbose=False,                                          # Print information during export
#   training=<TrainingMode.EVAL: 0>,                        # Training mode, posible values: 
#                                                               TrainingMode.TRAINING: Model is in training mode
#                                                               TrainingMode.EVAL: Model is in eval mode
#                                                               TrainingMode.PRESERVE: Preserve training mode
#   input_names=None,                                       # Input names
#   output_names=None,                                      # Output names
#   operator_export_type=<OperatorExportTypes.ONNX: 0>,     # Operator export type, posible values:
#                                                               OperatorExportTypes.ONNX: Export model as standard ONNX operator set
#                                                               OperatorExportTypes.ONNX_ATEN: Export model as ATen operators
#                                                               OperatorExportTypes.ONNX_ATEN_FALLBACK: Export model as ATen operators, when ONNX operators are not available
#                                                               OperatorExportTypes.ONNX_FALLBACK: Export model as standard ONNX operator set, when ONNX operators are not available
#   opset_version=None,                                     # Opset version to export to
#   do_constant_folding=True,                               # Whether to execute constant folding for optimization
#   dynamic_axes=None,                                      # Dynamic axes
#   keep_initializers_as_inputs=None,                       # Whether to keep model initializers as graph inputs
#   custom_opsets=None,                                     # Custom operator sets
#   export_modules_as_functions=False)                      # Whether to export modules as functions
torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=["input"], output_names=["output"], operator_export_type=OperatorExportTypes.ONNX_EXPLICIT_BATCH)
print("Done exporting to ONNX")