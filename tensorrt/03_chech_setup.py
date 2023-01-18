# Check nvcc version
import os
print("\nnvcc")
os.system('nvcc --version')

# Check nvidia-smi
print("\nnvidia-smi")
os.system('nvidia-smi')

# Check TensorRT version
print("\nTensorRT")
os.system('dpkg -l | grep TensorRT')

# Check pycuda
print("\npycuda")
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
print(f"Pycuda version:  {pycuda.VERSION_TEXT}")

# Check TensorRT Python API
print("\nTensorRT Python API")
import tensorrt as trt
print(f"TensorRT version:  {trt.__version__}")

# Check onnx
print("\nonnx")
import onnx
print(f"onnx version:  {onnx.__version__}, opset: {onnx.defs.onnx_opset_version()}")

# Check onnx-tensorrt
# print("\nonnx-tensorrt")
# import onnx_tensorrt.backend as backend
# print(f"onnx-tensorrt version:  {backend.__version__}")

# Check opencv
print("\nopencv")
import cv2
print(f"opencv version:  {cv2.__version__}")

# Check matplotlib
# print("\nmatplotlib")
# from matplotlib import pyplot as plt
# print(f"matplotlib version:  {plt.__version__}")

# Check numpy
print("\nnumpy")
import numpy as np
print(f"numpy version:  {np.__version__}")

# Check pytorch
print("\npytorch")
import torch
print(f"pytorch version:  {torch.__version__}")

