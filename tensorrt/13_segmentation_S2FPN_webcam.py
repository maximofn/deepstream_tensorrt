import torch
from torch import nn
import torchvision
import torch.backends.cudnn as cudnn
from torch2trt import torch2trt
import cv2
import numpy as np
import time
from thread import InputThread
from udp_socket import udp_socket
from video import video
import segmentation_models_pytorch as smp

import sys
sys.path.append('S2-FPN')
import S2FPN

from torch.onnx import OperatorExportTypes
import subprocess


# Create input thread
input_thread = InputThread()
input_thread.start()

# Create UDP socket
sock_frame = udp_socket('localhost', 8554, send=True)
sock_mask = udp_socket('localhost', 8555, send=True)
sock_coloridez_mask = udp_socket('localhost', 8556, send=True)

# Open webcam
WEBCAM = True
if not WEBCAM:
    from PIL import Image
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30
video_frame = video(resize=False, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, name="frame", display=False)
video_mask = video(resize=False, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, name="mask", display=False)
video_coloridez_mask = video(resize=False, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, name="coloridez_mask", display=False)
video_frame.open(device=0)

# Configuration of text on the screen
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.3
jump = 15
fontColor = (0, 0, 255)#(10,10,10)
lineThickness= 1
lineType = cv2.LINE_AA

# Time variables
T0 = time.time()
t_camera = 0
t_preprocess = 0
t_inference = 0
t_postprocess = 0
t_bucle = 0
FPS = 0

# Media variables
iteracctions = -1
t_read_frame_list = []
t_img_gpu_list = []
t_preprocess_list = []
t_model_gpu_list = []
t_inference_list = []
t_postprocess_list = []
t_bucle_list = []
FPS_list = []

# Build the model
net="SSFPN" #model name: [DSANet,SPFNet,SSFPN]
classes = 11 #number of classes
print(f"Bulding model: {net}", end=" ")
model = S2FPN.load_model(net, classes)
print("done")

# Loading checkpoint
checkpoint_path = "S2-FPN/weigths/SSFPN18.pth"
print(f"loading checkpoint '{checkpoint_path}'", end=" ")
model = S2FPN.load_checkpoints(model, checkpoint_path)
print("done")

# convert to TensorRT feeding sample data as input
# print("Converting to TensorRT")
width = 600 #1920
height = 600 #1088
# x = np.ones((1, 3, width, height), dtype=np.uint8)
# x = torch.from_numpy(x).float() # numpy to tensor
# x = x.cuda()                    # move image to GPU
# x = torch.flip(x, dims=[2])     # RGB to BGR
# x = x / 255.0                   # 0-255 to 0-1
# # x = x.permute(2, 0, 1)        # HWC to CHW
# # x = x.unsqueeze(0)            # Add batch dimension
# model_trt = torch2trt(model, [x])
# print("Converted to TensorRT")

# Export model to ONNX
print("Exporting model to ONNX...", end="")
onnx_file_with_cuda = f"S2FPN.onnx"
dummy_input=torch.randn(1, 3, width, height).cuda()
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

# Export model to TensorRT
print("Exporting model to TensorRt...", end="")
command = "/usr/src/tensorrt/bin/trtexec --onnx=S2FPN.onnx --saveEngine=S2FPN_terminal.engine  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
subprocess.call(command, shell=True)
print("\tdone exporting to TensorRt")

while True:
    t0 = time.time()
    t_start = time.time()
    if WEBCAM:
        ret, frame = video_frame.read()
    else:
        # Read image S2-FPN/images/0001TP_006690.png
        frame = cv2.imread('S2-FPN/images/0001TP_006690.png', cv2.IMREAD_COLOR)
        # img = frame.copy()
        # img = np.asarray(img, np.float32)
        ret = True
    t_camera = time.time() - t0
    if not ret:
        continue

    # Preprocess image
    t0 = time.time()
    if WEBCAM:
        img = cv2.resize(frame, (height, width))    # Resize
        # img = torch.from_numpy(img)                 # numpy to tensor
        # img = img.cuda()                            # move image to GPU
        # img = torch.flip(img, dims=[2])             # RGB to BGR
        # # img = img / 255.0                           # 0-255 to 0-1
        # img = img.type(torch.float32)             # uint8 to float32
        # img = img.permute(2, 0, 1)                # HWC to CHW
        # img = img.unsqueeze(0)                    # Add batch dimension
    else:
        img = cv2.resize(frame, (height, width))    # Resize
        # img = img[:, :, ::-1]  # change to RGB
        # img = img.transpose((2, 0, 1)).copy()  # HWC -> CHW
        # img = torch.from_numpy(img)
        # img = img[None,:,:,:]
        # # img = img.cuda()
        # with torch.no_grad():
        #     input_var = torch.autograd.Variable(img).cuda()
    t_preprocess = time.time() - t0

    # Inference
    t0 = time.time()
    # mask = model_trt(img)
    if WEBCAM:
        mask, img = S2FPN.inference(model, img_orig=img)
        # mask = mask[0].squeeze(0)
    else:
        mask, img = S2FPN.inference(model, img_orig=img)
        # mask = model(input_var)
        # torch.cuda.synchronize()
    t_inference = time.time() - t0

    # Postprocess
    t0 = time.time()
    if WEBCAM:
        pass
    #     mask = torch.argmax(mask, axis=0).type(torch.uint8)
    #     mask = mask.detach().cpu().numpy()
    else:
        pass
        # mask = mask.cpu().data[0].numpy()
        # mask = mask.transpose(1, 2, 0)
        # mask = np.asarray(np.argmax(mask, axis=2), dtype=np.uint8)
    t_postprocess = time.time() - t0

    # Bucle time
    t_bucle = time.time() - t_start

    # FPS
    FPS = 1 / t_bucle

    # Put text
    y = 30
    cv2.putText(frame, f"Modelo en GPU:", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"FPS: {FPS:.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"Image shape: {frame.shape}, image type: {type(frame)}, img dtype: {frame.dtype}, img max: {frame.max()}, img min: {frame.min()}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"Mask shape: {mask.shape}, mask type: {type(mask)}, mask dtype: {mask.dtype}, mask max: {mask.max():0.2f}, mask min: {mask.min():0.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"t camera: {t_camera*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"t preprocess: {t_preprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"t inference: {t_inference*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"t postprocess: {t_postprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
    cv2.putText(frame, f"t bucle: {t_bucle*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump

    # Media variables
    iteracctions += 1
    if iteracctions >= 0:
        t_read_frame_list.append(t_camera)
        t_preprocess_list.append(t_preprocess)
        t_inference_list.append(t_inference)
        t_postprocess_list.append(t_postprocess)
        t_bucle_list.append(t_bucle)
        FPS_list.append(FPS)
        cv2.putText(frame, f"Media: {iteracctions} iteracctions", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
        cv2.putText(frame, f"    t read frame {np.mean(t_read_frame_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
        cv2.putText(frame, f"    t preprocess {np.mean(t_preprocess_list)*1000:.2f} ms,", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
        cv2.putText(frame, f"    t inference {np.mean(t_inference_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
        cv2.putText(frame, f"    t postprocess {np.mean(t_postprocess_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
        cv2.putText(frame, f"    t bucle {np.mean(t_bucle_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump
        cv2.putText(frame, f"    FPS {np.mean(FPS_list):.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += jump

    # Mandamos el frame por el socket
    success, encoded_frame = video_frame.encode_frame(frame, format='.jpg')
    if success:
        message = encoded_frame.tobytes(order='C')
        sock_frame.send(message)

    # Mandamos la máscara por el socket
    success, encoded_mask = video_mask.encode_frame(mask)
    if success:
        message = encoded_mask.tobytes(order='C')
        sock_mask.send(message)

    # Mandamos la máscara coloreada por el socket
    coloridez_mask = S2FPN.colorize_mask(mask, img, overlay=0.3)
    success, encoded_colorized_mask = video_coloridez_mask.encode_frame(coloridez_mask, format='.jpg')
    if success:
        message = encoded_colorized_mask.tobytes(order='C')
        sock_coloridez_mask.send(message)

    # If user press type 'q' into the console in non blocking mode, exit
    if input_thread.get_data() is not None and input_thread.get_data().strip() == 'q':
        print("Se ha ha parado por el usuario")
        input_thread.clear_data()
        break


# Cerramos el socket y la cámara
sock_frame.close()
sock_mask.close()
sock_coloridez_mask.close()
video_frame.close()
video_mask.close()
video_coloridez_mask.close()