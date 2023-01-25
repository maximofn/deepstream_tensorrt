import torch
from torch import nn
import torchvision
from torch2trt import torch2trt
import cv2
import numpy as np
import time
from thread import InputThread
from udp_socket import udp_socket
from video import video
import segmentation_models_pytorch as smp
# import matplotlib.pyplot as plt

# Create input thread
input_thread = InputThread()
input_thread.start()

# Create UDP socket
sock = udp_socket('localhost', 8554, send=True)

# Open webcam
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30
video = video(resize=False, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, name="frame", display=False)
video.open(device=0)

# Configuration of text on the screen
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
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

# Custom model
print("Creating model")
class segmentation(nn.Module):
    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet', number_of_channels=3, classes=1000, activation=None):
        super(segmentation, self).__init__()
        self.encoder = smp.Unet(encoder_name, encoder_weights=encoder_weights, in_channels=number_of_channels, classes=classes, activation=activation)

    def forward(self, img):
        # Input image is a numpy array with shape (H, W, C)
        # Preprocess image
        # img = torch.from_numpy(img)             # TODO: Quitar para tensorrt
        # img = torch.flip(img, dims=[2])         # RGB to BGR
        # # img = self.resize(img)                 # Resize
        # img = img.permute(2, 0, 1)              # HWC to CHW
        # img = (img / 255.0).type(torch.float32) # 0-255 to 0-1
        # img = img.unsqueeze(0)                  # Add batch dimension
        # img = img.cuda()                        # TODO: Quitar para tensorrt
        # Output image is a torch tensor with shape (1, C, H, W)

        # Inference
        mask = self.encoder(img)
        # Mask is a torch tensor with shape (1, C, H, W)

        # Postprocess mask
        # mask = torch.argmax(mask, axis=1)
        # mask = mask.squeeze(0).type(torch.uint8)
        # Mask is a torch tensor with shape (H, W)

        # Postprocess mask
        # img = img.squeeze(0).permute(1, 2, 0) * 255         # CHW to HWC
        # img = torch.flip(img, dims=[2]).type(torch.uint8)   # RGB to BGR
        # Image is a torch tensor with shape (H, W, C)

        return mask#, img
model = segmentation().eval().cuda()

# convert to TensorRT feeding sample data as input
print("Converting to TensorRT")
width = 224 # 1920
height = 224 # 1088
x = np.ones((1, 3, width, height), dtype=np.uint8)
x = torch.from_numpy(x).float()
x = x.cuda()
x = torch.flip(x, dims=[2])
x = x / 255.0                   # 0-255 to 0-1
# x = x.permute(2, 0, 1)          # HWC to CHW
# x = x.unsqueeze(0)              # Add batch dimension
model_trt = torch2trt(model, [x])
print("Converted to TensorRT")

while True:
    t0 = time.time()
    t_start = time.time()
    ret, frame = video.read()
    t_camera = time.time() - t0
    if not ret:
        continue

    # Preprocess image
    t0 = time.time()
    img = cv2.resize(frame, (height, width))
    img = torch.from_numpy(img)
    img = img.cuda()
    img = torch.flip(img, dims=[2])
    img = img / 255.0    
    t_preprocess = time.time() - t0

    # Inference
    t0 = time.time()
    mask = model_trt(img)
    print(mask.shape)
    t_inference = time.time() - t0

    # Postprocess
    t0 = time.time()
    # mask = mask.detach().cpu().numpy()
    # frame = frame.detach().cpu().numpy()
    t_postprocess = time.time() - t0

    # Bucle time
    t_bucle = time.time() - t_start

    # FPS
    FPS = 1 / t_bucle

    # Put text
    y = 30
    cv2.putText(frame, f"Modelo en GPU:", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"FPS: {FPS:.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"Image shape: {frame.shape}, img dtype: {frame.dtype}, img max: {frame.max()}, img min: {frame.min()}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"Mask shape: {mask.shape}, mask dtype: {mask.dtype}, mask max: {mask.max()}, mask min: {mask.min()}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t camera: {t_camera*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t preprocess: {t_preprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t inference: {t_inference*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t postprocess: {t_postprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t bucle: {t_bucle*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30

    # Media variables
    iteracctions += 1
    if iteracctions >= 0:
        t_read_frame_list.append(t_camera)
        t_preprocess_list.append(t_preprocess)
        t_inference_list.append(t_inference)
        t_postprocess_list.append(t_postprocess)
        t_bucle_list.append(t_bucle)
        FPS_list.append(FPS)
        cv2.putText(frame, f"Media: {iteracctions} iteracctions", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t read frame {np.mean(t_read_frame_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t preprocess {np.mean(t_preprocess_list)*1000:.2f} ms,", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t inference {np.mean(t_inference_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t postprocess {np.mean(t_postprocess_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t bucle {np.mean(t_bucle_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    FPS {np.mean(FPS_list):.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30

    # Mandamos el frame por el socket
    success, encoded_frame = video.encode_frame(frame)
    if success:
        message = encoded_frame.tobytes(order='C')
        sock.send(message)

    # Mandamos la máscara por el socket
    # success, encoded_frame = video.encode_frame(mask_pred)
    # if success:
    #     message = encoded_frame.tobytes(order='C')
    #     sock.send(message)

    # If user press type 'q' into the console in non blocking mode, exit
    if input_thread.get_data() is not None and input_thread.get_data().strip() == 'q':
        print("Se ha ha parado por el usuario")
        input_thread.clear_data()
        break


# Cerramos el socket y la cámara
sock.close()
video.close()