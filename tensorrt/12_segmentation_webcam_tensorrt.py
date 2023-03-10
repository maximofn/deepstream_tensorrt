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
sock_frame = udp_socket('localhost', 8554, send=True)
sock_mask = udp_socket('localhost', 8555, send=True)

# Open webcam
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30
video_frame = video(resize=False, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, name="frame", display=False)
video_mask = video(resize=False, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT, fps=CAPTURE_FPS, name="mask", display=False)
video_frame.open(device=0)

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
t_bucle_read_frame = 0
t_bucle_no_read_frame = 0
FPS = 0

# Media variables
iteracctions = -1
t_read_frame_list = []
t_img_gpu_list = []
t_preprocess_list = []
t_model_gpu_list = []
t_inference_list = []
t_postprocess_list = []
t_bucle_read_frame_list = []
t_bucle_no_read_frame_list = []
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
        # img = torch.from_numpy(img)             # Numpy to torch
        # img = img.cuda()                        # CPU to GPU
        # img = torch.flip(img, dims=[2])         # RGB to BGR
        # img = img / 255.0                       # 0-255 to 0-1
        img = img.permute(2, 0, 1)              # HWC to CHW
        img = img.unsqueeze(0)                  # Add batch dimension
        # Output image is a torch tensor with shape (1, C, H, W)

        # Inference
        mask = self.encoder(img)
        # Mask is a torch tensor with shape (1, C, H, W)

        # Postprocess mask
        mask = mask.squeeze(0)                # Remove batch dimension
        # mask = torch.argmax(mask, axis=0)   # CHW to HW
        # Mask is a torch tensor with shape (H, W)

        return mask
model = segmentation().eval().cuda()

# convert to TensorRT feeding sample data as input
print("Converting to TensorRT")
mult = 21
width = 32*mult # 1920
height = 32*mult # 1088
x = np.ones((width, height, 3), dtype=np.uint8)
x = torch.from_numpy(x).float() # numpy to tensor
x = x.cuda()                    # move image to GPU
x = torch.flip(x, dims=[2])     # RGB to BGR
x = x / 255.0                   # 0-255 to 0-1
# x = x.permute(2, 0, 1)        # HWC to CHW
# x = x.unsqueeze(0)            # Add batch dimension
model_trt = torch2trt(model, [x])
print("Converted to TensorRT")

while True:
    t0 = time.time()
    t_start = time.time()
    ret, frame = video_frame.read()
    t_camera = time.time() - t0
    if not ret:
        continue

    # Preprocess image
    t0 = time.time()
    img = cv2.resize(frame, (height, width))    # Resize
    img = torch.from_numpy(img)                 # numpy to tensor
    img = img.cuda()                            # move image to GPU
    img = torch.flip(img, dims=[2])             # RGB to BGR
    img = img / 255.0                           # 0-255 to 0-1
    # img = img.permute(2, 0, 1)                # HWC to CHW
    # img = img.unsqueeze(0)                    # Add batch dimension
    t_preprocess = time.time() - t0

    # Inference
    t0 = time.time()
    mask = model_trt(img)
    t_inference = time.time() - t0

    # Postprocess
    t0 = time.time()
    mask = torch.argmax(mask, axis=0).type(torch.uint8)
    mask = mask.detach().cpu().numpy()
    t_postprocess = time.time() - t0

    # Bucle time
    t_bucle_no_read_frame = t_preprocess + t_inference + t_postprocess
    t_bucle_read_frame = time.time() - t_start

    # FPS
    FPS = 1 / t_bucle_no_read_frame

    # Put text
    y = 30
    cv2.putText(frame, f"Modelo en GPU:", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"FPS: {FPS:.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"Image shape: {frame.shape}, image type: {type(frame)}, img dtype: {frame.dtype}, img max: {frame.max()}, img min: {frame.min()}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"Mask shape: {mask.shape}, mask type: {type(mask)}, mask dtype: {mask.dtype}, mask max: {mask.max():0.2f}, mask min: {mask.min():0.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t camera: {t_camera*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t preprocess: {t_preprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t inference: {t_inference*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t postprocess: {t_postprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t bucle (read frame): {t_bucle_read_frame*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t bucle (no read frame): {t_bucle_no_read_frame*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30

    # Media variables
    iteracctions += 1
    if iteracctions >= 0:
        t_read_frame_list.append(t_camera)
        t_preprocess_list.append(t_preprocess)
        t_inference_list.append(t_inference)
        t_postprocess_list.append(t_postprocess)
        t_bucle_read_frame_list.append(t_bucle_read_frame)
        t_bucle_no_read_frame_list.append(t_bucle_no_read_frame)
        FPS_list.append(FPS)
        cv2.putText(frame, f"Media: {iteracctions} iteracctions", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t read frame {np.mean(t_read_frame_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t preprocess {np.mean(t_preprocess_list)*1000:.2f} ms,", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t inference {np.mean(t_inference_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t postprocess {np.mean(t_postprocess_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t bucle (read frame) {np.mean(t_bucle_read_frame_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t bucle (no read frame) {np.mean(t_bucle_no_read_frame_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    FPS {np.mean(FPS_list):.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30

    # Mandamos el frame por el socket
    success, encoded_frame = video_frame.encode_frame(frame)
    if success:
        message = encoded_frame.tobytes(order='C')
        sock_frame.send(message)

    # Mandamos la m??scara por el socket
    success, encoded_mask = video_mask.encode_frame(mask)
    if success:
        message = encoded_mask.tobytes(order='C')
        sock_mask.send(message)

    # If user press type 'q' into the console in non blocking mode, exit
    if input_thread.get_data() is not None and input_thread.get_data().strip() == 'q':
        print("Se ha ha parado por el usuario")
        input_thread.clear_data()
        break


# Cerramos el socket y la c??mara
sock_frame.close()
sock_mask.close()
video_frame.close()
video_mask.close()