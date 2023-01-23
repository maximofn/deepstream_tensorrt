import torch
import torchvision
import cv2
import numpy as np
import time
from thread import InputThread
from udp_socket import udp_socket
from video import video
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# Create input thread
input_thread = InputThread()
input_thread.start()

# Create UDP socket
sock = udp_socket('localhost', 8554, send=True)

# Open webcam
height = 1000 # 1080
video = video(resize=False, width=1920, height=height, fps=30, name="frame", display=False)
video.open(device=0)

# Download model
print("Creating model...")
encoder_name = 'resnet50'
encoder_weights = 'imagenet'
number_of_channels = 3
classes=1000
activation=None
model = smp.Unet(
    encoder_name, 
    encoder_weights=encoder_weights, 
    in_channels=number_of_channels, 
    classes=classes, activation=activation
)

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

while True:
    t0 = time.time()
    t_start = time.time()
    ret, frame = video.read()
    t_camera = time.time() - t0
    if not ret:
        continue

    # Preprocess image
    t0 = time.time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1920, 1088))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    t_preprocess = time.time() - t0

    # Send model to GPU
    t0 = time.time()
    model = model.cuda()
    t_model_gpu = time.time() - t0

    # Send image to GPU
    t0 = time.time()
    img = img.cuda()
    t_img_gpu = time.time() - t0

    # Inference
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        mask_pred = model(img)
        end = time.time()
    t_inference = time.time() - t0

    # Postprocess
    t0 = time.time()
    mask_pred = torch.argmax(mask_pred, axis=1)
    mask_pred = mask_pred.squeeze(0).type(torch.float32).detach().cpu().numpy()

    frame = img.squeeze(0).permute(1, 2, 0) * 255
    frame = torch.flip(frame, dims=[2])     # RGB to BGR
    frame = frame.type(torch.uint8).detach().cpu().numpy()
    # frame = cv2.resize(frame, (1920, 1080))
    t_postprocess = time.time() - t0

    # Bucle time
    t_bucle = time.time() - t_start

    # FPS
    FPS = 1 / t_bucle

    # Put text
    y = 30
    cv2.putText(frame, f"Modelo en GPU:", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"FPS: {FPS:.2f}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"Image shape: {img.shape}, img dtype: {img.dtype}, img max: {img.max()}, img min: {img.min()}", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t camera: {t_camera*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t preprocess: {t_preprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t model to gpu: {t_model_gpu*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t image to gpu: {t_img_gpu*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t inference: {t_inference*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t postprocess: {t_postprocess*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
    cv2.putText(frame, f"t bucle: {t_bucle*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30

    # Media variables
    iteracctions += 1
    if iteracctions >= 0:
        t_read_frame_list.append(t_camera)
        t_img_gpu_list.append(t_img_gpu)
        t_preprocess_list.append(t_preprocess)
        t_model_gpu_list.append(t_model_gpu)
        t_inference_list.append(t_inference)
        t_postprocess_list.append(t_postprocess)
        t_bucle_list.append(t_bucle)
        FPS_list.append(FPS)
        cv2.putText(frame, f"Media: {iteracctions} iteracctions", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t read frame {np.mean(t_read_frame_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t img to gpu {np.mean(t_img_gpu_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t preprocess {np.mean(t_preprocess_list)*1000:.2f} ms,", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
        cv2.putText(frame, f"    t model to gpu {np.mean(t_model_gpu_list)*1000:.2f} ms", (10, y), font, fontScale, fontColor, lineThickness, lineType); y += 30
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