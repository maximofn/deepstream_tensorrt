import torch
import torch.nn.functional as F
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

# Create model
number_of_classes = 1000
number_of_channels = 3
def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv2d(ci, co, 3, padding=1),
        torch.nn.BatchNorm2d(co),
        torch.nn.ReLU(inplace=True)
    )

def encoder_conv(ci, co):
  return torch.nn.Sequential(
        torch.nn.MaxPool2d(2),
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
    )

class deconv(torch.nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)
    
    # recibe la salida de la capa anetrior y la salida de la etapa
    # correspondiente del encoder
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        # concatenamos los tensores
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class out_conv(torch.nn.Module):
    def __init__(self, ci, co, coo):
        super(out_conv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv = conv3x3_bn(ci, co)
        self.final = torch.nn.Conv2d(co, coo, 1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))
        x = self.conv(x1)
        x = self.final(x)
        return x

class UNetResnet(torch.nn.Module):
    def __init__(self, n_classes=number_of_classes, in_ch=number_of_channels):
        super().__init__()

        self.encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)           
        if in_ch != 3:
          self.encoder.conv1 = torch.nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.deconv1 = deconv(512,256)
        self.deconv2 = deconv(256,128)
        self.deconv3 = deconv(128,64)
        self.out = out_conv(64, 64, n_classes)

    def forward(self, x):
        # x_in = torch.tensor(x.clone().detach())
        x_in = x.clone().detach()
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x = self.encoder.layer4(x3)
        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x, x_in)
        return x

# Download model
print("Creating model...")
encoder_name = 'resnet50'
encoder_weights = 'imagenet'
number_of_channels = 3
classes=1000
activation=None

model_library = "segmentation_models_pytorch"
if model_library == "custom":
    model = UNetResnet()
elif model_library == "segmentation_models_pytorch":
    model = smp.Unet(
        encoder_name, 
        encoder_weights=encoder_weights, 
        in_channels=number_of_channels, 
        classes=classes,
        activation=activation
    )
print(model)

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
    if model_library == "custom":
        pass
    elif model_library == "segmentation_models_pytorch":
        img = cv2.resize(img, (1024, 576))
    # img = cv2.resize(img, (1920, 1088))
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