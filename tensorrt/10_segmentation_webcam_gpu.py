import torch
import torchvision
import cv2
import numpy as np
import time
from thread import InputThread
from udp_socket import udp_socket
from video import video

# Create input thread
input_thread = InputThread()
input_thread.start()

# Create UDP socket
sock = udp_socket('localhost', 8554, send=True)

# Open webcam
video = video(resize=False, width=1920, height=1080, fps=30, name="frame", display=False)
video.open(device=0)