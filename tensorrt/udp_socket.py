import socket
import numpy as np
import cv2

class udp_socket():
    def __init__(self, ip, port, max_buffer_size=65000, len_start_message=10, len_end_message=20, send=False):
        self.ip = ip
        self.port = port
        self.max_buffer_size = max_buffer_size
        self.len_start_message = len_start_message
        self.len_end_message = len_end_message
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,max_buffer_size)
        if not send: self.sock.bind((self.ip, self.port))
        self.len_message = 0
        self.len_message_old = 0
        self.shape_frame = (0, 0, 0)
        self.shape_frame_old = (0, 0, 0)
        self.frame = []
        self.start_frame = False
        self.end_frame = False
        self.num_end_end_frame = 0
        self.resize = False
        self.image = False
        self.message = None
        self.addr = None
        self.frame_decoded = None
        self.shape_frame_decoded = (0, 0, 0)
        self.shape_frame_decoded_old = (0, 0, 0)

    def receive(self):
        # Recibimos el mensaje del socket
        self.message, self.addr = self.sock.recvfrom(self.max_buffer_size)

        if len(self.message) == self.len_start_message:
            self.start_frame = True
            return None

        # Creamos un array NumPy a partir de la secuencia de bytes
        if len(self.message) == self.len_end_message:
            self.end_frame = True
            if self.num_end_end_frame == 0:
                self.num_end_end_frame += 1
            elif self.num_end_end_frame == 1:
                self.num_end_end_frame += 1
        else:
            if self.start_frame:
                self.frame = np.frombuffer(self.message, dtype=np.uint8)
                self.start_frame = False
            else:
                self.frame = np.append(self.frame, np.frombuffer(self.message, dtype=np.uint8))
    
    def receive_frame_ready(self):
        if self.end_frame and self.num_end_end_frame > 1:
            self.end_frame = False
            self.decode_frame()
            return True
        else:
            return False

    def get_encode_frame(self):
        return self.frame

    def decode_frame(self):
        self.frame_decoded = cv2.imdecode(self.frame, -1)
        self.shape_frame_decoded = self.frame_decoded.shape
    
    def get_frame_decoded(self):
        return self.frame_decoded
    
    def get_shape_frame_decoded(self):
        return self.shape_frame_decoded

    def new_shape_frame(self):
        if self.shape_frame_decoded != self.shape_frame_decoded_old:
            self.shape_frame_decoded_old = self.shape_frame_decoded
            return True
        else:
            return False
    
    def send(self, message):
        self.sock.sendto(message[:self.len_start_message], (self.ip, self.port))
        send = 0
        for i in range(0, len(message), self.max_buffer_size):
            self.sock.sendto(message[i:i + self.max_buffer_size], (self.ip, self.port))
            send += len(message[i:i + self.max_buffer_size])
        self.sock.sendto(message[:self.len_end_message], (self.ip, self.port))
        
    def close(self):
        self.sock.close()