from udp_socket import udp_socket
from video import video

# Creamos el socket UDP
socket = udp_socket('localhost', 8554)

# Creamos la clase video
RESIZE = False
video = video(resize=RESIZE, width=1920, height=1080, fps=30, name="received frame")

# Creamos un bucle infinito para recibir y mostrar el vídeo por el socket
while True:
    socket.receive()
    # Si hemos recibido el frame completo, lo mostramos
    if socket.receive_frame_ready():
        frame_decoded = socket.get_frame_decoded()
        if RESIZE: frame_decoded = video.resize_frame(frame_decoded, width=127, height=170)
        if not video.imshow(frame_decoded): break

# Cerramos la ventana y el socket
video.close()
socket.close()
