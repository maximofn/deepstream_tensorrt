from udp_socket import udp_socket
from video import video

# Creamos el socket UDP
socket_frame = udp_socket('localhost', 8554)
socket_mask = udp_socket('localhost', 8555)

# Creamos la clase video
RESIZE = False
frame = video(resize=RESIZE, width=1920, height=1080, fps=30, name="received frame")
mask = video(resize=RESIZE, width=1920, height=1080, fps=30, name="received mask")

# Creamos un bucle infinito para recibir y mostrar el vídeo por el socket
while True:
    socket_frame.receive()
    # Si hemos recibido el frame completo, lo mostramos
    if socket_frame.receive_frame_ready():
        frame_decoded = socket_frame.get_frame_decoded()
        if RESIZE: frame_decoded = frame.resize_frame(frame_decoded, width=127, height=170)
        if not frame.imshow(frame_decoded): break

    socket_mask.receive()
    # Si hemos recibido la máscara copleta, la mostramos
    if socket_mask.receive_frame_ready():
        mask_decoded = socket_mask.get_frame_decoded()
        if RESIZE: mask_decoded = mask.resize_frame(mask_decoded, width=127, height=170)
        if not mask.maskshow(mask_decoded): break

# Cerramos la ventana y el socket
frame.close()
mask.close()
socket_frame.close()
socket_mask.close()
