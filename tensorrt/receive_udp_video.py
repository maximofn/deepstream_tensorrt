from udp_socket import udp_socket
from video import video

# Creamos el socket UDP
socket_frame = udp_socket('localhost', 8554)
socket_mask = udp_socket('localhost', 8555)
socket_mask_colorized = udp_socket('localhost', 8556)

# Creamos la clase video
RESIZE = False
frame = video(resize=RESIZE, width=1920, height=1080, fps=30, name="received frame")
mask = video(resize=RESIZE, width=1920, height=1080, fps=30, name="received mask, colorized into host")
mask_colorized = video(resize=RESIZE, width=1920, height=1080, fps=30, name="received colorized into docker mask")

# Creamos un bucle infinito para recibir y mostrar el vídeo por el socket
frame_decoded = None
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
        # Chequeamos si los tres canales de la máscara son iguales
        if (mask_decoded[:, :, 0] == mask_decoded[:, :, 1]).all() and (mask_decoded[:, :, 1] == mask_decoded[:, :, 2]).all():
            mask_decoded = mask_decoded[:, :, 0]
        else:
            print("Los tres canales de la máscara no son iguales")
        if RESIZE: mask_decoded = mask.resize_frame(mask_decoded, width=127, height=170)
        if frame_decoded is not None:
            if not mask.maskshow(mask_decoded, img=frame_decoded, overlay=0.2, colorize=True): break
        
    socket_mask_colorized.receive()
    # Si hemos recibido la máscara coloreada copleta, la mostramos
    if socket_mask_colorized.receive_frame_ready():
        mask_colorized_decoded = socket_mask_colorized.get_frame_decoded()
        if RESIZE: mask_colorized_decoded = mask_colorized.resize_frame(mask_colorized_decoded, width=127, height=170)
        if not mask_colorized.imshow(mask_colorized_decoded): break

# Cerramos la ventana y el socket
frame.close()
mask.close()
mask_colorized.close()
socket_frame.close()
socket_mask.close()
socket_mask_colorized.close()
