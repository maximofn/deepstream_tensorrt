from thread import InputThread
from udp_socket import udp_socket
from video import video

display = False
resize = False

# Creamos una instancia de la clase InputThread y la iniciamos.
input_thread = InputThread()
input_thread.start()

# Creamos el socket UDP
sock = udp_socket('localhost', 8554, send=True)

# Abrimos la cámara en full HD
video = video(resize=resize, width=1920, height=1080, fps=30, name="frame", display=display)
video.open(device=0)

len_frame = 0
len_frame_old = 0
len_resized_frame = 0
len_resized_frame_old = 0

# Creamos un bucle infinito para enviar el vídeo por el socket
while True:
    # Leemos el frame de la cámara
    ret, frame = video.read()
    if not ret: break
    len_frame = len(frame.tobytes())
    if len_frame != len_frame_old:
        print(f"len_frame: {len_frame}, shape: {frame.shape}")
        len_frame_old = len_frame

    # Redimensionamos el array NumPy a un tamaño de 127x170
    if resize: frame = video.resize_frame(frame, width=127, height=170)
    if resize:
        len_resized_frame = len(frame.tobytes())
        if len_resized_frame != len_resized_frame_old:
            print(f"len_resized_frame: {len_resized_frame}, shape: {frame.shape}")
            len_resized_frame_old = len_resized_frame

    # Mostramos el frame
    if display:
        if not video.imshow(frame): break

    # Mandamos el frame por el socket
    success, encoded_frame = video.encode_frame(frame)
    if success:
        message = encoded_frame.tobytes(order='C')
        sock.send(message)

    # If user press type 'q' into the console in non blocking mode, exit
    if input_thread.get_data() is not None and input_thread.get_data().strip() == 'q':
        print("Se ha ha parado por el usuario")
        input_thread.clear_data()
        break

# Cerramos el socket y la cámara
sock.close()
video.close()
