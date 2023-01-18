Se puede usar cualquier código de los de antes, la diferencia está en la llamada a la red mediante el archivo de configuración

# Compilar Yolo
 * Ir a la carpeta de *objectDetector_Yolo*
 * Descargar los modelos y los pesos: ejecutar $./prebuild.sh$
 * Ir a la carpeta *nvdsinfer_custom_impl_Yolo* que está dentro de *objectDetector_Yolo*
 * Cambiar las rutas relativas por rutas absolutas
   * Cambiar $CFLAGS+= -I../../includes -I/usr/local/cuda-$(CUDA_VER)/include$ por $CFLAGS+= -I/opt/nvidia/deepstream/deepstream-6.1/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include$
   * Cambiar $CUFLAGS:= -I../../includes -I/usr/local/cuda-$(CUDA_VER)/include$ por $CUFLAGS:= -I/opt/nvidia/deepstream/deepstream-6.1/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include$
 * Ver la versión de cuda viendo qué hay dentro de */usr/local* y ver la vesión de cuda en los nombres de las carpetas de cuda
 * Compilar mediante $CUDA_VER=$CUDA_VER make$, en mi caso la versión de cuda es la 11.7, así que ejecuto $CUDA_VER=11.7 make$

# Uso Yolo
python3 yolo.py -i \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_run.mov \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_walk.mov \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_1080p_h265.mp4 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_cam6.mp4 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_push.mov \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_ride_bike.mov

o

python3 yolo.py -i \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264