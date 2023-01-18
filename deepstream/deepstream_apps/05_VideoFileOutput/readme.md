# Uso
python3 videoFileOutput.py -i \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_run.mov \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_walk.mov \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_1080p_h265.mp4 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_cam6.mp4 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_push.mov \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_ride_bike.mov

o

python3 videoFileOutput.py -i \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264 \
    file:///opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_apps/samples/sample_qHD.h264

Aquí *tiler_src_pad_buffer_probe* hace lo mismo que antes hacía *osd_sink_pad_buffer_probe*