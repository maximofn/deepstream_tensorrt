#!/usr/bin/env python3

import argparse
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-6.1/deepstream_python_apps/apps/')

# Gstreamer
import gi # PyGObject is a Python package that enables links to librarys based on GObjects like GTK , GStreamer , WebKitGTK , GLib , GIO and more.
gi.require_version('Gst', '1.0') # GStreamer
gi.require_version('GstRtspServer', '1.0') # GStreamer rtsp server
from gi.repository import GObject, Gst, GstRtspServer, GLib

# common library
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS

# Config parser
import configparser

# Python bindings for NVIDIA DeepStream SDK
import pyds

# FPS
fps_stream = {}

# Ready
ready = False

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

# past traking meta data
past_tracking_meta=[0]

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

ready = False



#############################################################################################################################################################################################################################################################################################################################
#   _________          _____________          _______________          _____________          _________          ___________          _________          _________          _________          ________________          _________          ________________          ____________          _______________          ____________          _________
#  |         |        | h264        |        |               |        |             |        |         |        |           |        |         |        |         |        |         |        |                |        |         |        |                |        |            |        |               |        |            |        |         |
#  | filesrc |------->|  /   parser |------->| nvv4l2decoder |------->| nvstreammux |------->| nvinfer |------->| nvtracker |------->| nvinfer |------->| nvinfer |------->| nvinfer |------->| nvvideoconvert |------->| nvdsosd |------->| nvvideoconvert |------->| capsfilter |------->| nvv4l2h264enc |------->| rtph264pay |------->| udpsink |
#  |_________|        |_h265________|        |_______________|        |_____________|        |_________|        |___________|        |_________|        |_________|        |_________|        |________________|        |_________|        |________________|        |____________|        |_______________|        |____________|        |_________|
#                   
#   open video         parse video            decode video             forms a batch         does inferencing    low-level tracker   does inferencing   does inferencing   does inferencing    convert video             draw overlay       convert video            not modify data         encode video            Payload-encode       sends UDP packets 
#                                             h264/h265 to nv12        of frames from        on input data       library to track    on input data      on input data      on input data       nv12 to RGBA              bounding boxes     RGBA to nv12             but can enforce         nv12 to h264/h265       H264 video into      to the network
#                                                                      multiple input        using TensorRT      the detected objcts using TensorRT     using TensorRT     using TensorRT      create buffer                                create buffer            limitations on                                  RTP packets
#                                                                      sources before AI                         with persistent                                                                                                                                     the data format
#                                                                                                                (possibly unique)
#                                                                                                                IDs over time
#############################################################################################################################################################################################################################################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input-video", help="Path to input H264 elementry stream (required)", required=True)
    parser.add_argument("--input-codec", default="H264", help="Input Codec H264/H265, default=H264", choices=['H264','H265'])
    parser.add_argument("--output-codec", default="H264", help="RTSP Streaming Codec H264/H265, default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000, help="Set the encoding bitrate, default=4000000", type=int)
    parser.add_argument("-p", "--port", default=8554, help="Port of RTSP stream, default=8554", type=int)
    parser.add_argument("--primary_config_file",   default="dstest2_pgie_config.txt", help="Config file, default=dstest2_pgie_config.txt")
    parser.add_argument("--secondary_config_file1", default="dstest2_sgie_config1.txt", help="Config file, default=dstest2_sgie_config1.txt")
    parser.add_argument("--secondary_config_file2", default="dstest2_sgie_config2.txt", help="Config file, default=dstest2_sgie_config2.txt")
    parser.add_argument("--secondary_config_file3", default="dstest2_sgie_config3.txt", help="Config file, default=dstest2_sgie_config3.txt")
    # parser.add_argument("--tertiary_config_file",  default="dstest2_tgie_config.txt", help="Config file, default=dstest2_tgie_config.txt")
    parser.add_argument("--tracker_config_file",   default="dstest2_tracker_config.txt", help="Config file, default=dstest2_tracker_config.txt")
    parser.add_argument("-m", "--meta", default=0, help="set past tracking meta, default=0", type=int)
    parser.add_argument("-s", "--stream-name", default="stream1", help="Stream name, default=stream1")
    
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    
    global stream_path
    global input_codec
    global output_codec
    global bitrate
    global port
    global primary_config_file
    global secondary_config_file1
    global secondary_config_file2
    global secondary_config_file3
    # global tertiary_config_file
    global tracker_config_file
    global past_tracking
    global stream_name
    
    stream_path = args.input_video
    input_codec = args.input_codec
    output_codec = args.output_codec
    bitrate = args.bitrate
    port = args.port
    primary_config_file = args.primary_config_file
    secondary_config_file1 = args.secondary_config_file1
    secondary_config_file2 = args.secondary_config_file2
    secondary_config_file3 = args.secondary_config_file3
    # tertiary_config_file = args.tertiary_config_file
    tracker_config_file = args.tracker_config_file
    past_tracking = args.meta
    stream_name = args.stream_name
    
    return 0


def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    
    global ready
    if ready == False:
        ready = True
        print("\n Ready to stream")
    
    fps_stream[0].update_fps()
    fps = fps_stream[0].get_fps()

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta # Number of rectangles ==> Number of objects
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = (f"Frame Number={frame_number:04d}\t\
fps={fps:.2f}\tNumber of Objects={num_rects:03d}   \
Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]:03d}   \
Person_count={obj_counter[PGIE_CLASS_ID_PERSON]:03d}")

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 20
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    
    
    #past traking meta data
    past_tracking_meta[0] = past_tracking
    if(past_tracking_meta[0]==1):
        l_user=batch_meta.batch_user_meta_list
        print(f"l_user = {l_user}")
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                user_meta=pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
                try:
                    # Note that user_meta.user_meta_data needs a cast to pyds.NvDsPastFrameObjBatch
                    # The casting is done by pyds.NvDsPastFrameObjBatch.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone
                    pPastFrameObjBatch = pyds.NvDsPastFrameObjBatch.cast(user_meta.user_meta_data)
                except StopIteration:
                    break
                for trackobj in pyds.NvDsPastFrameObjBatch.list(pPastFrameObjBatch):
                    print("streamId=",trackobj.streamID)
                    print("surfaceStreamID=",trackobj.surfaceStreamID)
                    for pastframeobj in pyds.NvDsPastFrameObjStream.list(trackobj):
                        print("numobj=",pastframeobj.numObj)
                        print("uniqueId=",pastframeobj.uniqueId)
                        print("classId=",pastframeobj.classId)
                        print("objLabel=",pastframeobj.objLabel)
                        for objlist in pyds.NvDsPastFrameObjList.list(pastframeobj):
                            print('frameNum:', objlist.frameNum)
                            print('tBbox.left:', objlist.tBbox.left)
                            print('tBbox.width:', objlist.tBbox.width)
                            print('tBbox.top:', objlist.tBbox.top)
                            print('tBbox.right:', objlist.tBbox.height)
                            print('confidence:', objlist.confidence)
                            print('age:', objlist.age)
            try:
                l_user=l_user.next
            except StopIteration:
                break
    return Gst.PadProbeReturn.OK




def main(args):
    # Init FPS
    fps_stream[0] = GETFPS(0)

    # Standard GStreamer initialization
    gst_status, _ = Gst.init_check(None)    # GStreamer initialization
    if not gst_status:
        sys.stderr.write("Unable to initialize Gst\n")
        sys.exit(1)

    ####################################################################################
    # Create gstreamer elements
    ####################################################################################
    print(" Create gstreamer elements")
    
    # Create Pipeline element that will form a connection of other elements
    print("\t Creating Pipeline")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    
    # Source element for reading from the file, reads the video data from file
    print("\t Creating Source, reads the video data from file")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")
    
    # Since the data format in the input file is elementary h264 or h265 stream, we need a h264parser h265parser, parses the incoming H264/H265 stream
    if input_codec == "H264":
        print("\t Creating H264Parser, parses the incoming H264/H265 stream")
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parser:
            sys.stderr.write(" Unable to create h264 parser \n")
    elif input_codec == "H265":
        print("\t Creating H265Parser")
        h265parser = Gst.ElementFactory.make("h265parse", "h265-parser")
        if not h265parser:
            sys.stderr.write(" Unable to create h265 parser \n")
    
    # Use nvdec_h264 or nvdec_h265 for hardware accelerated decode on GPU, hardware accelerated decoder; decodes video streams using NVDEC
    print("\t Creating Decoder, hardware accelerated decoder; decodes video streams using NVDEC")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")
    
    # Create nvstreammux instance to form batches from one or more sources, batch video streams before sending for AI inference
    print("\t Creating nvstreammux instance, batch video streams before sending for AI inference")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    
    # Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file, runs inference using TensorRT
    print("\t Creating nvinfer, runs primary inference using TensorRT")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    
    # Allows the DS pipeline to use a low-level tracker library to track the detected objects with persistent (possibly unique) IDs over time
    print("\t Creating nvtracker, allows the DS pipeline to use a low-level tracker library to track the detected objects with persistent (possibly unique) IDs over time")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    # Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file, runs inference using TensorRT
    print("\t Creating nvinfer, runs secondary inference using TensorRT")
    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-inference")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 \n")

    # Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file, runs inference using TensorRT
    print("\t Creating nvinfer, runs secondary inference using TensorRT")
    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-inference")
    if not sgie2:
        sys.stderr.write(" Unable to make sgie2 \n")

    # Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file, runs inference using TensorRT
    print("\t Creating nvinfer, runs secondary inference using TensorRT")
    sgie3 = Gst.ElementFactory.make("nvinfer", "secondary3-inference")
    if not sgie3:
        sys.stderr.write(" Unable to make sgie3 \n")

    # tgie = Gst.ElementFactory.make("nvinfer", "tertiary-inference")
    # if not tgie:
    #     sys.stderr.write(" Unable to make tgie \n")

    # qgie = Gst.ElementFactory.make("nvinfer", "quaternary-inference")
    # if not qgie:
    #     sys.stderr.write(" Unable to make qgie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd, performs video color format conversion (I420 to RGBA)
    print("\t Creating convertor, performs video color format conversion (I420 to RGBA)")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    
    # Create OSD to draw on the converted RGBA buffer, draw bounding boxes, text and region of interest (ROI) polygons
    print("\t Creating OSD, draw bounding boxes, text and region of interest (ROI) polygons")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Performs video color format conversion (RGBA to I420)
    print("\t Performs video color format conversion (RGBA to I420)")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    
    # Create a caps filter, enforces limitations on data (no data modification)
    print("\t Creating caps filter, enforces limitations on data (no data modification)")
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    
    # Make the encoder, encodes RAW data in I420 format to H264/H265
    if output_codec == "H264":
        print("\t Creating H264 Encoder, encodes RAW data in I420 format to H264")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    elif output_codec == "H265":
        print("\t Creating H265 Encoder, encodes RAW data in I420 format to H265")
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', bitrate)
    if is_aarch64():
        print("\t\t Is aarch64")
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)
    
    # Make the payload-encode video into RTP packets, converts H264/H265 encoded Payload to RTP packets (RFC 3984)
    print("\t Make the payload-encode video into RTP packets, converts H264/H265 encoded Payload to RTP packets (RFC 3984)")
    if output_codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("\t Creating H264 rtppay")
    elif output_codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("\t Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    
    # Make the UDP sink, sends UDP packets to the network. When paired with RTP payloader (Gst-rtph264pay) it can implement RTP streaming
    updsink_port_num = 5400
    print(f"\t Make the UDP sink in port {updsink_port_num}, sends UDP packets to the network. When paired with RTP payloader (Gst-rtph264pay) it can implement RTP streaming")
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
        
        
    ####################################################################################
    # Configure sink properties
    ####################################################################################
    print(" Configure sink properties")
    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)
    
    
    ####################################################################################
    # Configure source properties
    ####################################################################################
    print(" Configure source properties")
    print("\t Playing file %s " %stream_path)
    source.set_property('location', stream_path)


    ####################################################################################
    # Configure streammux properties
    ####################################################################################
    print(" Configure streammux properties")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    
    
    # ####################################################################################
    # # Configure inference properties
    # ####################################################################################
    print(" Configure inference properties")
    print(f"\t Open {primary_config_file} file\t", end="")
    pgie.set_property('config-file-path', primary_config_file)
    print(color.GREEN + "OK" + color.END)
    print(f"\t Open {secondary_config_file1} file\t", end="")
    sgie1.set_property('config-file-path', secondary_config_file1)
    print(color.GREEN + "OK" + color.END)
    print(f"\t Open {secondary_config_file2} file\t", end="")
    sgie2.set_property('config-file-path', secondary_config_file2)
    print(color.GREEN + "OK" + color.END)
    print(f"\t Open {secondary_config_file3} file\t", end="")
    sgie3.set_property('config-file-path', secondary_config_file3)
    print(color.GREEN + "OK" + color.END)
    # print(f"\t Open {tertiary_config_file} file\t", end="")
    # tgie.set_property('config-file-path', tertiary_config_file)
    # print(color.GREEN + "OK" + color.END)
    
    
    ####################################################################################
    # Set properties of tracker
    ####################################################################################
    print(" Set properties of tracker")
    config = configparser.ConfigParser()
    print(f"\t Open {tracker_config_file} file\t", end="")
    config.read(tracker_config_file)
    print(color.GREEN + "OK" + color.END)
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    
    
    ####################################################################################
    # Adding elements to Pipeline
    ####################################################################################
    print(" Adding elements to Pipeline")
    pipeline.add(source)
    if input_codec == "H264":
        pipeline.add(h264parser)
    elif input_codec == "H265":
        pipeline.add(h265parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(sgie2)
    pipeline.add(sgie3)
    # pipeline.add(tgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    
    ####################################################################################
    # Link the elements together
    ####################################################################################
    print(" Linking elements in the Pipeline")
    
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
        
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    if input_codec == "H264":
        source.link(h264parser) # source-parser
        h264parser.link(decoder) # parser-decoder
    elif input_codec == "H265":
        source.link(h265parser) # source-parser
        h265parser.link(decoder) # parser-decoder
    srcpad.link(sinkpad) # decoder-streammux
    streammux.link(pgie) # streammux-pgie
    pgie.link(tracker) # pgie-tracker
    tracker.link(sgie1) # tracker-sgie1
    sgie1.link(sgie2) # sgie1-sgie2
    # sgie2.link(nvvidconv) # sgie1-nvvidconv
    sgie2.link(sgie3) # sgie2-sgie3
    # sgie.link(tgie) # sgie-tgie
    # tgie.link(nvvidconv) # tgie-nvvidconv
    sgie3.link(nvvidconv) #sgie3-nvvidconv
    nvvidconv.link(nvosd) # nvvidconv-nvosd
    nvosd.link(nvvidconv_postosd) # nvosd-nvvidconv_postosd
    nvvidconv_postosd.link(caps) # nvvidconv_postosd-caps
    caps.link(encoder) # caps-encoder
    encoder.link(rtppay) # encoder-rtppay
    rtppay.link(sink) # rtppay-sink
    
    
    ###################################################################################
    # create an event loop and feed gstreamer bus mesages to it
    ####################################################################################
    print(" Creating an event loop and feed gstreamer bus mesages to it")
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    
    ###################################################################################
    # Configure RTSP server
    ####################################################################################
    print(" Configure RTSP port")
    server = GstRtspServer.RTSPServer.new()
    server.props.service = f"{port}"
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( f"( udpsrc name=pay0 port={updsink_port_num} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string){output_codec}, payload=96 \" )")
    factory.set_shared(True)
    server.get_mount_points().add_factory(f"/{stream_name}", factory)
    print("\t Launched RTSP Streaming at " + color.UNDERLINE + color.GREEN + f"rtsp://localhost:{port}/{stream_name}" + color.END)
    
    
    ###################################################################################
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    ####################################################################################
    print(" Get metadata from OSD element")
    print("\t Get sink of OSD element")
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    print("\t Get probe to get informed of the meta data generated")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # start play back and listen to events
    print(" Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass


    ###################################################################################
    # cleanup
    ###################################################################################
    pipeline.set_state(Gst.State.NULL)

    
    


if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv))

