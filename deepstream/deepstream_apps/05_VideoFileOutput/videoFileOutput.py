#!/usr/bin/env python3

import argparse
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-6.1/deepstream_python_apps/apps/')

# Gstreamer
import gi # PyGObject is a Python package that enables links to librarys based on GObjects like GTK , GStreamer , WebKitGTK , GLib , GIO and more.
gi.require_version('Gst', '1.0') # GStreamer
gi.require_version('GstRtspServer', '1.0') # GStreamer rtsp server
from gi.repository import GObject, Gst, GstRtspServer, GLib

from ctypes import *
# import time
import math
# import platform

# common library
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS

# Config parser
# import configparser

# Python bindings for NVIDIA DeepStream SDK
import pyds

# FPS
fps_streams = {}
getfps_streams = {}

# Ready
ready = False

MAX_DISPLAY_LEN=64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"

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
#   ______________          _____________          _________          ____________________          ________________          _________          ________________          ____________          _______________          ___________          ________          __________
#  |              |        |             |        |         |        |                    |        |                |        |         |        |                |        |            |        |               |        |           |        |        |        |          |
#  | uridecodebin |------->| nvstreammux |------->| nvinfer |------->| nvmultistreamtiler |------->| nvvideoconvert |------->| nvdsosd |------->| nvvideoconvert |------->| capsfilter |------->| nvv4l2h264enc |------->| h264parse |------->| mp4mux |------->| filesink |
#  |______________|   |    |_____________|        |_________|        |____________________|        |________________|        |_________|        |________________|        |____________|        |_______________|        |___________|        |________|        |__________|
#         .           |
#         .           |    forms a batch          does inferencing   create subplots               convert video             draw overlay       convert video            not modify data         encode video            parses the           merges streams     write incoming 
#         .           |    of frames from         on input data                                    nv12 to RGBA              bounding boxes     RGBA to nv12             but can enforce         nv12 to h264/h265       incoming             (audio and video)  data to a file 
#   ______________    |    multiple input         using TensorRT                                   create buffer                                create buffer            limitations on                                  H264/H265 stream     into ISO MPEG-4    in the local 
#  |              |   |    sources before AI                                                                                                                             the data format                                                      (.mp4) files       file system
#  | uridecodebin |----
#  |______________|
#
#     input videos
#
#############################################################################################################################################################################################################################################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input-video", nargs='+', help="List of Path to input H264 elementry stream (required)", required=True)
    parser.add_argument("-o", "--output", default='out.mp4', help="Set the output file path ")
    parser.add_argument("--input-codec", default="H264", help="Input Codec H264/H265, default=H264", choices=['H264','H265'])
    parser.add_argument("--output-codec", default="H264", help="RTSP Streaming Codec H264/H265, default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000, help="Set the encoding bitrate, default=4000000", type=int)
    parser.add_argument("-p", "--port", default=8554, help="Port of RTSP stream, default=8554", type=int)
    parser.add_argument("--primary_config_file",   default="dstest2_pgie_config.txt", help="Config file, default=dstest2_pgie_config.txt")
    parser.add_argument("-s", "--stream-name", default="stream1", help="Stream name, default=stream1")
    
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    batch_size = len(sys.argv)-2
    
    global stream_path
    global output_path
    global input_codec
    global output_codec
    global bitrate
    global port
    global primary_config_file
    global stream_name
    
    stream_path = args.input_video
    output_path = args.output
    input_codec = args.input_codec
    output_codec = args.output_codec
    bitrate = args.bitrate
    port = args.port
    primary_config_file = f"dstest3_pgie_config_b{batch_size}.txt"
    stream_name = args.stream_name
    
    return 0


# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
def tiler_src_pad_buffer_probe(pad,info,u_data):
    global ready
    if ready == False:
        ready = True
        print("\n Ready to stream")
    
    getfps_streams["stream{0}".format(0)].update_fps()
    fps_streams["stream{0}".format(0)] = getfps_streams["stream{0}".format(0)].get_fps()

    frame_number=0
    old_frame_number=-1
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
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
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

        # py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])
        py_nvosd_text_params.display_text = f"stream={frame_meta.pad_index}\n\
Frame Number={frame_number:04d}\nfps={fps_streams['stream{0}'.format(0)]:.2f}\n\
Number of Objects={num_rects:03d}\n\
Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]:03d}\n\
Person_count={obj_counter[PGIE_CLASS_ID_PERSON]:04d}"
        
        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10;
        py_nvosd_text_params.y_offset = 12;
        
        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        
        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
        
        # send the display overlay to the screen
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        if old_frame_number != frame_number:
            # print("Frame Number=", frame_number, "Number of Objects=",num_rects,"Vehicle_count=",obj_counter[PGIE_CLASS_ID_VEHICLE],"Person_count=",obj_counter[PGIE_CLASS_ID_PERSON])
            old_frame_number = frame_number
        
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print(" In cb_newpad: gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print(" Decodebin child added:", name)
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

def create_source_bin(index,uri):
    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    print(f"\t\t\tCreating uridecodebin for {bin_name}: {uri}. Source element for reading from the uri")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin



def main(args):
    # Check input arguments (altered using argparse)
    if len(args) < 2:
        sys.stderr.write("usage: %s -i <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    # Get number of video sources
    number_sources=len(args)-2

    # Init FPS
    for i in range(0,len(args)-1):
        getfps_streams["stream{0}".format(i)]=GETFPS(i)
        fps_streams["stream{0}".format(i)]=0

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

    # The first element type in the pipeline to be created is nvstreammux, with name streammux, which is used to multiplex, or batch, the input sources. 
    # Create nvstreammux instance to form batches from one or more sources, batch video streams before sending for AI inference
    print(" \t Creating nvstreammux instance, multiplex video streams")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    
    # For each URI path source provided by the user, a source_bin is created and added to the pipeline to be linked later. Each source bin can be thought of as a mini-pipeline that automatically decodes the input referenced by the URI (either file or stream). Each new source bin has its own source and sink pad:
    pipeline.add(streammux)
    is_live = False
    for i in range(number_sources):
        uri_name=args[i+2]
        print(f"\t\t Creating source_bin {i:02d}: {uri_name}")
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    
    # Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file, runs inference using TensorRT
    print("\t Creating nvinfer, runs primary inference using TensorRT")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    
    # Use nvmultistreamtiler to create subplots
    print(" \t Creating tiler, to create subplots")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    
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

    # Since the data format in the input file is elementary h264 or h265 stream, we need a h264parser h265parser, parses the incoming H264/H265 stream
    print("\t Creating H264Parser, parses the incoming H264/H265 stream")
    codecparse = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not codecparse:
        sys.stderr.write(" Unable to create h264 parser \n")
    
    # This element merges streams (audio and video) into ISO MPEG-4 (.mp4) files
    print("\t Creating mp4mux, merges streams (audio and video) into ISO MPEG-4 (.mp4) files")
    mux = Gst.ElementFactory.make("mp4mux", "mux")
    if not mux:
        sys.stderr.write(" Unable to create mux \n")

    # Write incoming data to a file in the local file system
    print("\t Creating filesink, write incoming data to a file in the local file system")
    sink = Gst.ElementFactory.make("filesink", "filesink")
    if not sink:
        sys.stderr.write(" Unable to create filesink \n")
    sink.set_property('location', output_path)
    
    
    ####################################################################################
    # Configure streammux properties
    ####################################################################################
    print(" Configure streammux properties")
    if is_live:
        print("\t Atleast one of the sources is live")
        streammux.set_property('live-source', 1)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    
    
    ####################################################################################
    # Configure inference properties
    ####################################################################################
    print(" Configure inference properties")
    print(f"\t Open {primary_config_file} file\t", end="")
    pgie.set_property('config-file-path', primary_config_file)
    pgie_batch_size = pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print(color.YELLOW + f"WARNING: Overriding infer-config batch-size {pgie_batch_size} with number of sources {number_sources}" + color.END)
        pgie.set_property("batch-size",number_sources)
    else:
        print(color.GREEN + "OK" + color.END)
    # print(f"\t Open {secondary_config_file} file\t", end="")
    # sgie.set_property('config-file-path', secondary_config_file)
    # print(color.GREEN + "OK" + color.END)
    # print(f"\t Open {tertiary_config_file} file\t", end="")
    # tgie.set_property('config-file-path', tertiary_config_file)
    # print(color.GREEN + "OK" + color.END)
    
    
    ####################################################################################
    # Configure tiler properties
    ####################################################################################
    print(" Configure tiler properties")
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)

    
    
    ####################################################################################
    # Adding elements to Pipeline
    ####################################################################################
    print(" Adding elements to Pipeline")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(codecparse)
    pipeline.add(mux)
    pipeline.add(sink)

    
    ####################################################################################
    # Link the elements together
    ####################################################################################
    print(" Linking elements in the Pipeline")
    
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tiler)
    tiler.link(queue3)
    queue3.link(nvvidconv)
    nvvidconv.link(queue4)
    queue4.link(nvosd)
    nvosd.link(queue5)
    queue5.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(codecparse)
    codecparse.link(mux)
    mux.link(sink)
    
    
    ###################################################################################
    # create an event loop and feed gstreamer bus mesages to it
    ####################################################################################
    print(" Creating an event loop and feed gstreamer bus mesages to it")
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    tiler_src_pad=pgie.get_static_pad("src")
    if not tiler_src_pad:
        sys.stderr.write("\t Unable to get src pad \n")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)
    
    
    ###################################################################################
    # list of sources
    ####################################################################################
    print(" Video sources:")
    for i, source in enumerate(args):
        if (i > 1):
            print(f"\t {i}: {source}")

    
    
    ##################################################################################
    # Starting pipeline, start play back and listed to events
    ###################################################################################
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

