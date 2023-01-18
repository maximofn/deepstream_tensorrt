
import os
#from pytorch2onnx_classifier_orig import convert_to_onnx
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
import onnx
import onnxruntime
from torchvision import models




def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('--model-path', type=str,
                        help='pytorch model path')
    args.add_argument('--version', type=str,
                        help='conceptual type, i.e. people, vehicles, age, gender, etc')

    return parser


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    return model_ft,input_size

def convert_to_onnx(pth_path, n_classes, output_path, opset=12, verbose=True, device='cuda:0'):

    model, input_size = initialize_model('resnet', n_classes, False, use_pretrained=False)
    model.to(torch.device(device)).eval()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device(device)))
    model = nn.Sequential(model,nn.Softmax(1))

    input_blob = torch.randn((1, 3, input_size,input_size), requires_grad=True).cuda()

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_blob.cuda(),
            output_path,
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},
        )
    onnx_net = onnx.load(output_path)
    try:
        onnx.checker.check_model(onnx_net)
        print('ONNX check passed.')
        sess = onnxruntime.InferenceSession(output_path)
        _, ch, h, w = sess.get_inputs()[0].shape
        print("The model expects input shape: ", sess.get_inputs()[0].shape)

    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print('ONNX check failed: {}.'.format(ex))

    return ch, h, w


if __name__ == "__main__":
    args = build_argparser().parse_args()
    pth_path = args.model_path
    version = args.version
    n_classes = 3
    min_batch = 1
    opt_batch = 1
    max_batch = 1

    src_path = os.sep.join(pth_path.split(os.sep)[:-1])
    pth_name = os.path.splitext(pth_path.split(os.sep)[-1])[0]

    onnx_output_name = "resnet34_" + version + ".onnx"
    onnx_path = os.path.join(src_path, onnx_output_name)

    ch, h, w = convert_to_onnx(pth_path, n_classes, onnx_path)

    trt_output_name = f"{os.path.splitext(onnx_output_name)[0]}.engine"
    trt_path = os.path.join(src_path, trt_output_name)

    trt_command = f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} \
                        --saveEngine={trt_output_name} \
                        --minShapes=input:{min_batch}x{ch}x{h}x{w} \
                        --optShapes=input:{opt_batch}x{ch}x{h}x{w} \
                        --maxShapes=input:{max_batch}x{ch}x{h}x{w} "
    os.system(trt_command)

