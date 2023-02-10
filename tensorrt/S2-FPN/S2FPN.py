from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from builders.model_builder import build_model
import torch.backends.cudnn as cudnn
import os
import cv2

navig_palette = [170, 170, 170, 0, 255, 0, 255, 0, 0, 0, 120, 255, 0, 0, 255, 255,255,153]
labels_names = ["path","travesable","no-travesable","sky","person","vehicle"]

def colorize_mask(mask, img_orig=None, overlay=0):
    # mask: numpy array of the mask
    # img_orig: original image
    # overlay: overlay factor (0-1)
    colorized_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    colorized_mask.putpalette(navig_palette)
    colorized_mask = np.array(colorized_mask.convert("RGB")) 
    colorized_mask = colorized_mask[:, :, ::-1].copy() 
    if overlay and img_orig is not None:
        result = cv2.addWeighted(img_orig, overlay, colorized_mask, 1-overlay, 0)
        return result
    else:
        return colorized_mask

def save_predict(output, img_name,img_orig, save_path,overlay=0):
    output_color = colorize_mask(output)
    if overlay:
        rgb_mask = np.array(output_color.convert("RGB")) 
        rgb_mask = rgb_mask[:, :, ::-1].copy() 
        result = cv2.addWeighted(img_orig,overlay,rgb_mask,1-overlay,0)
        cv2.imwrite(os.path.join(save_path, img_name + '_color.png'),result)

def load_model(network, classes): 
    if not torch.cuda.is_available():
        raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(network, num_classes=classes)
    model = model.cuda()  # using GPU for inference
    cudnn.benchmark = True

    return model

def inference(model, file=None, img_orig=None):
    """
    args:
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    
    if file:
        img_orig = cv2.imread(file, cv2.IMREAD_COLOR)
        print(type(img_orig), img_orig.shape, img_orig.dtype)
    image = np.asarray(img_orig, np.float32)
    # image = image.astype(np.float32) / 255.0
    image = image[:, :, ::-1]  # change to RGB
    image = image.transpose((2, 0, 1)).copy()  # HWC -> CHW
    image = torch.from_numpy(image)
    image = image[None,:,:,:]

    with torch.no_grad():
        input_var = Variable(image).cuda()
    output = model(input_var) #,output5,output6,output7,output8
    torch.cuda.synchronize()
    output = output.cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

    return output, img_orig

def load_checkpoints(model, checkpoint_path):
    """
    load checkpoint
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"no checkpoint found at '{checkpoint_path}'")
        raise FileNotFoundError("no checkpoint found at '{}'".format(checkpoint_path))
    
    return model