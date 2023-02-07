import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from argparse import ArgumentParser
# user
from PIL import Image

import seaborn as sn
import matplotlib.pyplot as plt
import glob
import cv2

import sys
sys.path.append('S2-FPN')
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test

net="SSFPN" #model name: [DSANet,SPFNet,SSFPN]
checkpoint_path = "S2-FPN/weigths/SSFPN18.pth" #use the file to load the checkpoint for evaluating or testing
save_seg_dir ="S2-FPN/borrar/" #saving path of prediction result
save =True #Save the predicted image
classes = 11 #number of classes
img_path = "S2-FPN/images/" #Images to be infered
navig_palette = [170, 170, 170, 0, 255, 0, 255, 0, 0, 0, 120, 255, 0, 0, 255, 255,255,153]
labels_names = ["path","travesable","no-travesable","sky","person","vehicle"]

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(navig_palette)
    return new_mask #gpu ids (default: 0)

def save_predict(output, img_name,img_orig, save_path,overlay=0):
    output_color = colorize_mask(output)
    if overlay:
        rgb_mask = np.array(output_color.convert("RGB")) 
        rgb_mask = rgb_mask[:, :, ::-1].copy() 
        result = cv2.addWeighted(img_orig,overlay,rgb_mask,1-overlay,0)
        cv2.imwrite(os.path.join(save_path, img_name + '_color.png'),result)

def load_model(network): 
    if not torch.cuda.is_available():
        raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(network, num_classes=classes)
    model = model.cuda()  # using GPU for inference
    cudnn.benchmark = True

    return model

def inference(model):
    """
    args:
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    if save:
        if not os.path.exists(save_seg_dir):
            os.makedirs(save_seg_dir)
    for i,file in enumerate(glob.glob(img_path+("./*.png"))):
        n_ims = len(glob.glob(img_path+("./*.png")))
        img_orig = cv2.imread(file, cv2.IMREAD_COLOR)
        name = file.split("/")[-1][:-4]

        image = np.asarray(img_orig, np.float32)
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1)).copy()  # HWC -> CHW
        image = torch.from_numpy(image)
        image = image[None,:,:,:]

        with torch.no_grad():
            input_var = Variable(image).cuda()
        start_time = time.time()
        output = model(input_var) #,output5,output6,output7,output8
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # save the predicted image
        if save:
            save_predict(output, name, img_orig, save_seg_dir,overlay=0.5)

def evaluate_model():
    """
     main function for testing
     param args: global arguments
     return: None
    """
    # build the model
    print(f"Bulding model: {net}", end=" ")
    model = load_model(net)
    print("done")

    print(f"loading checkpoint '{checkpoint_path}'", end=" ")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        print("done")
    else:
        print(f"no checkpoint found at '{checkpoint_path}'")
        raise FileNotFoundError("no checkpoint found at '{}'".format(checkpoint_path))

    inference(model)

evaluate_model()