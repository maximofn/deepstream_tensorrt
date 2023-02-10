import os
# import numpy as np
# from PIL import Image
import glob

import sys
sys.path.append('S2-FPN')
import S2FPN

net="SSFPN" #model name: [DSANet,SPFNet,SSFPN]
checkpoint_path = "S2-FPN/weigths/SSFPN18.pth" #use the file to load the checkpoint for evaluating or testing
save_seg_dir ="S2-FPN/borrar/" #saving path of prediction result
save =True #Save the predicted image
classes = 11 #number of classes
img_path = "S2-FPN/images/" #Images to be infered



def evaluate_model():
    """
     main function for testing
     param args: global arguments
     return: None
    """
    # build the model
    print(f"Bulding model: {net}", end=" ")
    model = S2FPN.load_model(net, classes)
    print("done")

    # load checkpoint
    print(f"loading checkpoint '{checkpoint_path}'", end=" ")
    model = S2FPN.load_checkpoints(model, checkpoint_path)
    print("done")

    # create save directory
    if save:
        if not os.path.exists(save_seg_dir):
            os.makedirs(save_seg_dir)

    # inference
    for i,file in enumerate(glob.glob(img_path+("/*.png"))):
        # Get path, name
        path, file_name = os.path.split(file)
        file_name, file_ext = os.path.splitext(file_name)

        # inference
        output, img = S2FPN.inference(model, file)
    
        # save the predicted image
        if save:
            S2FPN.save_predict(output, file_name, img, save_seg_dir,overlay=0.5)

evaluate_model()