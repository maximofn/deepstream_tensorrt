import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from PIL import Image

import seaborn as sn
import matplotlib.pyplot as plt
import glob
import cv2

net="SSFPN" #model name: [DSANet,SPFNet,SSFPN]
dataset="navig" #dataset: cityscapes or camvid
num_workers=1 #the number of parallel threads
batch_size=1 #the batch_size is set to 1 when evaluating or testing
checkpoint_path = "./SSFPN18.pth" #use the file to load the checkpoint for evaluating or testing
save_seg_dir ="./borrar/" #saving path of prediction result
best = False #Get the best result among last few checkpoints
save =True #Save the predicted image
cuda=True #run on CPU or GPU
gpus="0" #gpu ids (default: 0)
classes = 11 #number of classes
is_inference = True #weather you want to infer over images (True) or evaluate  the model (False)
img_path = "./images/" #Images to be infered
navig_palette = [170, 170, 170, 0, 255, 0, 255, 0, 0, 0, 120, 255, 0, 0, 255, 255,255,153]
labels_names = ["path","travesable","no-travesable","sky","person","vehicle"]

def generateM(gt,pred):
    m = np.zeros((classes,classes))
    assert (len(gt) == len(pred))
    for i in range(len(gt)):
        if gt[i] < classes:  # and pred[i] < self.nclass:
            m[gt[i], pred[i]] += 1.0
    return m

def jaccard(cm):
    jaccard_perclass = []
    for i in range(classes):
        #if not cm[i, i] == 0:
        jaccard_perclass.append(cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]))

    return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass

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

def load_model(): 
    if cuda:
        print("=====> use gpu id: '{}'".format(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(net, num_classes=classes)
    if cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    return model

def test(test_loader, model,eval=False):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    
    cm = np.zeros((classes, classes))

    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = Variable(input).cuda()
        start_time = time.time()
        output = model(input_var) #,output5,output6,output7,output8
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        #print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        cm += generateM(gt.flatten(), output.flatten())
        
        # # save the predicted image
        # if save and not best:
        #     save_predict(output, gt, name[0].split("/")[-1],save_seg_dir,save, gt_color_save=False,overlay=False,img_orig=None)
    
    meanIoU, per_class_iu = jaccard(cm)

    return meanIoU, per_class_iu, cm

def inference(model):
    """
    args:
      test_loader: loaded for test dataset
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
        print('[%d/%d]  time: %.2f' % (i + 1, n_ims, time_taken))
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
    model = load_model()

    # load the test set
    # datas, testLoader = build_dataset_test(dataset, num_workers)

    if not best or is_inference:
        if os.path.isfile(checkpoint_path):
            print("=====> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            # print(f"checkpoint: {checkpoint}")
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(checkpoint_path))
            raise FileNotFoundError("no checkpoint found at '{}'".format(checkpoint_path))

        if is_inference: 
            inference(model)
        else:
            
            print("=====> beginning validation")
            print("validation set length: ", len(testLoader))
            mIOU_val, per_class_iu,cm = test(testLoader, model,eval = True)
            print("=====> Mean IoU: ", str(mIOU_val))
            print("=====> Per class IoU", str(per_class_iu))
            sn.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],annot=True,annot_kws={"size": 16},fmt=".2f",xticklabels=labels_names, yticklabels=labels_names)
            plt.title("Confusion matrix")
            plt.xlabel("Predictions")
            plt.ylabel("True labels")

    # Get the best test result among the last 10 model records.
    else:
        if os.path.isfile(checkpoint_path):
            dirname, basename = os.path.split(checkpoint_path)
            epoch = int(os.path.splitext(basename)[0].split('_')[1])
            mIOU_val = []
            per_class_iu = []
            for i in range(epoch - 9, epoch+1):
                basename = 'model_' + str(i) + '.pth'
                resume = os.path.join(dirname, basename)
                checkpoint = torch.load(resume)
                model.load_state_dict(checkpoint['model'])
                print("=====> beginning test the " + basename)
                #print("validation set length: ", len(testLoader))
                mIOU_val_0, per_class_iu_0,cm = test(testLoader, model,eval=True)
                mIOU_val.append(mIOU_val_0)
                per_class_iu.append(per_class_iu_0)

            index = list(range(epoch - 9, epoch+1))[np.argmax(mIOU_val)]
            print("The best mIoU among the last 10 models is", index)
            print(mIOU_val)
            per_class_iu = per_class_iu[np.argmax(mIOU_val)]
            mIOU_val = np.max(mIOU_val)
            print("=====> Best Mean IoU: ", str(mIOU_val))
            print("=====> Best Per class IoU", str(per_class_iu))
            sn.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],annot=True,annot_kws={"size": 16},fmt=".2f",xticklabels=labels_names, yticklabels=labels_names)
            plt.title("Confusion matrix")
            plt.xlabel("Predictions")
            plt.ylabel("True labels")
            

        else:
            print("=====> no checkpoint found at '{}'".format(checkpoint_path))
            raise FileNotFoundError("no checkpoint found at '{}'".format(checkpoint_path))

    # Save the result
    if not best:
        model_path = os.path.splitext(os.path.basename(checkpoint_path))
        logFile = 'test_' + model_path[0] + '.txt'
        logFileLoc = os.path.join(os.path.dirname(checkpoint_path), logFile)
    else:
        logFile = 'test_' + 'best' + str(index) + '.txt'
        logFileLoc = os.path.join(os.path.dirname(checkpoint_path), logFile)

evaluate_model()