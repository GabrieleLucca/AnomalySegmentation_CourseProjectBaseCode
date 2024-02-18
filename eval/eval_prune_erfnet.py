# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from ENet import ENet
from BiSeNetV1 import BiSeNetV1
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as funct
from torchvision.transforms import Resize
from iouEval import iouEval
from temperature_scaling import ModelWithTemperature
from thop import profile
from torchsummary import summary
import torch.quantization


seed = 42


# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--method', type=str, default='msp')
    parser.add_argument('--model', type=str, default='erfnet')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []
    
    
    modelname = 'erfenet'
    model=ERFNet(NUM_CLASSES)
    model_pruned=ERFNet(NUM_CLASSES)
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + "erfnet_pretrained.pth"

    modelpath_prune = args.loadDir + args.loadModel 
    weightspath_prune  = args.loadDir + "erfnet_pruning.pth"

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    
    print ("Loading model: " + modelpath_prune)
    print ("Loading weights: " + weightspath_prune)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model_pruned = load_my_state_dict(model, torch.load(weightspath_prune, map_location=lambda storage, loc: storage))

    print('Model and weights LOADED successfully')
 
    input_size = (6, 3, 512, 256) 
 
    if torch.cuda.is_available(): 
        model_pruned = model_pruned.cuda() 

    input_data = torch.randn(input_size) 
    if torch.cuda.is_available(): 
        input_data = input_data.cuda() 
    output = model_pruned(input_data) 

    total_ops = 0 
    for name, param in model_pruned.named_parameters(): 
        total_ops += torch.prod(torch.tensor(param.shape)) 
    total_ops *= 2  
 
    print(f"FLOPS: {total_ops}") 
 
    summary(model_pruned, input_size=(3, 512, 256)) 


#method 1
    #model.eval()

    #dtype = torch.qint8

    #quantized_model = torch.quantization.quantize(model, {'': dtype})

   # summary(quantized_model, input_size=(3, 512, 256))

#method 2 
    # model_int8 = torch.ao.quantization.quantize_dynamic(
    # model,  
    # {torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d},  # a set of layers to dynamically quantize
    # dtype=torch.qint8)  # the target dtype for quantized weights
    
    # for name, param in model.named_parameters():
    #       print(f"{name}: {param.dtype}")

    # print("---------------------------------------------------")

    # for name, param in model_int8.named_parameters():
    #       print(f"{name}: {param.dtype}")
    # summary(model_int8, input_size=(3, 512, 256))
    
    #modules_to_quantize = {torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d}
        
    #quantized_model = torch.quantization.quantize_dynamic(model_pruned, modules_to_quantize, dtype=torch.qint8)

    #summary(quantized_model, input_size=(3, 512, 256))
        
#method 3 
    quantized_model = torch.quantization.quantize_dynamic(model_pruned, {torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d}, dtype=torch.qint8)
    #summary(quantized_model, input_size=(3, 512, 256))

#method 4 static quantization
    #quantized_model = torch.quantization.QuantStub()(model)
    #quantized_model.load_state_dict(model.state_dict())
    #quantized_model = torch.quantization.convert(quantized_model)

    #for name, param in model.named_parameters():
    #      print(f"{name}: {param.dtype}")

   # print("---------------------------------------------------")
    
   # for name, param in quantized_model.named_parameters():
     #     print(f"{name}: {param.dtype}")

    # summary(quantized_model, input_size=(3, 512))
   
if __name__ == '__main__':
    main()