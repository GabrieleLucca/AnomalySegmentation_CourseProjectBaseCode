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
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + "erfnet_pretrained.pth"

    modelpath_prune = args.loadDir + args.loadModel 
    weightspath_prune  = args.loadDir + "erfnet_PRUNOOOO.pth"

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

    
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    model_pruned = load_my_state_dict(model, torch.load(weightspath_prune, map_location=lambda storage, loc: storage))

    print('Model and weights LOADED successfully')
    




    # Specifica la dimensione dell'input (batch_size, channels, height, width)
    input_size = (6, 3, 512, 256)

    # Sposta il modello sulla GPU se disponibile
    if torch.cuda.is_available():
        model = model.cuda()

    # Passa un input di esempio attraverso il modello
    input_data = torch.randn(input_size)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    output = model(input_data)

    # Conta il numero di operazioni floating point effettuate
    total_ops = 0
    for name, param in model.named_parameters():
        total_ops += torch.prod(torch.tensor(param.shape))
    total_ops *= 2  # Moltiplica per 2 perché tipicamente ogni operazione richiede 2 operazioni floating point (moltiplicazione e somma)

    print(f"FLOPS: {total_ops}")

    # Calcolo dei parametri per il modello non prune
    summary(model, input_size=(3, 512, 256))

    # Calcolo dei FLOPS per il modello prune

    # Specifica la dimensione dell'input (batch_size, channels, height, width)
    
    # Sposta il modello sulla GPU se disponibile
    if torch.cuda.is_available():
        model_pruned = model_pruned.cuda()

    # Passa un input di esempio attraverso il modello
    input_data = torch.randn(input_size)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    output = model_pruned(input_data)

    # Conta il numero di operazioni floating point effettuate
    total_ops = 0
    for name, param in model_pruned.named_parameters():
        total_ops += torch.prod(torch.tensor(param.shape))
    total_ops *= 2  # Moltiplica per 2 perché tipicamente ogni operazione richiede 2 operazioni floating point (moltiplicazione e somma)

    print(f"FLOPS: {total_ops}")

    # Calcolo dei parametri per il modello prune
    summary(model_pruned, input_size=(3, 512, 256))

    # Stampa dei nomi dei moduli
    print("Nomi dei moduli del modello non pruned:")
    for name, module in model.named_modules():
        print(name)

    print("\nNomi dei moduli del modello pruned:")
    for name, module in model_pruned.named_modules():
        print(name)

    # Conteggio dei parametri
    num_params_model = sum(p.numel() for p in model.parameters())
    num_params_model_pruned = sum(p.numel() for p in model_pruned.parameters())
    print("\nNumero di parametri del modello non pruned:", num_params_model)
    print("Numero di parametri del modello pruned:", num_params_model_pruned)

    # Confronto dei pesi
    for param1, param2 in zip(model.parameters(), model_pruned.parameters()):
        if not torch.equal(param1, param2):
            print("I pesi dei due modelli sono diversi.")
            break
    else:
        print("I pesi dei due modelli sono uguali.")
    

    #modules_to_quantize = {nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}
        
    #quantized_model = torch.quantization.quantize_dynamic(model, modules_to_quantize, dtype=torch.qint8)

    #quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        
    # # Stampare i tipi di dati dei parametri prima della quantizzazione
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.dtype}")

    # # Utilizza la quantizzazione dinamica solo per i moduli lineari
    # quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # # Stampare i tipi di dati dei parametri dopo la quantizzazione
    # for name, param in quantized_model.named_parameters():
    #     print(f"{name}: {param.dtype}")

    # # Esegui la summary della rete
    # summary(quantized_model, input_size=(3, 512))
   

#     model.eval()
    
#     for path in glob.glob(os.path.expanduser(str(args.input[0]))):
#         print(path)
#         images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
#         images = images.permute(0, 3, 1, 2)

#         images = images.cuda()

#         if modelname == "bisenetv1":
#           result = model(images)[0].squeeze(0)
#         else:
#           result = model(images).squeeze(0)


#         if args.method == 'msp':
#             softmax_probs = torch.nn.functional.softmax(result.squeeze(0) / float(args.temperature), dim=0)
#             anomaly_result = 1.0 - (np.max(softmax_probs.data.cpu().numpy(), axis=0))  
#         elif args.method == 'maxlogit':
#             anomaly_result = -torch.max(result, dim=0)[0]
#             anomaly_result = anomaly_result.data.cpu().numpy()
#         elif args.method == 'maxentropy':
#             anomaly_result = torch.div(
#                 torch.sum(-funct.softmax(result, dim=0) * funct.log_softmax(result, dim=0), dim=0),
#                 torch.log(torch.tensor(result.size(0))),
#             )
#             anomaly_result = anomaly_result.data.cpu().numpy()
#         elif args.method == 'void':
#             anomaly_result = funct.softmax(result, dim=0)[-1].data.cpu().numpy()
#         else:
#             print("Unknown method")

#         pathGT = path.replace("images", "labels_masks")                
#         if "RoadObsticle21" in pathGT:
#            pathGT = pathGT.replace("webp", "png")
#         if "fs_static" in pathGT:
#            pathGT = pathGT.replace("jpg", "png")                
#         if "RoadAnomaly" in pathGT:
#            pathGT = pathGT.replace("jpg", "png")  

#         mask = Image.open(pathGT)
#         ood_gts = np.array(mask)

#         if "RoadAnomaly" in pathGT:
#             ood_gts = np.where((ood_gts==2), 1, ood_gts)
#         if "LostAndFound" in pathGT:
#             ood_gts = np.where((ood_gts==0), 255, ood_gts)
#             ood_gts = np.where((ood_gts==1), 0, ood_gts)
#             ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

#         if "Streethazard" in pathGT:
#             ood_gts = np.where((ood_gts==14), 255, ood_gts)
#             ood_gts = np.where((ood_gts<20), 0, ood_gts)
#             ood_gts = np.where((ood_gts==255), 1, ood_gts)

#         if 1 not in np.unique(ood_gts):
#             continue              
#         else:
#              ood_gts_list.append(ood_gts)
#              anomaly_score_list.append(anomaly_result)
#         del result, anomaly_result, ood_gts, mask
#         torch.cuda.empty_cache()

#     file.write( "\n")

#     ood_gts = np.array(ood_gts_list)
#     anomaly_scores = np.array(anomaly_score_list)

#     ood_mask = (ood_gts == 1)
#     ind_mask = (ood_gts == 0)

#     ood_out = anomaly_scores[ood_mask]
#     ind_out = anomaly_scores[ind_mask]

#     ood_label = np.ones(len(ood_out))
#     ind_label = np.zeros(len(ind_out))
    
#     val_out = np.concatenate((ind_out, ood_out))
#     val_label = np.concatenate((ind_label, ood_label))

#    # print("Val out and val label ",val_out.shape, val_label.shape)

#     prc_auc = average_precision_score(val_label, val_out)
#     fpr = fpr_at_95_tpr(val_out, val_label)

#     print(f'Model: {modelname.upper()}')
#     print(f'Method: {args.method}')
#     if args.method == 'msp':
#         print(f'Temperature: {args.temperature}')
#     print(f'AUPRC score: {prc_auc*100.0}')
#     print(f'FPR@TPR95: {fpr*100.0}')

#     file.write(
#         f'Model: {modelname.upper()}    Method: {args.method}   {f"   Temperature: {args.temperature}" if args.method == "msp" else ""}    AUPRC score: {prc_auc * 100.0}   FPR@TPR95: {fpr * 100.0}'
#     )
#     file.close()
if __name__ == '__main__':
    main()