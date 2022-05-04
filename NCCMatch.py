
import argparse
import numpy as np
import cv2
import pylab as pl
import sys
from tqdm import tqdm
import torch
from torchvision.ops import nms


def Normalised_Cross_Correlation(region_of_interest, target_area):
    
    # mean_roi=np.mean(region_of_interest)
    # mean_target=np.mean(target_area)
    # region_of_interest=region_of_interest-mean_roi
    # target_area=target_area - mean_target

    correlation = np.sum(region_of_interest * target_area)
    normalisation = np.sqrt( (np.sum(region_of_interest ** 2))) * np.sqrt(np.sum(target_area ** 2))

    return correlation/normalisation


def Template_Matcher(img, target,thresh=0.99,single=False):
    """
    img: Image to search over as a numpy array
    target: Object to serach as a numpy array
    thresh: minimum value of NCC for to detect a match, used only when single is False 
    single: Detect multiple mathces when False or only 1 top match otherwise.

    Return values:
    X_coords, Y_coords, NCC Values  (all as numpy arrays)
    
    # returns only highest NCC location and value when single is true
    """

    try:
        height, width,_ = img.shape
    except:
        height, width = img.shape

    try:
        target_height, target_width,_ = target.shape
    except:
        target_height, target_width = target.shape
      
    img = np.array(img, dtype="int")
    target = np.array(target, dtype="int")
    NccValue = np.zeros((height-target_height, width-target_width))

    
    for h in tqdm(range(height-target_height)):
        for w in range(width-target_width):           
            region_of_interest = img[h : h+target_height, w : w+target_width]
            NccValue[h, w] = Normalised_Cross_Correlation(region_of_interest, target)
    # print(np.min(NccValue))
    # print(np.unravel_index(np.argmax(NccValue, axis=None), NccValue.shape),best_Y,best_X)
    # best_Y,best_X=np.unravel_index(np.argmax(NccValue, axis=None), NccValue.shape)
    print(np.sort(NccValue,axis=None))
    # print(np.argsort(NccValue,axis=None))
    if single:
        best_Y,best_X=np.unravel_index(np.argmax(NccValue, axis=None), NccValue.shape)
        return (best_X,best_Y),NccValue[best_Y,best_X]
    else: 
        best_Y,best_X=np.where(NccValue>thresh)    
        return (best_X,best_Y),NccValue[best_Y,best_X]


if __name__ == '__main__':


    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to input image")
    ap.add_argument("-t", "--template", required = True, help = "Path to template")
    ap.add_argument("-g", "--grayscale",action="store_true", help = "convert image to grayscale")
    ap.add_argument("-thr", "--ncc_threshold", help = "NCC Threshold for match detection",default=0.99)
    ap.add_argument("-s", "--single", help = "single detection",action="store_true")
    ap.add_argument("-o", "--output", help = "output file location",default="result.png")
    ap.add_argument("-iou", "--iou_threshold", help = "IoU Threshold for match selection",default=0.2)
    args = vars(ap.parse_args())
    threshold=float(args['ncc_threshold'])
    iou_thresh=float(args['iou_threshold'])

    if args['grayscale']:
        image = cv2.imread(args["image"], 0)
    else:
        image = cv2.imread(args["image"], cv2.IMREAD_COLOR)

    if args['grayscale']:
        template = cv2.imread(args["template"], 0)
    else:
        template = cv2.imread(args["template"], cv2.IMREAD_COLOR)

    if template is None:
        print("Template file not found")
        sys.exit(1)

    if image is None:
        print("Image file not found")
        sys.exit(1)


    try:
        height, width,_ = template.shape
    except:
        height, width = template.shape
    
    matched_coords,Corrs = Template_Matcher(image, template, threshold,args['single']) # returns only highest NCC corrs and value when single is true
    
    if args['single']:
        print(f'normalised cross correlation: {Corrs:.5f}')
        cv2.rectangle(image, (matched_coords[0],matched_coords[1]), (matched_coords[0] + width, matched_coords[1] + height), 0, 3)

    else:
        boxes=torch.tensor([[a,b,a + width, b + height] for a,b in zip(matched_coords[0],matched_coords[1])],dtype=torch.float64)
        scores=torch.tensor(Corrs,dtype=torch.float64)
        box_idx=nms(boxes,scores,iou_thresh)

        for idx in box_idx:
            a=matched_coords[0][idx]
            b=matched_coords[1][idx]
            print(a,b)
            cv2.rectangle(image, (a,b), (a + width, b + height), 0, 3)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pl.imshow(image)
    pl.title("Detected Matches")
    pl.savefig(args['output'])
    pl.show()
