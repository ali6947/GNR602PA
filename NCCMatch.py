
import argparse
import numpy as np
import cv2
import pylab as pl
import sys
from tqdm import tqdm
import torch
from torchvision.ops import nms


def Normalised_Cross_Correlation(region_of_interest, target_area):
    
    mean_roi = np.mean(region_of_interest)
    mean_target = np.mean(target_area)
    region_of_interest = region_of_interest-mean_roi
    target_area = target_area - mean_target

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

    
    for h in tqdm(range(height-target_height), desc='Finding Matches: '):
        for w in range(width-target_width):           
            region_of_interest = img[h : h+target_height, w : w+target_width]
            NccValue[h, w] = Normalised_Cross_Correlation(region_of_interest, target)
    
    if single:
        best_Y,best_X=np.unravel_index(np.argmax(NccValue, axis=None), NccValue.shape)
        return (best_X,best_Y),NccValue[best_Y,best_X]
    else: 
        best_Y,best_X=np.where(NccValue>thresh)    
        return (best_X,best_Y),NccValue[best_Y,best_X]


if __name__ == '__main__':


    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to input image")
    ap.add_argument("-t", "--template", required = True, help = "Path to input template")
    ap.add_argument("-g", "--grayscale",action="store_true", help = "convert image to grayscale")
    ap.add_argument("-thr", "--ncc_threshold", help = "",default=0.9)
    ap.add_argument("-s", "--single", help = "single detection",action="store_true")
    ap.add_argument("-o", "--output", help = "output file location",default="")
    ap.add_argument("-iou", "--iou_threshold", help = "IoU Threshold for match selection",default=0.2)
    ap.add_argument("-ni", "--noisy_image",action="store_true", help = "Add additive Gaussian noise to image")
    ap.add_argument("-nt", "--noisy_template",action="store_true", help = "Add additive Gaussian noise to template")
    ap.add_argument("-stdi", "--std_image", help = "Standard Deviation for the noise to be added to the image",default=0.5)
    ap.add_argument("-stdt", "--std_template", help = "Standard Deviation for the noise to be added to the template",default=0.5)
    ap.add_argument("-meani", "--mean_image", help = "Mean for the noise to be added to the image",default=0)
    ap.add_argument("-meant", "--mean_template", help = "Mean for the noise to be added to the template",default=0)

    args = vars(ap.parse_args())
    threshold=float(args['ncc_threshold'])
    iou_thresh=float(args['iou_threshold'])
    
    try:
        if args['grayscale']:
            image = cv2.imread(args["image"], 0)
        else:
            image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
    except:
        print("Bad Image file")
        sys.exit(1)

    try:
        if args['grayscale']:
            template = cv2.imread(args["template"], 0)
        else:
            template = cv2.imread(args["template"], cv2.IMREAD_COLOR)

    except:
        print("Bad Template file")
        sys.exit(1)

    if template is None:
        print("Bad Template file")
        sys.exit(1)

    if image is None:
        print("Bad Image file")
        sys.exit(1)

    if args['noisy_image']:
        image=image+np.float32(np.random.normal(scale=float(args['std_image']),size=image.shape))

    if args['noisy_template']:
        template=template+np.float32(np.random.normal(scale=float(args['std_template']),size=template.shape))

    try:
        height, width,_ = template.shape
    except:
        height, width = template.shape
    
    matched_coords,Corrs = Template_Matcher(image, template, threshold,args['single']) # returns only highest NCC corrs and value when single is true
    
    if args['single']:
        print(f'Found maxmimum Normalised Cross Correlation: {Corrs:.5f}')
        cv2.rectangle(image, (matched_coords[0],matched_coords[1]), (matched_coords[0] + width, matched_coords[1] + height), 0, 3)

    elif matched_coords[0].size!=0:

        boxes=torch.tensor([[a,b,a + width, b + height] for a,b in zip(matched_coords[0],matched_coords[1])],dtype=torch.float64)
        scores=torch.tensor(Corrs,dtype=torch.float64)
        box_idx=nms(boxes,scores,iou_thresh)
        if len(box_idx)>0:
            print(f'\nFound {len(box_idx)} matches with the following Normalised Cross Correlations:')
            for sc in scores[box_idx]:
                print(sc.item())
        else:
            print("No matches found")
        for idx in box_idx:
            a=matched_coords[0][idx]
            b=matched_coords[1][idx]
            
            cv2.rectangle(image, (a,b), (a + width, b + height), 0, 3)

    else:
        print("No matches found")

    image = cv2.cvtColor(np.clip(np.round(image),0,255), cv2.COLOR_BGR2RGB)
    image=image.astype(np.uint8)

    pl.imshow(image)
    pl.title("Detected Matches")
    if args['output'] != '':
        pl.savefig(args['output'])
    pl.show()
