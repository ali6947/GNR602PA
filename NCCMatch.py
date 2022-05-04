
import argparse
import numpy as np
import cv2
import pylab as pl


def Normalised_Cross_Correlation(region_of_interest, target_area):
    
    # mean_roi=np.mean(region_of_interest)
    # mean_target=np.mean(target_area)
    # region_of_interest=region_of_interest-mean_roi
    # target_area=target_area - mean_target

    correlation = np.sum(region_of_interest * target_area)
    normalisation = np.sqrt( (np.sum(region_of_interest ** 2))) * np.sqrt(np.sum(target_area ** 2))

    return correlation/normalisation


def Template_Matcher(img, target):
    
    height, width = img.shape
    target_height, target_width = target.shape
      
    img = np.array(img, dtype="int")
    target = np.array(target, dtype="int")
    NccValue = np.zeros((height-target_height, width-target_width))

    
    for h in range(height-target_height):
        for w in range(width-target_width):           
            region_of_interest = img[h : h+target_height, w : w+target_width]
            NccValue[h, w] = Normalised_Cross_Correlation(region_of_interest, target)

    
    best_Y,best_X=np.unravel_index(np.argmax(NccValue, axis=None), NccValue.shape)
    return (best_X,best_Y),NccValue[best_Y,best_X]


if __name__ == '__main__':


    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to input image")
    ap.add_argument("-t", "--template", required = True, help = "Path to template")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"], 0)
    template = cv2.imread(args["template"], 0)

    height, width = template.shape
    
    matched_coords,norm_cross_corr = Template_Matcher(image, template)
    
    print(f'normalised cross correlation: {norm_cross_corr:.5f}')
    cv2.rectangle(image, matched_coords, (matched_coords[0] + width, matched_coords[1] + height), 0, 3)

    pl.imshow(image)
    pl.title(f'normalised cross correlation: {norm_cross_corr:.5f}')
    pl.show()
