# GNR602 PA - Normalised Cross Correlation based Template Matching

## Requirements
Run the following commands for installation: <br />
`pip3 install --upgrade pip` <br />
`pip3 install opencv-python` <br />

## Code Usage <br />
Our code written in Python3 has the following command line arguments: <br />
`-i` OR `--image`: Path to input image <br />
`-t` OR `--template`: Path to template <br />
`-g` OR `--grayscale`: If flag is given, image will be converted to grayscale if multichannel <br />
`-thr` OR `--threshold`: NCC Threshold for match detection (Default: 0.9) <br />
`-s` OR `--single`: If flag is given, only single max NCC value detection is made, else multiple <br />
`-o` OR `--output`: Path to output file. If not given, image is not saved <br />
`-iou` OR `--iou_threshold`: IoU Threshold for match selection" (Default: 0.2) <br />
`-ni` OR `--noisy_image`: If provided, add additive Gaussian noise to image'
`-nt` OR `--noisy_template`: If provided, add additive Gaussian noise to template
`-stdi` OR `--std_image`: Standard Deviation for the noise to be added to the image (Default: 0.5)
`-stdt` OR `--std_template`: Standard Deviation for the noise to be added to the template (Default: 0.5)
`-meani` OR `--mean_image`: Mean for the noise to be added to the image (Default: 0)
`-meant` OR `--mean_template`: Mean for the noise to be added to the template (Default: 0)


## Sample Use Command for Grayscale image <br />
`python3 NCCMatch.py -i Images/brain.jpg -t Images/brain_target.jpg -thr 0.9 -o -g result.png -iou 0.2`

## Sample Use Command for Colour image <br />
`python3 NCCMatch.py -i Images/icons.png -t Images/icons_target.png -thr 0.9 -o result.png -iou 0.2`
