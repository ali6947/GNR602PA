# GNR602 PA1 - Normalised Cross Correlation based Template Matching

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

## Sample Usa Command <br />
`python3 NCCMatch.py -i Images/brain.jpg -t Images/brain_target.jpg -thr 0.9 -o result.png -iou 0.2`