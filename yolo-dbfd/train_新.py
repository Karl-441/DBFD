import os
os.system("python train.py --data fire_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 100 --batch-size 2 --device cpu")