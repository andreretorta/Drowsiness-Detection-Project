# %% [markdown]
# # 1. Install and Import Dependencies

# %%
!pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# %%
!git clone https://github.com/ultralytics/yolov5

# %%
!cd yolov5 & pip install -r requirements.txt

# %%
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

# %% [markdown]
# # 2. Load Model

# %%
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# %%
model

# %% [markdown]
# # 3. Make Detections with Images

# %%
img = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg/800px-Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg'

# %%
results = model(img)
results.print()

# %%
%matplotlib inline 
plt.imshow(np.squeeze(results.render()))
plt.show()

# %%
results.render()

# %% [markdown]
# # 4. Real Time Detections

# %%
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

## %% [markdown]
## # 5. Train from scratch

## %%
#import uuid   # Unique identifier
#import os
#import time

## %%
#IMAGES_PATH = os.path.join('data', 'images') #/data/images
#labels = ['awake', 'drowsy']
#number_imgs = 5

## %%
#cap = cv2.VideoCapture(0)
## Loop through labels
#for label in labels:
#    print('Collecting images for {}'.format(label))
#    time.sleep(5)
    
    # Loop through image range
#    for img_num in range(number_imgs):
#        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        # Webcam feed
#        ret, frame = cap.read()
        
#        # Naming out image path
#        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
#        # Writes out image to file 
#        cv2.imwrite(imgname, frame)
        
        # Render to the screen
#        cv2.imshow('Image Collection', frame)
        
        # 2 second delay between captures
#        time.sleep(2)
        
#        if cv2.waitKey(10) & 0xFF == ord('q'):
#            break
#cap.release()
#cv2.destroyAllWindows()

## %%
#print(os.path.join(IMAGES_PATH, labels[0]+'.'+str(uuid.uuid1())+'.jpg'))

## %%
#for label in labels:
#    print('Collecting images for {}'.format(label))
#    for img_num in range(number_imgs):
#        print('Collecting images for {}, image number {}'.format(label, img_num))
#        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
#        print(imgname)   

## %%
#!git clone https://github.com/tzutalin/labelImg

# %%
#!pip install pyqt5 lxml --upgrade
#!cd labelImg && pyrcc5 -o libs/resources.py resources.qrc

## %%
#!cd yolov5 && python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt --workers 2

## %% [markdown]
# ## 6. Load Custom Model

## %%
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp15/weights/last.pt', force_reload=True)

## %%
#img = os.path.join('data', 'images', 'awake.c9a24d48-e1f6-11eb-bbef-5cf3709bbcc6.jpg')

## %%
#results = model(img)

## %%
#results.print()

## %%
#%matplotlib inline 
#plt.imshow(np.squeeze(results.render()))
#plt.show()

## %%
#cap = cv2.VideoCapture(0)
#while cap.isOpened():
#    ret, frame = cap.read()
    
#    # Make detections 
#    results = model(frame)
    
#    cv2.imshow('YOLO', np.squeeze(results.render()))
    
#    if cv2.waitKey(10) & 0xFF == ord('q'):
#        break
#cap.release()
#cv2.destroyAllWindows()

## %%



