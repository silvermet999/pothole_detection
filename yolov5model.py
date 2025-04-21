#!/usr/bin/env python
# coding: utf-8

# !git clone https://github.com/ultralytics/yolov5  # clone repo
# 

# In[3]:


get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().system('git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0')


# !pip install -qr requirements.txt  # install dependencies (ignore errors)
# 

# In[4]:


import torch

from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# !pip install roboflow
# 
# from roboflow import Roboflow
# rf = Roboflow(api_key="qsriDCqyPmufazKUFp0g")
# project = rf.workspace("project-ssayl").project("potholes-detection-d4rma")
# dataset = project.version(1).download("yolov5")
# 

# In[5]:


get_ipython().run_line_magic('cat', 'Potholes-Detection-1/data.yaml')


# In[7]:


import yaml
with open("Potholes-Detection-1/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])


# In[ ]:


import os
HOME = os.getcwd()


# In[13]:


{HOME}


# In[18]:


get_ipython().run_line_magic('cat', '/{HOME}/Potholes-Detection-1/data.yaml')


# In[19]:


from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


# In[24]:


get_ipython().run_cell_magic('writetemplate', '{HOME}/models/custom_yolov5s.yaml', "\n# parameters\nnc: {num_classes}  # number of classes\ndepth_multiple: 0.33  # model depth multiple\nwidth_multiple: 0.50  # layer channel multiple\n\n# anchors\nanchors:\n  - [10,13, 16,30, 33,23]  # P3/8\n  - [30,61, 62,45, 59,119]  # P4/16\n  - [116,90, 156,198, 373,326]  # P5/32\n\n# YOLOv5 backbone\nbackbone:\n  # [from, number, module, args]\n  [[-1, 1, Focus, [64, 3]],  # 0-P1/2\n   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4\n   [-1, 3, BottleneckCSP, [128]],\n   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8\n   [-1, 9, BottleneckCSP, [256]],\n   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16\n   [-1, 9, BottleneckCSP, [512]],\n   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32\n   [-1, 1, SPP, [1024, [5, 9, 13]]],\n   [-1, 3, BottleneckCSP, [1024, False]],  # 9\n  ]\n\n# YOLOv5 head\nhead:\n  [[-1, 1, Conv, [512, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 6], 1, Concat, [1]],  # cat backbone P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 13\n\n   [-1, 1, Conv, [256, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 4], 1, Concat, [1]],  # cat backbone P3\n   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)\n\n   [-1, 1, Conv, [256, 3, 2]],\n   [[-1, 14], 1, Concat, [1]],  # cat head P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)\n\n   [-1, 1, Conv, [512, 3, 2]],\n   [[-1, 10], 1, Concat, [1]],  # cat head P5\n   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)\n\n   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)\n  ]\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "%cd {HOME}/\n!python train.py --img 416 --batch 16 --epochs 100 --data Potholes-Detection-1/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache\n")


# In[29]:


get_ipython().run_line_magic('cd', '{HOME}')


# In[31]:


from utils.plots import plot_results  # plot results.txt as results.png
Image(filename=f'{HOME}/runs/train/yolov5s_results/results.png', width=1000) 


# In[33]:


print("GROUND TRUTH TRAINING DATA:")
Image(filename=f'{HOME}/runs/train/yolov5s_results/val_batch0_labels.jpg', width=900)


# In[34]:


print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename=f'{HOME}/runs/train/yolov5s_results/train_batch0.jpg', width=900)


# In[36]:


get_ipython().run_line_magic('ls', 'runs/')


# In[37]:


get_ipython().run_line_magic('ls', 'runs/train/yolov5s_results/weights')


# In[42]:


get_ipython().system('python detect.py --weights {HOME}/runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source Potholes-Detection-1/test/images')


# In[ ]:




