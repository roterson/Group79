#!/usr/bin/python
#coding:utf-8
from PIL import Image
import os
import cv2

DATADIR="/home/roterson/Desktop/DL/Xray/Normal/"

path=os.path.join(DATADIR) 

img_list=os.listdir(path)
ind=0
for i in img_list:

    #img_array=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
    img_array=Image.open(os.path.join(path,i))
    new_array=img_array.convert("RGB") 
    #new_array=cv2.cvtColor( img_array, cv2.COLOR_GRAY2RGB )
    img_name=str(ind)+'.jpg'

    save_path='/home/roterson/Desktop/DL/Xray/NormalRGB/'+str(ind)+'.jpg'
    ind=ind+1

    #cv2.imwrite(save_path,new_array)
    new_array.save(save_path)

