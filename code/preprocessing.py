
# Preprocessing script

import os
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tensorflow import keras
#from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
#import SimpleITK as sitk
from medpy.io import load


data_fold = r'D:\STOIC2021\data\mha'
metadata_fold = r'D:\STOIC2021\metadata\reference.csv'

path_to_label = r'D:\STOIC2021\metadata\reference.csv'
label_file = pd.read_csv(path_to_label, index_col=0)

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 150 #mean depth is 433
    desired_width = 150
    desired_height = 150
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = desired_depth / current_depth
    width_factor = desired_width / current_width
    height_factor = desired_height / current_height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor),mode='nearest', order=3)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume, header =load(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

def get_label(image):
    
    single_ID = image.replace(".mha","")
    ID_row = label_file.loc[label_file['PatientID'] == int(single_ID)]
    lbl_cvd = int(ID_row.iloc[:,1])
    return lbl_cvd

def get_ID(path):
    path = path.replace("D:\STOIC2021\data\mha\\", "")
    single_ID, f_ext = os.path.splitext(path)
    #print('ID: ',single_ID)
    return single_ID

def list_full_paths(dir):
    return [os.path.join(dir, file) for file in os.listdir(dir)]

paths = [os.path.join(data_fold, file) for file in os.listdir(data_fold)]
 
def process_data():
    
    #data = np.array([process_scan(p) for p in tqdm(paths)])
    labels = np.array([get_label(ima) for ima in tqdm(os.listdir(data_fold))])
    #np.save(r'C:\Users\hugo.morvan\Desktop\STOIC2021\Preprocessed_data-HD\dataHD.npy',data, allow_pickle=True, fix_imports=True)
    #print("saved data")
    np.save(r'D:\STOIC2021\Preprocessed_data-HD\labelsHD.npy',labels, allow_pickle=True, fix_imports=True)
    print("saved labels")
        
process_data()

def indiv_process_data():
    for p in tqdm(paths):
        data = np.array(process_scan(p))
        ID = get_ID(p)
        np.save(r'D:\STOIC2021\PreProcessed_data-HD\data\\'+ID, data, allow_pickle=True, fix_imports=True)

#indiv_process_data()

def get_info():
    pos, sev = 0, 0
    #print(label_file(1,1))
    for i in tqdm(range(2000)):
        pos = pos + label_file.iat[i,0]
    print(pos)
    for i in tqdm(range(2000)):
        sev = sev + label_file.iat[i,1]
    print(sev)
    """The dataset contains 2000 patients, including 1205 Covid+ which include 301 severe cases"""


