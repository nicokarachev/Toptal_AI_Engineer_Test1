#if you use Google Colab or have wget, you can (but do not have to) load the dataset with:
# ! wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1GIR3hdXVVr0uYWXdZdb5XKDdJUFlFeR4' -O 'Test Pipeline T1.zip'

import zipfile
import os
import numpy as np
import cv2
import io
import pandas as pd
from PIL import Image
from typing import Tuple, List, Generator       

image_extensions = {'.jpg', '.jpeg',  '.png', '.bmp', '.tiff', '.tif'}

def make_square(image: np.ndarray, target_size: int = 224) -> np.ndarray:

    h, w, c = image.shape
    
    if h > w:
        pad_left = (h-w) // 2
        pad_right = h - w - pad_left
        padded = np.pad( image, ( (0, 0), (pad_left, pad_right), (0, 0)), mode = 'constant' )
        resized = cv2.resize(padded, (target_size, target_size))
    elif w > h:
        pad_top = (w-h) // 2
        pad_bottom = w - h - pad_top
        padded = np.pad( image, ((pad_top, pad_bottom), (0,0), (0,0)), mode= 'constant' )
        resized = cv2.resize(padded, (target_size, target_size))
    else:
        resized = cv2.resize(image, (target_size, target_size))

    return resized

def extract_label_from_path(file_path: str) -> str:

    path_parts = file_path.split('/')
    if len(path_parts) > 1:
        return path_parts[0]

    return "unknown"

def data_loader(zip_path: str, res_type: str = "train") -> Generator[Tuple[List[np.ndarray], List[str]], None, None]:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        
        image_files = [
            f for f in file_list
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]
        
        df = ""
        
        with zip_ref.open("crop.csv") as f:
            df = pd.read_csv(f)

        batch_size = 10 #len(image_files) / 5
        current_batch_images = []
        current_batch_labels = []

        for i, image_file in enumerate(image_files):
            try:
                with zip_ref.open(image_file) as file:
                    
                    rows = df[df["file_name"] == image_file]
                    
                    image_data = file.read()
                    pil_image = Image.open(io.BytesIO(image_data))

                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                        
                    if not rows.empty:
                        row = rows.iloc[0]  # get first row as Series
                        pil_image = pil_image.crop((row['x_start'], row['y_start'], row['x_end'], row['y_end']))
                    
                    image_array = np.array(pil_image)

                    square_image = make_square(image_array)

                    label = extract_label_from_path(image_file)

                    current_batch_images.append(square_image)
                    current_batch_labels.append(label)

                    if len(current_batch_images) >= batch_size:
                        yield np.array(current_batch_images), current_batch_labels
                        current_batch_images = []
                        current_batch_labels = []

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        if current_batch_images:
            yield np.array(current_batch_images), current_batch_labels

def score(data_loader, subset="training"):
    ds=None
    zip='Test Pipeline T1.zip'
    batch_images, batch_labels = next(data_loader(zip, subset))
    print('batch size', len(batch_images))
    print('Image dimensions: ', batch_images[0].shape)
    print("Classnames found: ",np.unique(np.array(batch_labels)))

    # test that data can be loaded infinitely
    unique_images = []
    try:
        for i in range(15):
            batch_images, batch_labels = next(data_loader(zip, subset))
            print(batch_labels)
            for batch_image in batch_images:
	    # if your images are returned as numpy arrays
                unique_images.append(batch_image.astype("uint8").ravel())
	    # if your images are returned as tensors:
	    # unique_images.append(batch_image.astype("uint8").ravel())
        print("Unique images loaded: ", len(np.unique(unique_images, axis=0)))            
    except Exception as e:
        print("Error loading 10 batches")


score(data_loader, 'training')

