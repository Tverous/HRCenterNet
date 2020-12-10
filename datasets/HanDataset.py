import torch
import torchvision
from PIL import Image
import pandas as pd
import numpy as np


def dataset_generator(args, data_dir, data_list, crop_size, output_size):
    
    train_tx = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
#       torchvision.transforms.Normalize((0.5, 0.5, 0.5), 
#                               (0.5, 0.5, 0.5)),
      ])

    data_set = HanDataset(data_list, data_dir, args.crop_ratio, crop_size, output_size, transform=train_tx)
    
    return data_set
    

class HanDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list, image_path, crop_ratio, crop_size, output_size, transform=None):
        
        self.data_list = data_list
        self.image_path = image_path
        self.transform = transform
        self.crop_ratio = crop_ratio
        self.crop_size = crop_size
        self.output_size = output_size
    
    def __len__(self):
        
        return len(self.data_list)
    
    def __getitem__(self,idx):
        
        file_name = self.data_list[idx][0]
        print(file_name)
        img = Image.open(self.image_path + file_name).convert('RGB')
        
        pic_width, pic_height=img.size
        output_width = self.output_size
        output_height = self.output_size
        label = np.zeros((output_height,output_width,6))      
        
        if pic_height < self.crop_size or pic_width < self.crop_size:
            img = torchvision.transforms.functional.resize(img, (self.crop_size, self.crop_size))
        
        if np.random.randint(0, 101) < self.crop_ratio * 100 and pic_height > self.crop_size and pic_width > self.crop_size:
            
            new_h, new_w = (self.crop_size, self.crop_size)
            top = np.random.randint(0, pic_height - new_h)
            left = np.random.randint(0, pic_width - new_w)
            img = img.crop((left, top, left + new_w, top + new_h))
        else:
            new_w, new_h = img.size
            top = 0
            left = 0
            img = torchvision.transforms.functional.resize(img, (self.crop_size, self.crop_size))
            
        for annotation in self.data_list[idx][1]: 

            if annotation[1] < left or annotation[2] < top or annotation[1] >= (left + new_w) or annotation[2] >= (top + new_h):
                continue
                
            # ignore the character that exceed to much to the boundary
            if (annotation[1] + (annotation[3] / 2)) >= (left + new_w) or (annotation[2] + (annotation[4] / 2)) >= (top + new_h): 
                continue
            
            x_c = (annotation[1] - left) * (output_width / new_w)
            y_c = (annotation[2] - top) * (output_height / new_h)
            width = annotation[3] * (output_width / new_w)
            height = annotation[4] * (output_height / new_h)
            
            heatmap=((np.exp(-(((np.arange(output_width) - x_c)/(width/10))**2)/2)).reshape(1,-1)
                    *(np.exp(-(((np.arange(output_height) - y_c)/(height/10))**2)/2)).reshape(-1,1))
          
            label[:,:,0]=np.maximum(label[:,:,0],heatmap[:,:])
            label[int(y_c//1), int(x_c//1), 1] = 1
            label[int(y_c//1),int(x_c//1),2]=y_c%1
            label[int(y_c//1),int(x_c//1),3]=x_c%1
            label[int(y_c//1), int(x_c//1), 4] = height / self.output_size 
            label[int(y_c//1), int(x_c//1), 5] = width / self.output_size
            
        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'labels': label}

        return sample