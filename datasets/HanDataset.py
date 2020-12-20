import torch
import torchvision
import numpy as np
from PIL import Image

def dataset_generator(data_dir, data_list, crop_size, crop_ratio, output_size, train=False):
    
    train_tx = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
      ])

    data_set = HanDataset(data_list, data_dir, crop_ratio, crop_size, output_size, transform=train_tx, train=train)
    
    return data_set
    
class HanDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list, image_path, crop_ratio, crop_size, output_size, transform=None, train=False):
        
        self.data_list = data_list
        self.image_path = image_path
        self.transform = transform
        self.crop_ratio = crop_ratio
        self.crop_size = crop_size
        self.output_size = output_size
        self.train = train
    
    def __len__(self):
        
        return len(self.data_list)
    
    def __getitem__(self,idx):
        
        file_name = self.data_list[idx][0]
        img = Image.open(self.image_path + file_name).convert('RGB')
        
        origin_pic_width, origin_pic_height = img.size
        output_width, output_height = (self.output_size, self.output_size)
        
        label = np.zeros((output_height, output_width, 6))      
        
        if origin_pic_height < self.crop_size or origin_pic_width < self.crop_size:
            img = torchvision.transforms.functional.resize(img, (self.crop_size, self.crop_size))
        
        pic_width, pic_height = img.size
        _CROPPED = False
        
        if np.random.randint(0, 101) < self.crop_ratio * 100 and pic_height >= self.crop_size and pic_width >= self.crop_size:
            
            top = np.random.randint(0, pic_height - self.crop_size + 1)
            left = np.random.randint(0, pic_width - self.crop_size + 1)
            img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
            _CROPPED = True
            centerX = (2*left + self.crop_size)/2
            centerY = (2*top + self.crop_size)/2
            offsetX = (centerX-self.crop_size/2)*self.output_size/self.crop_size
            offsetY = (centerY-self.crop_size/2)*self.output_size/self.crop_size
        else:
            
            top = 0
            left = 0
            img = torchvision.transforms.functional.resize(img, (self.crop_size, self.crop_size))
            offsetX = 0
            offsetY = 0
            
        for annotation in self.data_list[idx][1]: 
            
            if _CROPPED:
                x_c = annotation[1] * (pic_width / origin_pic_width) * (output_width / self.crop_size) - offsetX
                y_c = annotation[2] * (pic_height / origin_pic_height) * (output_height / self.crop_size) - offsetY
                width = annotation[3] * (pic_width / origin_pic_width)  * (output_width / self.crop_size) 
                height = annotation[4] * (pic_height / origin_pic_height) * (output_height / self.crop_size) 
            else:
                x_c = annotation[1] * (output_width / origin_pic_width) 
                y_c = annotation[2] * (output_height / origin_pic_height) 
                width = annotation[3] * (output_width / origin_pic_width)  
                height = annotation[4] * (output_height / origin_pic_height) 
    
            if x_c >= self.output_size or y_c >= self.output_size or x_c <= 0 or y_c <= 0 :
                continue
    
            heatmap = ((np.exp(-(((np.arange(output_width) - x_c)/(width/10))**2)/2)).reshape(1,-1)
                    *(np.exp(-(((np.arange(output_height) - y_c)/(height/10))**2)/2)).reshape(-1,1))
            
            label[:, :, 0] = np.maximum(label[:,:,0], heatmap[:,:])
            label[int(y_c//1), int(x_c//1), 1] = 1
            label[int(y_c//1), int(x_c//1),2] = y_c % 1
            label[int(y_c//1), int(x_c//1),3] = x_c % 1
            label[int(y_c//1), int(x_c//1), 4] = height / self.output_size 
            label[int(y_c//1), int(x_c//1), 5] = width / self.output_size
        
        if self.transform:
            img = self.transform(img)
            
        if self.train:
            sample = {'image': img, 'labels': label}
        elif not self.train:
            sample = {'image': img, 'labels': label, 'img_size': [origin_pic_width, origin_pic_height]}

        return sample