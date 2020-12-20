import argparse
import torch
import sys
from tqdm import tqdm
import os

from datasets.HanDataset import dataset_generator
from utils.utility import csv_preprocess, _nms_eval_iou
from utils.losses import calc_loss
from models.HRCenterNet import HRCenterNet

crop_size = 512
output_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('divece: ', device)

def main(args):
    
    dataloader = dict()
    
    train_list = csv_preprocess(args.train_csv_path)
    print("found", len(train_list), "of images for training")
    train_set = dataset_generator(args.train_data_dir, train_list, crop_size, args.crop_ratio, output_size, train=True)
    dataloader['train'] = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    if args.val:
        val_list = csv_preprocess(args.val_csv_path)
        print("found", len(val_list), "of images for validation")
        val_set = dataset_generator(args.val_data_dir, val_list, crop_size, 0, output_size, train=False)
        dataloader['val'] = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)
    
    if not (args.log_dir == None):
        print("Load HRCenterNet from " + args.log_dir)
    
    model = HRCenterNet(args)
    model = model.to(device)
    model.train()
    
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)
    
    train(args, dataloader, model)
    
def train(args, dataloader, model):
    
    num_epochs = args.epoch
    loss_average = 0.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_iou = -1.
    
    print('Start training...')
    for epoch in range(num_epochs):
        loss = 0.
        metrics = {}
        
        for batch_idx, sample in enumerate(dataloader['train']):
            inputs = sample['image'].to(device, dtype=torch.float)
            labels = sample['labels'].to(device, dtype=torch.float)

            outputs = model(inputs)

            loss = calc_loss(outputs, labels, metrics)

            loss_average = loss_average + loss
            sys.stdout.write('\r')
            sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]  Loss: %.4f heatmap_loss: %.5f size_loss: %.5f offset_loss: %.5f'
            %(epoch+1, num_epochs, batch_idx, (len(dataloader['train'].dataset)//(args.batch_size))+1, 
              loss.item(), metrics['heatmap'], metrics['size'], metrics['offset']))
            sys.stdout.write(' average loss: %.5f'%(loss_average / (((len(dataloader['train'].dataset)//(args.batch_size))+1)*epoch + (batch_idx + 1))))
            sys.stdout.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
    
        avg_iou = evaluate(dataloader, model)
        print('Average IoU: ', avg_iou)
        if avg_iou > best_iou:
            print('IoU improve from', best_iou, 'to', avg_iou)
            best_iou = avg_iou
            print('Saving model to', args.weight_dir, 'best.pth')
            torch.save(model.state_dict(), args.weight_dir + 'best.pth')
            
        if epoch % args.save_epoch == 0:
            print('Saving model to', args.weight_dir, str(epoch), '.pth')
            torch.save(model.state_dict(), args.weight_dir + str(epoch) +'.pth')
    
                    

def evaluate(dataloader, model):
    
    print('\nEvaluation...')
    
    iou_sum = 0.
    model.eval()
    
    for batch_idx, sample in enumerate(tqdm(dataloader['val'])):
        with torch.no_grad():
            inputs = sample['image'].to(device, dtype=torch.float)
            labels = sample['labels'].to(device, dtype=torch.float)
            outputs = model(inputs)
            img_width, img_height = sample['img_size']
            iou = _nms_eval_iou(labels, outputs, img_width.item(), img_height.item(), output_size, nms_score=0.3, iou_threshold=0.1)
            
            iou_sum = iou_sum + iou
            
    return iou_sum / len(dataloader['val'].dataset)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRCenterNet.")
    
    parser.add_argument("--train_data_dir", required=True,
                      help="Path to the training images folder, preprocessed for torchvision.")
    
    parser.add_argument("--train_csv_path", required=True,
                       help="Path to the csv file for training")
    
    parser.add_argument('--val', default=False, action='store_true',
                       help="traing with validation")
    
    parser.add_argument("--val_data_dir", required=False,
                      help="Path to the validated images folder, preprocessed for torchvision.")
    
    parser.add_argument("--val_csv_path", required=False,
                       help="Path to the csv file for validation")
    
    parser.add_argument("--log_dir", required=False, default=None,
                      help="Where to load for the pretrained model.")
    
    parser.add_argument("--epoch", type=int, default=80,
                       help="number of epoch")
    
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="learning rate")
    
    parser.add_argument("--batch_size", type=int, default=8,
                       help="number of batch size")
    
    parser.add_argument("--crop_ratio", type=float, default=0.5,
                       help="crop ration for random crop in data augumentation")
    
    parser.add_argument('--weight_dir', default='./weights/',
                       help="Where to save the weight")
    
    parser.add_argument('--save_epoch', type=int, default=10,
                       help="save model weight every number of epoch")
    

    main(parser.parse_args())