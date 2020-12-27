import torch

def calc_loss(pred, gt, metrics):
    
    mask = torch.sign(gt[..., 1])
    N = torch.sum(mask)
    
    if N == 0:
        N = 1
        
    _heatmap_loss = heatmap_loss(pred, gt, mask, metrics)
    _size_loss = size_loss(pred, gt, mask, metrics)
    _offset_loss = offset_loss(pred, gt, mask, metrics) 
    
    all_loss = (-1 * _heatmap_loss + 10. * _size_loss + 5. * _offset_loss) / N
    
    metrics['loss'] = all_loss.item() 
    metrics['heatmap'] = (-1 *  _heatmap_loss / N).item()
    metrics['size'] = (10. * _size_loss / N).item()
    metrics['offset'] = (5. * _offset_loss / N).item()
    
    return all_loss

def heatmap_loss(pred, gt, mask, metrics):
    
    alpha = 2.
    beta = 4.
    
    heatmap_gt_rate = torch.flatten(gt[...,:1])
    heatmap_gt = torch.flatten(gt[...,1:2])
    heatmap_pred = torch.flatten(pred[:,:1,...])
    
    heatloss = torch.sum(heatmap_gt * ((1 - heatmap_pred) ** alpha) * torch.log(heatmap_pred + 1e-9) + 
              (1 - heatmap_gt) * ((1 - heatmap_gt_rate) ** beta) * (heatmap_pred ** alpha) * torch.log(1 - heatmap_pred + 1e-9))
    
    return heatloss

def offset_loss(pred, gt, mask, metrics):
    
    offsetloss = torch.sum(torch.abs(gt[...,2] - pred[:,1,...]*mask) + torch.abs(gt[...,3] - pred[:,2, ...] * mask))
    
    return offsetloss

def size_loss(pred, gt, mask, metrics):
    
    sizeloss = torch.sum(torch.abs(gt[...,4] - pred[:,3, ...]*mask) + torch.abs(gt[...,5] - pred[:,4,...] * mask))
    
    return sizeloss


