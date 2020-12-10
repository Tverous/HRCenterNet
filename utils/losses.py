import torch

def calc_loss(y_pred, y_true, metrics):
    
    alpha=2.
    beta=4.
    
    mask=torch.sign(y_true[...,4])
    N=torch.sum(mask)
    
    heatmap_true_rate = torch.flatten(y_true[...,:1])
    heatmap_true = torch.flatten(y_true[...,1:2])
    heatmap_pred = torch.flatten(y_pred[:,:1,...])
    
    heatloss= torch.sum(heatmap_true*((1-heatmap_pred)**alpha)*torch.log(heatmap_pred+1e-9)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*torch.log(1-heatmap_pred+1e-9))
    offsetloss = torch.sum(torch.abs(y_true[...,2]-y_pred[:,1,...]*mask)+torch.abs(y_true[...,3]-y_pred[:,2, ...]*mask))
    sizeloss = torch.sum(torch.abs(y_true[...,4]-y_pred[:,3, ...]*mask)+torch.abs(y_true[...,5]-y_pred[:,4,...]*mask))

    
    
    all_loss=(-1*heatloss+5.*sizeloss+5.*offsetloss)/N
    metrics['loss'] = all_loss.data.cpu().numpy() 
    metrics['heatmap'] = (-1*heatloss/N).data.cpu().numpy()
    metrics['size'] = (5.*sizeloss/N).data.cpu().numpy()
    metrics['offset'] = (5.*offsetloss/N).data.cpu().numpy()
    
    return all_loss

def offset_loss(pred, gt, metrics):
    
    offsetloss = torch.sum(torch.abs(y_true[...,2]-y_pred[:,1,...]*mask)+torch.abs(y_true[...,3]-y_pred[:,2, ...]*mask))
    
    return offsetloss


