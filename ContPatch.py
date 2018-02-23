import numpy as np
import os
from scipy import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable as V
from collections import defaultdict
import cPickle as pickle
import matplotlib.pyplot as plt

#Contour matrix format: n x 6. The rows are: x,y, label, delta_nx,delta_ny, radius. point coords, contours label, residual correction to normale, radius to extract patch.



def denormalizeContour(contour, img_w, img_h):
    contour[:,0] = contour[:,0] * img_w;
    contour[:,1] = contour[:,1] * img_h;
    return contour

def normalizeContour(contour, img_w, img_h):
    contour[:,0] = contour[:,0] / float(img_w);
    contour[:,1] = contour[:,1] / float(img_h);
    return contour

def getGaussKernel1D(sigma):
    g = torch.arange(-3*sigma, 3*sigma + 1);
    g = torch.exp(-g*g / (2.*sigma*sigma));
    g = g / g.sum(0, keepdim=True).expand_as(g);
    return g
    
def SelectContPatch(cpatches, conts, idx):
    idxs = conts[:,2] == idx;
    return cpatches[0,0,idxs,:]

def ShowContPatch(cpatches, conts, idx):
    plt.imshow(255 - SelectContPatch(cpatches, conts, idx).data.t().cpu().numpy())
    return

def smoothContour(points, sigma = 6.0):
    g = V(getGaussKernel1D(sigma).view(1,1,-1))
    if points.is_cuda:
        g = g.cuda()
    xcx = F.pad(points.unsqueeze(0)[:,0:1,:], (int(3*sigma),int(3*sigma)), 'replicate')
    xcy = F.pad(points.unsqueeze(0)[:,1:2,:], (int(3*sigma),int(3*sigma)), 'replicate')
    return torch.cat([F.conv1d(xcx, g),  F.conv1d(xcy, g)],dim = 1)

def smoothAndGetNormals(points, sigma = 6.0):
    x = smoothContour(points, sigma)
    gx =  nn.Conv1d(1, 1, kernel_size=(1,3), bias = False)
    gx.weight.data = torch.from_numpy(np.array([[[-0.1, 0, 0.1]]], dtype=np.float32))
    if points.is_cuda:
        gx = gx.cuda()
    df =  torch.cat([gx(F.pad(x[:,0:1,:],(1,1), 'replicate')), gx(F.pad(x[:,1:2,:],(1,1), 'replicate'))],dim = 1)
    df2 =  torch.cat([gx(F.pad(df[:,0:1,:],(1,1), 'replicate')), gx(F.pad(df[:,1:2,:],(1,1), 'replicate'))],dim = 1)
    n = torch.cat([df[:,1:2], -df[:,0:1]], dim=1)
    un = n / (torch.sqrt((n*n).sum(dim=1)+1e-10).unsqueeze(1).expand_as(n))
    return x, un

def GetContoursPatchGrid(conts, patchHeight):
    sigma = 1.0;
    ph = patchHeight;
    x = torch.zeros(1, 2, conts.size(0))
    un = torch.zeros(1, 2, conts.size(0))
    if conts.is_cuda:
        x = x.cuda()
        un = un.cuda()
    x = V(x)
    un = V(un)
    if type(conts) is not torch.autograd.Variable:
        conts = V(conts)
    idxs_uniq = np.unique(conts[:,2].data.cpu().numpy())
    for i in idxs_uniq:
        idxs = conts[:,2] == int(i)
        if idxs.float().sum() < 1:
            continue
        x[:,:,idxs], un[:,:,idxs] = smoothAndGetNormals((conts[idxs,0:2]).t(), sigma)
        #print (conts[idxs.nonzero(),:][:,0:2].squeeze()).t().shape
        #try:
        #except:
        #    x[:,:,idxs], un[:,:,idxs] = smoothAndGetNormals((conts[idxs.nonzero(),:][:,0:2].squeeze()).t(), sigma)
    x = x.squeeze().t().unsqueeze(0)
    un = (un.squeeze().t().unsqueeze(0) + conts[:,3:5].unsqueeze(0))\
        * conts[:,5:6].view(-1,1).repeat(1,2).unsqueeze(0)
    py_r_sizex = V(torch.linspace(-1.0, 1.0, ph).repeat(2,1).t()).unsqueeze(1)
    if conts.is_cuda:
        py_r_sizex = py_r_sizex.cuda()
    grid = un.repeat(ph,1,1) * py_r_sizex.repeat(1, x.size(1), 1) + x.repeat(ph, 1, 1)
    return grid.t().unsqueeze(0)

def GetKLongestFromM(M, K = -1):
    if K < 0:
        return M
    idxs_dict = {}
    idxs = np.unique(M[:,2])
    for i in idxs:
        idxs_dict[str(i)] = M[:,2] == i
    idxs_by_len = sorted(idxs_dict, key=lambda x : idxs_dict[x].sum())[::-1]
    idxs_to_keep = []
    for i in range(K):
        idxs_to_keep = idxs_to_keep + list(idxs_dict[idxs_by_len[i]].nonzero()[0])
    return M[idxs_to_keep, :]

def makeContoursFromMNumpy(M, radius, delta_un = None):
    n = M.shape[0];
    cont = V(torch.zeros(n,6).float())
    xylabel = V(torch.from_numpy(M[:,0:3]).float())
    if type(radius) is float:
        r = radius * V(torch.ones(n,1).float())
    if type(radius) is torch.autograd.Variable:
        idxs = np.unique(M[:,2])
        num_conts = len(idxs)
        assert (radius.size(0) == num_conts) or (radius.size(0)  == n) 
        #you can provide radius per contour or per point
        if radius.size(0)  == n:
            r = radius
        else:
            r = V(torch.ones(n,1).float())
            for i in idxs:
                cur_idxs = xylabel[:,2] == int(i)
                r[cur_idxs] = cur_idxs * radius[i]
    if delta_un is not None:
        assert delta_un.size(0) == n and delta_un.size(1) == 2
        un = delta_un
    else:
        un = V(torch.zeros(n,2).float())
    return torch.cat([xylabel, un, r], dim = 1)
def MakeContourPatches(img, conts, patchHeight):
    grid = GetContoursPatchGrid(conts,  patchHeight)
    grid[:,:,:,0] = 2.0 * grid[:,:,:,0] / float(img.size(3))  - 1.0
    grid[:,:,:,1] = 2.0 * grid[:,:,:,1] / float(img.size(2))  - 1.0  
    return F.grid_sample(img,  grid)

def DrawContourBoundaries(img, conts, c = ('g', 'b')):
    ph = 41
    grid = GetContoursPatchGrid(conts, ph).transpose(1,2)
    plt.figure()
    plt.imshow(255 - img.cpu().data.numpy().squeeze())
    if type(conts) is not torch.autograd.Variable:
        conts = V(conts)
    for i in np.unique(conts[:,2].data.cpu().numpy()):
        idxs = torch.nonzero(conts[:,2] == int(i)).long()
        if len(idxs) < 1:
            continue
        boundary_x = np.concatenate([grid[0,0, idxs, 0].data.cpu().numpy(), grid[0,ph-1,idxs,0].data.cpu().numpy()[::-1]])
        boundary_y = np.concatenate([grid[0,0, idxs, 1].data.cpu().numpy(), grid[0,ph-1,idxs,1].data.cpu().numpy()[::-1]])
        boundary_x = np.concatenate([boundary_x,boundary_x[0:1]])
        boundary_y = np.concatenate([boundary_y,boundary_y[0:1]])
        plt.plot(boundary_x, boundary_y, c[0])
        plt.plot(grid[0,ph/2, idxs, 0].data.cpu().numpy(), grid[0,ph/2, idxs,1].data.cpu().numpy(), c[1])
    plt.show()
    return