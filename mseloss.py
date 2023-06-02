import numpy as np
import torch,random
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image

class meshloss(nn.Module):
    def __init__(self, use_gpu = True):

        super(meshloss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        internel = batch_size
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                #print ('posi',posi_loss)
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                    #print ('naga1',nega_loss)
                    if len(pre_loss[i][(loss_label[i] < 0.1)]) == 0:
                        nega_loss = 0
                    #print ( 'ss',len(pre_loss[i][(loss_label[i] < 0.1)]))

                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                    #print ('naga2',nega_loss)
                sum_loss += nega_loss

            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss
                #print ('naga3',nega_loss)
            #sum_loss += loss/average_number

        return sum_loss



    def forward(self,  gh_label, gv_label, p_gh, p_gv, mask):
        #gbh_label = gbh_label
        #gah_label = gah_label
        #p_gh = p_gh
        #p_gah = p_gah
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        #loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)

        #print (p_gbh.size(),gbh_label.size())
        #assert p_gbh.size() == gbh_label.size()
        #print (p_gh.size(), gh_label.size())

#        h, w = p_gh.shape[1:]
#        gh_label = transforms.ToTensor()(transforms.Resize([h,w])(Image.fromarray(gh_label.numpy())))
#
#        h, w = p_gv.shape[1:]
#        gv_label = transforms.ToTensor()(transforms.Resize([h,w])(Image.fromarray(gv_label.numpy())))

        loss_h = loss_fn(p_gh, gh_label)
        loss_v = loss_fn(p_gv, gv_label)
        #loss = torch.mul(loss, mask)
        #loss_a = torch.mul(loss2, mask)
        #if loss_g.shape[0] == 0 or loss_a.shape[0]==0:
        #    print ('error',mask)
        #print ('sb1',loss_g,loss_a)

        char_loss = self.single_image_loss(loss_h, gh_label)
        affi_loss = self.single_image_loss(loss_v, gv_label)
        #box_loss = self.single_image_loss(loss,gbh_label)
        #print ('sb2',char_loss,affi_loss)
        return char_loss/loss_h.shape[0] + affi_loss/loss_h.shape[0]
        #return box_loss/loss.shape[0]



