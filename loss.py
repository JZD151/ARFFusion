import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torch.fft as fft


class FourierLoss(nn.Module):  # nn.Module
    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super(FourierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_mag, sr_ang = self.comFourier(sr_w)
        hr_mag, hr_ang = self.comFourier(hr_w)
        mag_loss = self.get_l1loss(sr_mag, hr_mag, weight=self.loss_weight, reduction=self.reduction)
        # ang_loss = self.get_angleloss(sr_ang, hr_ang, weight=self.loss_weight, reduction=self.reduction)
        ang_loss = self.get_l1loss(sr_ang, hr_ang, weight=self.loss_weight, reduction=self.reduction)
        return (mag_loss + ang_loss)

    def comFourier(self, image):
        frm_list = []
        fre_quen = []
        for i in range(1):
            in_ = image[:, i:i + 1, :, :]
            fftn = fft.fftn(in_, dim=(2, 3))
            # add shift
            fftn_shift = fft.fftshift(fftn)  # + 1e-8
            # print('fftn:', fftn_shift.size())
            frm_list.append(fftn_shift.real)
            fre_quen.append(fftn_shift.imag)
        fre_mag = torch.cat(frm_list, dim=1)
        fre_ang = torch.cat(fre_quen, dim=1)

        return fre_mag, fre_ang

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h).reshape(h, 1)
        win2 = torch.hann_window(w).reshape(1, w)
        win = torch.mm(win1, win2).cuda()
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_angleloss(self, pred, target, weight, reduction):
        # minimum = torch.minimum(pred, target)
        diff = pred - target
        minimum = torch.min(diff)
        loss = torch.mean(torch.abs(minimum))
        return weight * loss

    def get_l1loss(self, pred, target, weight, reduction):
        loss = F.l1_loss(pred, target, reduction=reduction)
        return weight * loss


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0).float()
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0).float()
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

    
class fusion_loss_adv(nn.Module):
    def __init__(self):
        super(fusion_loss_adv, self).__init__()
        self.sobelconv = Sobelxy()
        self.fourierLoss = FourierLoss()

    def forward(self, generate_img_adv, generate_img):
        # L_mse
        loss_mse = F.mse_loss(generate_img_adv, generate_img)

        # L_ssim
        loss_ssim = 1 - ssim(generate_img_adv, generate_img)

        # L_fre
        loss_fre = self.fourierLoss(generate_img_adv, generate_img)
        
        # L_total
        loss_total = 100 * loss_mse + 100 * loss_ssim + loss_fre
        return loss_total, loss_mse, loss_ssim, loss_fre
    
class label_generator_loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sobelconv = Sobelxy()

    def forward(self, image_vis_ycrcb, image_ir, logits):
        N, C, H, W = image_vis_ycrcb.shape

        x_in_max = torch.max(image_vis_ycrcb, image_ir)
        loss_in = F.l1_loss(x_in_max, logits)

        vis_grad = self.sobelconv(image_vis_ycrcb)
        ir_grad = self.sobelconv(image_ir)
        logits_grad = self.sobelconv(logits)
        x_grad_max = torch.max(vis_grad, ir_grad)
        loss_grad = F.l1_loss(logits_grad, x_grad_max)

        loss_ssim = 1 - ssim(image_vis_ycrcb, logits)/2 - ssim(image_ir, logits)/2

        loss_total = loss_in + loss_grad + loss_ssim
        return loss_total, loss_in, loss_grad, loss_ssim
