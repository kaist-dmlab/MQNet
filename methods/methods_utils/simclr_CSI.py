import time
import torch.optim

from .transform_layers import HorizontalFlipLayer
from .contrastive_loss import get_similarity_matrix, NT_xent
from .ccal_util import AverageMeter, normalize
import numpy as np

# P == args
def csi_train_epoch(P, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    assert P.K_shift > 1

    hflip = HorizontalFlipLayer().to(P.device)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    losses['shift'] = AverageMeter()

    check = time.time()
    for n, (images, labels, _) in enumerate(loader):
        model.train()
        count = n #* P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(P.device)
            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(P.device), images[1].to(P.device)
        labels = labels.to(P.device)

        images1 = torch.cat([P.shift_trans(images1, k) for k in range(P.K_shift)])
        images2 = torch.cat([P.shift_trans(images2, k) for k in range(P.K_shift)])
        shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_shift)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(2)

        images_pair = torch.cat([images1, images2], dim=0)  # 8B
        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr)#, multi_gpu=P.multi_gpu)
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda

        loss_shift = criterion(outputs_aux['shift'], shift_labels)

        ### total loss ###
        loss = loss_sim + loss_shift

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        simclr_norm = outputs_aux['simclr'].norm(dim=1).mean()

        penul_1 = outputs_aux['penultimate'][:batch_size]
        penul_2 = outputs_aux['penultimate'][P.K_shift * batch_size: (P.K_shift + 1) * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])  # only use original rotation

        ### Linear evaluation ###
        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())
        # loss_linear = criterion(outputs_linear_eval, labels.repeat(2))

        idx = np.where(labels.cpu().numpy() < P.num_IN_class)[0]
        loss_linear = criterion(outputs_linear_eval[idx], labels[idx].type(torch.LongTensor).to(P.device))  # .repeat(2)

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)
        losses['shift'].update(loss_shift.item(), batch_size)

        if epoch%100 == 0 and count % 3000 == 0:
            print('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n''[LossC %f] [LossSim %f] [LossShift %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value, losses['shift'].value))