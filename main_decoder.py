import argparse
import os
import shutil
import time
import torchvision.models as models
import torch.nn as nn
#from sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from scipy import stats
from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F
from inference_module import inference

from build_model_decoder import *

# from Loader_10k import *
from Loader_TID import *

# from Loader_Clive import *


# configured in main configuration
os.environ['CUDA_VISIBLE_DEVICES']='2'

# sync_bn = SynchronizedBatchNorm2d(10, eps=1e-5, affine=False)
# sync_bn = DataParallelWithCallback(sync_bn, device_ids=[2, 3])



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=700, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('plot_dir', metavar='PLOT', default='./plot20/', type=str)
parser.add_argument('--weight', default=1, type=float)



rootdir='../TID2013/TID_data/'
# rootdir='../10k/'
min_loss=100
loss1=100
loss2=100
loss=100

notice='TID_dali_decoder_L1_10loss2'
# pkl_file='./split/split_10k.pkl'
pkl_file='./split/split_TID.pkl'
# notice='ate'
# notice='PLCC_finetune_l2'
# saved_model='./10k_TRANS_resize300/model_best.pth.tar'

def main():

    global args, min_loss, max_cc
    args = parser.parse_args()
    args.plot_dir = './'+notice+'/'
    args.resume='./'+notice+'/'+'model_best.pth.tar'

    model = Vgg16(reg=True, requires_grad=True).cuda()
    # model=distortion_net()

    loss_net=Vgg16_fix(requires_grad=False).cuda()


    # model_dict = model.state_dict()
    #
    # best_model = torch.load(saved_model)
    # best_dict = best_model['state_dict']
    # print('plcc', best_model['plcc'])
    # print('srocc', best_model['srocc'])
    #
    # print('loading done')
    #
    # best_dict = {k: v for k, v in best_dict.items() if k in model_dict}
    #
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(best_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)
    #
    # model = model.cuda()



    # pretrained_dict = models.vgg16(pretrained=True).state_dict()
    #
    # model_dict=model.state_dict()
    #
    # print('loading model.....')
    # #1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #
    # # l=pretrained_dict.items()    # list 0:key    1:value
    # # pretrained_dict_same = {k: l[idx][1] for idx,(k, _v) in enumerate(model_dict.items()) if 'feature' in k}
    #
    # # 2. overwrite entries in the existing state dict
    #
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(model_dict)
    # model = model.cuda()





    deconv_params = list(
        map(id, [decon.parameters() for (name, decon) in model.named_children() if 'deconv' in name or 'in' in name]))
    # base_params = list(map(id, model.my_fc.parameters()))
    deconv = filter(lambda p: id(p) in deconv_params, model.parameters())
    reg_params = filter(lambda p: id(p) not in deconv_params, model.parameters())

    params = [
        {'params': reg_params},
        {'params': deconv, 'lr': 1e-2},
        # {'params': model.my_fc.parameters()}
    ]
    #
    # optimizer_all = torch.optim.SGD( params, 1e-5,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer_all=torch.optim.Adam(params,1e-3)

    # model = nn.DataParallel(model, device_ids=[0, 1])


    critition=nn.MSELoss()

    #critition=nn.L1Loss()

    # model = TransformerNet().cuda()


    mse_loss = torch.nn.MSELoss()



    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_plcc = checkpoint['plcc']
            best_srocc = checkpoint['srocc']
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            # optimizer_all.load_state_dict(checkpoint['optimizer'])
            # max_cc = [best_plcc, best_srocc]
            print("=> loaded checkpoint '{}' (epoch {}) with lcc&srocc {} {} "
                  .format(args.resume, checkpoint['epoch'], checkpoint['plcc'], checkpoint['srocc']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = get_imagenet_normalize()

    #### CLIVE
    size=256
    train_loader = torch.utils.data.DataLoader(
        Reg_Multi_Loader(rootdir, 'train',  transforms.Compose([
            # transforms.RandomCrop(700),
            # transforms.Resize((500,500)),
            transforms.RandomCrop(size),
            # transforms.Resize((500, 500)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_trans = transforms.Compose([
        transforms.RandomCrop(size),
        # transforms.RandomCrop(700),
        # transforms.Resize((500, 500)),
        transforms.ToTensor(),
        normalize,
    ])
    val_loader = torch.utils.data.DataLoader(
        Reg_Multi_Loader(rootdir, 'val', val_trans),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    plot_losses1 = []
    plot_losses2 = []
    plot_losses=[]
    plot_lcc = []
    plot_srocc = []
    train_losses1 = []
    train_losses2 = []
    train_losses=[]

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)


    for epoch in range(args.start_epoch, args.epochs):

        print(notice)

        # if epoch<1:
        #     optimizer=optimizer_0
        # else:
        #     optimizer=optimizer_all

        optimizer = optimizer_all

        #adjust_learning_rate(optimizer, epoch)
        deconv=None

        train_loss1, train_loss2, train_loss = train(train_loader, val_loader, model, deconv, loss_net, critition,
                                                     mse_loss,
                                                     optimizer, epoch)

        val_loss1, val_loss2, val_loss, lcc, srocc = validate(val_loader, model, deconv, loss_net, critition, mse_loss,
                                                              epoch)


        is_best = (int(val_loss1 < loss1) +int( val_loss2<loss2) +int(val_loss<loss) ) >= 2

        print('loss1:%.3f, loss2:%.3f loss:%.3f[BEST]' % (val_loss1,val_loss2,val_loss))

        # if  is_best:
        #     print('inference')
        #     plcc, srocc = inference(notice,model, val_trans, rootdir, pkl_file, args.plot_dir)

        # plot
        plot_losses1.append(val_loss1)
        plot_losses2.append(val_loss2)
        plot_losses.append(val_loss)
        plot_lcc.append(lcc)
        plot_srocc.append(srocc)
        train_losses1.append(train_loss1)
        train_losses2.append(train_loss2)
        train_losses.append(train_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'plcc': lcc,
            'srocc': srocc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.plot_dir)


        if (epoch+1)%10==0:
            plt.subplot(3,1,1)
            # ylim(0, 0.5)
            plt.grid(axis='y')
            # yticks(np.arange(0, 0.5, 0.05))
            plt.plot(args.start_epoch+np.arange(len(plot_losses1)), plot_losses1)
            plt.plot(args.start_epoch+np.arange(len(train_losses1)), train_losses1)
            # savefig(args.plot_dir + 'loss_ep%d_.jpg' % (epoch))
            # close()
            #
            plt.subplot(3,1,2)
            # ylim(0, 0.2)
            plt.grid(axis='y')
            # yticks(np.arange(0, 0.2, 0.02))
            plt.plot(args.start_epoch+np.arange(len(plot_losses2)), plot_losses2)
            plt.plot(args.start_epoch+np.arange(len(train_losses2)), train_losses2)
            # savefig(args.plot_dir + 'loss_ep%d_.jpg' % (epoch))
            # close

            plt.subplot(3, 1, 3)
            # ylim(0, 0.2)
            plt.grid(axis='y')
            # yticks(np.arange(0, 0.2, 0.02))
            plt.plot(args.start_epoch + np.arange(len(plot_losses)), plot_losses)
            plt.plot(args.start_epoch + np.arange(len(train_losses)), train_losses)

            # subplot(3,1,3)
            # # ylim(0.5, 1.0)
            # grid(axis='y')
            # # yticks(np.arange(0.5, 1, 0.05))
            # plot(args.start_epoch+np.arange(len(plot_lcc)), plot_lcc)
            # plot(args.start_epoch+np.arange(len(plot_srocc)), plot_srocc)
            plt.savefig(args.plot_dir + 'ep%d_.jpg' % (epoch))
            plt.close()


def train(train_loader, val_loader, model, deconv_net, vgg, critition,mse_loss, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses = AverageMeter()

    img_loss=nn.L1Loss()
    # switch to train mode
    #img_loss=nn.MSELoss()
    model.train()

    end = time.time()
    for i, (input, target, std) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        x = input

        target = target.cuda(async=True).unsqueeze(1).float()
        std = std.cuda(async=True).unsqueeze(1)
        # compute output
        output, y, _, _ = model(input)
        if random.random() < 0.1:
            new_img = transforms.ToPILImage()(y[0].cpu())
            # plt.imshow(new_img)
            new_img.save('./plot_dali/%d_%d_train.jpg' % (epoch, i))

        out_batchmean=torch.mean(output)
        out_batchstd=torch.std(output)

        target_batchmean = torch.mean(target, dim=0)
        target_batchstd=torch.std(target)
        # loss1=1-torch.mean(((output-out_batchmean)*(target-target_batchmean)/(out_batchstd*target_batchstd)))
        # loss3= critition(output, target)
        y = normalize_batch(y)
        loss4 = 1 * img_loss(y, input)
        # x = normalize_batch(x)   # already norm
        # features_y = vgg(y)

        features_y = vgg(y)
        features_x = vgg(x)

        loss2_2 = 10*mse_loss(features_y.relu2_2, features_x.relu2_2)


        loss = loss4 + loss2_2 

        losses.update(loss.item(), input.size(0))
        losses1.update(loss4.item(), input.size(0))
        losses2.update(loss2_2.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss1=losses1,loss2=losses2))
    return losses1.avg, losses2.avg, losses.avg




def validate(val_loader, model, deconv_net,vgg, critition,mse_loss,  epoch):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    val_output=np.asarray([])
    val_target=np.asarray([])

    # switch to evaluate mode
    img_loss = nn.L1Loss()
    # img_loss=nn.MSELoss()
    model.eval()

    for i, (input, target, std) in enumerate(val_loader):
        # if i>10:
        #     break
        val_target=np.append(val_target, target)   # all the validation set

        input = input.cuda(async=True)
        x=input
        target = target.cuda(async=True).unsqueeze(1).float()
        std = std.cuda(async=True).unsqueeze(1)

        with torch.no_grad():

            output, y,_ ,_ = model(input)

            if random.random()<0.1:
                new_img = transforms.ToPILImage()(y[0].cpu())
                # plt.imshow(new_img)
                new_img.save('./plot_dali/%d_%d.jpg'%(epoch,i))



        # loss = critition(output, target)
        # losses.update(loss.item(), input.size(0))
        out_batchmean=torch.mean(output)
        out_batchstd=torch.std(output)



        target_batchmean = torch.mean(target, dim=0)
        target_batchstd = torch.std(target)
        # loss1 = 1-torch.mean(((output-out_batchmean)*(target-target_batchmean)/(out_batchstd*target_batchstd)))
        # loss3 = critition(output, target)

        y = normalize_batch(y)
        loss4=1 * img_loss(y,input)


        features_y = vgg(y)
        features_x = vgg(x)

        loss2_2 = 10*mse_loss(features_y.relu2_2, features_x.relu2_2)
        loss=loss4+loss2_2

        losses.update(loss.item(), input.size(0))
        losses1.update(loss4.item(), input.size(0))
        losses2.update(loss2_2.item(), input.size(0))

        val_output=np.append(val_output, output.squeeze(1).tolist())


        if i % 5 == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                .format(
                   i, len(val_loader),loss=losses, loss1=losses1,loss2=losses2))

    # whole_output=np.asarray(val_output)
    # whole_target = val_target
    # val_lcc, _= stats.pearsonr(whole_output, whole_target)
    # val_srocc, _ = stats.spearmanr(whole_output, whole_target)
    # print('plcc:%.3f, srocc:%.3f' % (val_lcc, val_srocc))
    #
    # plt.scatter(whole_target, whole_output, alpha=0.1)
    # # plt.show()
    # plt.savefig('./%s/epoch%d_plcc%.3f srocc%.3f.jpg' % (notice,epoch,val_lcc, val_srocc))
    # plt.close()

    return losses1.avg, losses2.avg, losses.avg, 0, 0


def adjust_learning_rate(optimizer, epoch):
    # if epoch <= 5:
    #     optimizer.param_groups
    lr = args.lr * (0.2 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_imagenet_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



if __name__ == '__main__':
    main()
