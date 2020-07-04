from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model,generate_model_t
from opts import parse_opts
from torch.autograd import Variable
import time
import sys
from utils import *
#from utils import AverageMeter, calculate_accuracy
import pdb
import math
from models.resnext import get_fine_tuning_parameters
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k


def kl_loss(pre1,pre2):
	criterion_softmax = torch.nn.Softmax(dim=1).cuda()
	pre1=criterion_softmax(pre1)
	pre2=criterion_softmax(pre2)
	loss=torch.mean(torch.sum(pre2*torch.log(1e-8+pre2/(pre1+1e-8)),1))
	return loss
if __name__=="__main__":
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    torch.manual_seed(opt.manual_seed)

    print("Preprocessing train data ...")
    train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
    print("Length of validation data = ", len(val_data))
    
    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    val_dataloader   = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))   

    log_path = os.path.join(opt.result_path, opt.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.log == 1:
        if opt.pretrain_path != '':
            epoch_logger = Logger_STS(os.path.join(log_path, 'PreKin_STS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.STS_alpha))
                            ,['epoch', 'loss', 'loss_MSE', 'loss_STS', 'acc', 'lr'], opt.STS_resume_path, opt.begin_epoch)
            val_logger   = Logger_STS(os.path.join(log_path, 'PreKin_STS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset,opt.split,  opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.STS_alpha))
                            ,['epoch', 'loss', 'acc'], opt.STS_resume_path, opt.begin_epoch)
        else:
            epoch_logger = Logger_STS(os.path.join(log_path, 'STS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.STS_alpha))
                            ,['epoch', 'loss', 'loss_MSE', 'loss_STS', 'acc', 'lr'], opt.STS_resume_path, opt.begin_epoch)
            val_logger   = Logger_STS(os.path.join(log_path, 'STS_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}.log'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                                opt.output_layers[0], opt.STS_alpha))
                            ,['epoch', 'loss', 'acc'], opt.STS_resume_path, opt.begin_epoch)

    if opt.pretrain_path!='' and opt.dataset!= 'Kinetics': 
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.00001
        
    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening

       
    # define the model 
    print("Loading STS model... ", opt.model, opt.model_depth)
    opt.input_channels =3
    model_STS, parameters_STS = generate_model_t(opt)
    model_STS_dict=model_STS.state_dict()


    print("Loading Flow model... ", opt.model, opt.model_depth) 
    opt.input_channels =2 
    if opt.pretrain_path != '':

        if opt.dataset == 'HMDB51':
            opt.n_classes = 51
        elif opt.dataset == 'Kinetics':
            opt.n_classes = 400 
        elif opt.dataset == 'UCF101':
            opt.n_classes = 101 

    model_Flow, parameters_Flow = generate_model(opt)
    
    criterion_STS  = nn.CrossEntropyLoss().cuda()

    criterion_MSE = nn.MSELoss().cuda()

    t=''
    t_s=[]

    
    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        model_Flow.load_state_dict(checkpoint['state_dict'])
    if opt.resume_path2:
        print('loading checkpoint {}'.format(opt.resume_path2))
        checkpoint = torch.load(opt.resume_path2)
        model_STS.load_state_dict(checkpoint['state_dict'])
    if opt.STS_resume_path:
        print('loading STS checkpoint {}'.format(opt.STS_resume_path))
        checkpoint = torch.load(opt.STS_resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model_STS.load_state_dict(checkpoint['state_dict'])

    #  assert opt.arch == checkpoint['arch']

#        opt.begin_epoch = checkpoint['epoch']



    
    print("Initializing the optimizer ...")
        
    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience) 

    optimizer = optim.SGD(
        parameters_STS,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    

    model_Flow.eval()
    print('run')
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        
        model_STS.train()
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses     = AverageMeter()
        losses_STS = AverageMeter()
        losses_MSE1 = AverageMeter()
        losses_kl=AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
            inputs_STS  = inputs[:,0:3,:,:,:]
            inputs_Flow = inputs[:,3:,:,:,:]
            
            targets = targets.cuda(non_blocking=True)
            # pdb.set_trace()
            inputs_STS  = Variable(inputs_STS)
            inputs_Flow = Variable(inputs_Flow)
            targets     = Variable(targets)



            outputs_STS  = model_STS(inputs_STS)


            outputs_Flow = model_Flow(inputs_Flow)


            #更改
            """
            Batchnorm1 = nn.BatchNorm2d(outputs_Flow.shape[1]).cuda()
            Batchnorm2 = nn.BatchNorm2d(outputs_STS[1].shape[1]).cuda()

            outputs_Flow1 = outputs_Flow.reshape(outputs_Flow.shape[0], outputs_Flow.shape[1], 1, 1)
            outputs_STS_feat1 = outputs_STS[1].reshape(outputs_STS[1].shape[0], outputs_STS[1].shape[1], 1, 1)


            outputs_Flow2=Batchnorm1(outputs_Flow1)
            outputs_STS_feat2=Batchnorm2(outputs_STS_feat1)

            print('size 0:', outputs_Flow2.size())
            print('size 1`', outputs_STS_feat2.size())
            outputs_Flow2 = outputs_Flow2.reshape(outputs_Flow2.shape[0],outputs_Flow2.shape[1])
            outputs_STS_feat2 = outputs_STS_feat2.reshape(outputs_STS_feat2.shape[0],outputs_STS_feat2.shape[1])
            
            """
            #####尝试用高斯核函数

            loss_MSE1=criterion_MSE(outputs_STS[2],outputs_Flow[2])

           # loss_MSE1=loss_MSE1.pow(0.5)
            kl=kl_loss(outputs_STS[0],outputs_Flow[0])




            ######



            loss_STS = criterion_STS(outputs_STS[0], targets)
            #loss_MSE = opt.STS_alpha*criterion_Flow(outputs_STS[1], outputs_Flow)
            """"
            print(outputs_Flow[1][0:100])
            outputs_STS_mmd=average(outputs_STS[1]).cuda()
            outputs_Flow_mmd=average(outputs_Flow).cuda()
            print(outputs_Flow_mmd[1][0:100])
            
            ddd=poly_mmd2(outputs_STS_feat2,outputs_Flow2).cuda()
            """



            loss     = 2*loss_STS + loss_MSE1*200+kl*4
            acc = calculate_accuracy(outputs_STS[0], targets)

            losses.update(loss.data, inputs.size(0))
            losses_STS.update(loss_STS.data, inputs.size(0))
            losses_MSE1.update(loss_MSE1.data, inputs.size(0))
            losses_kl.update(kl.data,inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_STS {loss_STS.val:.4f} ({loss_STS.avg:.4f})\t'
                  'Loss_MSE {loss_MSE.val:.4f} ({loss_MSE.avg:.4f})\t'
                  'kl {kl.val:.4f} ({kl.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss_STS=losses_STS,
                      loss_MSE=losses_MSE1,
                      kl=losses_kl,
                      acc=accuracies))
                      
        if opt.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'loss_MSE' : losses_MSE1.avg,
                'loss_STS': losses_STS.avg,
                'kl':losses_kl.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        if epoch % opt.checkpoint == 0:
            if opt.pretrain_path != '':
                save_file_path = os.path.join(log_path, 'STS_preKin_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index, 
                                opt.output_layers[0], opt.STS_alpha, epoch))
            else:
                save_file_path = os.path.join(log_path, 'STS_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_layer{}_alpha{}_{}.pth'
                            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index, 
                                opt.output_layers[0], opt.STS_alpha, epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model_STS.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        
        model_STS.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataloader):
                
                data_time.update(time.time() - end_time)
                inputs_STS  = inputs[:,0:3,:,:,:]
                
                targets = targets.cuda(non_blocking=True)
                inputs_STS  = Variable(inputs_STS)
                targets     = Variable(targets)
                
                outputs_STS  = model_STS(inputs_STS)
                
                loss = criterion_STS(outputs_STS[0], targets)
                acc  = calculate_accuracy(outputs_STS[0], targets)

                losses.update(loss.data, inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Val_Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch,
                        i + 1,
                        len(val_dataloader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accuracies))
                          
        if opt.log == 1:
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        scheduler.step(losses.avg)
        



