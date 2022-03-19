import argparse
import gc
import importlib
import os
import sys
import time
from tkinter import N

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import dataset_get
from utils import cal_loss, get_logger, read_yaml

# sys.path.append("./") 

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--config', default='configs/config.yaml', type=str, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--exp_name', default=None, type=str, metavar='PATH',
                    help='exp_name')     
parser.add_argument('--seed', type=int, help='random seed')      

def main():
    args = parser.parse_args()
    assert args.config is not None
    cfg = read_yaml(args.config)
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if cfg.data_name == 'scanobjectnn':
        cfg.num_class = 15
    
    # prepare file structures
    time_str = time.strftime("%Y-%m-%d_%H:%M_", time.localtime())
    root_dir = cfg.log_dir+time_str+cfg.exp_name
    backup_dir = root_dir + '/backups'

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    else:
        a = input("File location already exists, overwrite or not:  y / n  ?")
        if a == 'n':
            # os._exit(0)
            try:
                sys.exit(0)
            except:
                print ('exit')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    os.system('cp '+ __file__ + ' ' + backup_dir + '/train.py.backup')
    os.system('cp models/{}.py '.format(cfg.model_name) + backup_dir + '/model.py.backup')
    os.system('cp models/{}.py models/{}.py'.format(cfg.model_name, cfg.model_copy_name))
    os.system('cp '+ cfg.config_dir + ' ' + backup_dir + '/config.yaml.backup')


    logger = get_logger(os.path.join(root_dir, cfg.logger_filename))
    logger.info("start code ---------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = torch.device("cuda")
    logger.info(f'Using NO.{str(cfg.gpu)} device')
    # logger.info("There are {} gpus in total".format(torch.cuda.device_count()))
    # logger.info("Torch Version: {}".format(torch.__version__))
    # logger.info("Cuda Version: {}".format(torch.version.cuda))
    # assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    # logger.info("Cudnn Version: {}".format(torch.backends.cudnn.version()))
    logger.info("use Model: [ {} ]".format(cfg.model_name))
    logger.info("use Data: [ {} ]".format(cfg.data_name))
    logger.info("data class number: [ {} ]".format(cfg.num_class))
    
    '''init'''
    if args.seed is not None:
        seed = args.seed
    else:
        seed = torch.randint(1, 10000,(1,))
    
    # seed = 9523
    logger.info("Seed: [ {} ]".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    '''MODEL LOADING'''
    logger.info('Load MODEL ---------------')
    model = getattr(importlib.import_module('models.{}'.format(cfg.model_copy_name)), 'Module')(cfg).cuda()
    if cfg.print_model:
        logger.info(model)

    optimizer = torch.optim.SGD(model.parameters(),lr=cfg.learning_rate,momentum=0.9,weight_decay=1e-4)
    lossfn = cal_loss
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate / 100)

    # try:
    #     checkpoint = torch.load('best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     logger.info('Use pretrain model')
    #     logger.info("=> loaded checkpoint '{}' (epoch {})"
    #                   .format(args.resume, checkpoint['epoch']))

    # except:
    #     logger.info('No existing model, starting training from scratch...')
    #     start_epoch = 0

    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        print('Use a local scope to avoid dangling references')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        

    '''DATA LOADING'''
    logger.info('Load dataset ---------------')
    TRAIN_DATASET, TEST_DATASET=dataset_get(cfg.data_dir, model_name = cfg.data_name, num_points=cfg.num_point)
    Train_DataLoader = DataLoader(TRAIN_DATASET, num_workers=cfg.workers, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    Test_DataLoader = DataLoader(TEST_DATASET, num_workers=cfg.workers, batch_size=cfg.test_batch_size, drop_last=True)
    
    
    global_epoch = 0
    best_test_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    '''TRANING'''
    logger.info('Start training...\n')
    writer = SummaryWriter(root_dir)
    cfg.logger_filename
    t1 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        epoch_t1 = time.time()
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, cfg.epochs))

        train_acc, train_loss = train(model, Train_DataLoader, optimizer, lossfn)
        logger.info('Train Accuracy: %f , Train Loss: %f'\
             % (train_acc, train_loss))
        writer.add_scalar('Test/train_Acc', train_acc, epoch)
        writer.add_scalar('Loss/train_Loss', train_loss, epoch)
        scheduler.step()


        test_acc, class_acc, test_loss= test(model, Test_DataLoader, cfg.num_class, lossfn)
        # print(list(model.state_dict().keys()))
        # raise ValueError
        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        if (test_acc >= best_test_acc):
            best_test_acc = test_acc
        logger.info('Test Acc: %f, Class Acc: %f, Loss:%f, Best: [%f]'% (test_acc, class_acc, test_loss,best_test_acc))
        # logger.info('Best Accuracy: [%f], Class Accuracy: [%f]'% (best_test_acc, best_class_acc))   
        if (test_acc >= best_test_acc):
            if (test_acc >= 0.8):
                best_epoch = epoch + 1
                savepath = root_dir+'/best_model.pth'
                logger.info('Save model..., Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'test_acc': test_acc,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        writer.add_scalar('Test/Test_Acc', test_acc, epoch)
        writer.add_scalar('Best/Best_Acc', best_test_acc, epoch)
        writer.add_scalar('Test/ClassAcc', class_acc, epoch)
        writer.add_scalar('Best/Best_ClassAcc', best_class_acc, epoch)
        writer.add_scalar('Loss/test_loss', test_loss, epoch)

        global_epoch += 1
        epoch_t2 = time.time()
        logger.info('%.4f h left'%((epoch_t2-epoch_t1)/3600*(cfg.epochs-epoch-1)))

    logger.info('End of training...')
    t2 = time.time()
    logger.info('trian and eval model time is %.4f h'%((t2-t1)/3600))
    writer.close()
    logger.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')


def train(model, Train_DataLoader, optimizer, lossfn):
    model.train()
    correct = 0
    epoch_loss = 0 
    num_len = len(Train_DataLoader.dataset)
    Train_DataLoader = tqdm(Train_DataLoader, ncols=100)

    for points, label in Train_DataLoader:
        points, label = points.cuda(), label.squeeze(-1).cuda()
        pred = model(points)
        loss = lossfn(pred, label.long())
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #防止nan
        optimizer.step()
        # 计算
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum()
        epoch_loss+=loss
        # points, target = prefetcher.next()
    train_instance_acc = correct / num_len
    epoch_loss = epoch_loss / len(Train_DataLoader)
    return train_instance_acc, epoch_loss

def test(model, Test_DataLoader, num_class=40, lossfn=N):
    model.eval()# 一定要model.eval()在推理之前调用方法以将 dropout 和批量归一化层设置为评估模式。否则会产生不一致的推理结果。
    class_acc = torch.zeros((num_class,3)).cuda()
    num_len = len(Test_DataLoader.dataset)
    correct=0
    with torch.no_grad():
        Test_DataLoader = tqdm(Test_DataLoader, ncols=100)
        for points, label in Test_DataLoader:
            points, label = points.cuda(), label.squeeze(-1).cuda()
            pred = model(points)
            loss = lossfn(pred, label.long())
            pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            correct += pred.eq(label.view_as(pred)).sum()

            for cat in torch.unique(label):
                cat_idex = (label==cat)
                classacc = pred[cat_idex].eq(label[cat_idex].view_as(pred[cat_idex])).sum()
                class_acc[cat,0] += classacc
                class_acc[cat,1] += cat_idex.sum()
            # correct += pred.eq(target.view_as(pred)).cpu().sum()

        test_instance_acc=correct / num_len
        class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
        class_acc_t = torch.mean(class_acc[:,2])
    return test_instance_acc, class_acc_t, loss


if __name__ == '__main__':
    # 垃圾回收gc.collect() 返回处理这些循环引用一共释放掉的对象个数
    gc.collect()
    main()
