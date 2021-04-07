import random
import torch
import time
import numpy as np
import os
from pathlib import Path
import shutil
from tqdm import tqdm
from .models import *
from .loss import CELoss, DistillationLoss
from .optim import SGD, Adam, Step
from .dataset import MyDataset
from .seed import fix_all_seed
from .metric import accuracy, Meter_cls, Meter_ood
try:
    from torch.cuda import amp
    AMP = True
except:
    AMP = False
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy

class trainManager(object):
    def __init__(self, args):
        fix_all_seed(args.seed)
        self.best_prec = 0
        try:
            pretrained = args.pretrained if hasattr(args, 'pretrained') else False
            # self.model = eval(args.model_arch)(num_classes=args.num_classes, pretrained=pretrained)
            self.model = create_model(
                args.model_arch,
                pretrained=False,
                num_classes=args.num_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
            )
            if args.resume != '':
                self.resume(args)
            self.teacher_model = None
            if args.distillation_type != 'none':
                self.teacher_model = resnet34(num_classes=args.num_classes)
                ckp = torch.load(args.teacher_path, map_location='cpu')
                self.teacher_model.load_state_dict(ckp['state_dict'])
                self.teacher_model.to(args.device)
                self.teacher_model.eval()
        except Exception as e:
            print(e)
            print('Model not implemented')
            self.model = None
        try:
            self.criterion = eval(args.loss)(args)
            self.criterion = DistillationLoss(
                self.criterion, self.teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
            )
            self.criterion_eval = eval(args.loss)(args)
        except:
            print('Criterion not implemented')
            self.criterion = None
        try:
            self.meter = eval(args.averagemeter)()
        except:
            print('Meter not implemented')
            self.meter = None
        try:
            self.metric = accuracy
        except:
            print('Metric not implemented')
            self.metric = None
        try:
            self.optimizer = self.get_optimizer(args)
        except:
            print('Optimizer not implemented')
            self.optimizer = None
        try:
            self.scheduler = self.get_scheduler(args)
        except:
            print('Scheduler not implemented')
            self.scheduler = None
        try:
            self.set_data_loader(args, 'train', shuffle=True)
        except Exception as e:
            print(e)
            print('Train loader not implemented')
            self.train_loader = None
        try:
            self.set_data_loader(args, 'val', shuffle=False)
        except Exception as e:
            print(e)
            print('Val loader not implemented')
            self.val_loader = None
        if self.model:
            if args.device == 'cuda':
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(args.device)
        self.args = args
        self.scaler = amp.GradScaler(enabled=AMP)

    def resume(self, args):
         ckp = torch.load(args.resume, map_location='cpu')
         # state_dict = {k: v for k, v in ckp['state_dict'].items() if
         #               k in self.model.state_dict() and self.model.state_dict()[k].numel() == v.numel()}
         # self.model.load_state_dict(state_dict, strict=False)
         self.model.load_state_dict(ckp['state_dict'])
         if not args.reset_epoch:
             args.start_epoch = ckp['epoch']
             self.best_prec = ckp['best_accu']

    def get_optimizer(self, args):
        if hasattr(self, 'optimizer'):
            print('Re-defining optimizer')
        return eval(args.optim)(self.model, args.lr)

    def get_scheduler(self, args):
        if hasattr(self, 'scheduler'):
            print('Re-defining scheduler')
        return eval(args.scheduler)(self.optimizer, args)

    def set_data_loader(self, args, mode, shuffle):
        dset = MyDataset(args, mode=mode)
        if not isinstance(dset, torch.utils.data.Dataset):
            raise TypeError
        setattr(self, f'{mode}_loader', torch.utils.data.DataLoader(
                dset, batch_size=args.batch_size, shuffle=shuffle,
                num_workers=args.num_workers, pin_memory=True))

    def fit(self):
        start_time = time.time()
        if self.args.mode != 'eval':
            for epoch in range(self.args.start_epoch, self.args.epochs):
                train_accu, train_loss = self.train(epoch)
                val_accu, val_loss = self.validate(epoch)
                self.scheduler.step()
                is_best = val_accu > self.best_prec
                if is_best:
                    self.best_prec = val_accu
                if epoch % self.args.save_freq == 0 or epoch == self.args.epochs - 1 or is_best:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'config': self.args.__dict__,
                        'state_dict': self.model.state_dict() if self.args.device != 'cuda' else self.model.module.state_dict(),
                        'best_accu': self.best_prec,
                        'optimizer': self.optimizer.state_dict(),
                    },
                        os.path.join(self.args.out_dir, self.args.model_dir), 'checkpoint.pth', is_best)
            print('Best accuracy: {}, time {}'.format(self.best_prec, time.time() - start_time))
        elif self.args.mode == 'eval':
            val_accu, val_loss = self.validate(0)
            print(f'Validation accuracy: {val_accu}')

    def save_checkpoint(self, state, folder, filename='checkpoint.pth', is_best=False):
        torch.save(state, os.path.join(folder, filename))
        if is_best:
            shutil.copy(os.path.join(folder, filename), os.path.join(folder, 'best_checkpoint.pth'))

    def train(self, epoch):
        self.meter.reset()
        self.model.train(True)
        tbar = tqdm(self.train_loader, mininterval=self.args.tqdm_minintervals)
        tic = time.time()
        for i, data in enumerate(tbar):
            data_time = time.time() - tic
            input_ = data['x'].to(self.args.device)  # batch, 3, 256, 256
            target = data['y'].to(self.args.device)
            with amp.autocast(enabled=AMP):
                output = self.model(input_)
                loss = self.criterion(input_, output, target)  # , target_cls)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses = loss.item()
            accs = self.metric(output, target)
            self.meter.update(losses=losses, top1=accs[0], topK=accs[1], batch_time=time.time() - tic,
                             data_time=data_time)
            tbar.set_description(
                    'Epoch: {}/{}| {}'.format(epoch, i, self.meter))
            tic = time.time()
        return self.meter.top1.avg, self.meter.topK.avg

    # @torch.no_grad()
    def validate(self, epoch):
        self.meter.reset()
        self.model.eval()
        tbar = tqdm(self.val_loader, mininterval=self.args.tqdm_minintervals)
        tic = time.time()
        with torch.no_grad():
            for i, data in enumerate(tbar):
                data_time = time.time() - tic
                input_ = data['x'].to(self.args.device)  # batch, 3, 256, 256
                target = data['y'].to(self.args.device)
                output = self.model(input_)
                loss = self.criterion_eval(output, target)  # , target_cls)
                losses = loss.item()

                accs = self.metric(output, target)
                self.meter.update(losses=losses, top1=accs[0], topK=accs[1], batch_time=time.time() - tic,
                             data_time=data_time)
                tbar.set_description(
                    'Val: {}/{}| {}'.format(epoch, i, self.meter))
                tic = time.time()
        return self.meter.top1.avg, self.meter.losses.avg

    def inference_ood(self, epoch):
        self.meter.reset()
        self.model.eval()
        tbar = tqdm(self.val_loader, mininterval=self.args.tqdm_minintervals)
        tic = time.time()
        with torch.no_grad():
            for i, data in enumerate(tbar):
                data_time = time.time() - tic
                input_ = data['x'].to(self.args.device)  # batch, 3, 256, 256
                target = data['y'].to(self.args.device)
                output = self.model(input_)
                loss = self.criterion(input_, output, target)  # , target_cls)
                losses = loss.item()

                accs = self.metric(output, target)
                self.meter.update(losses=losses, top1=accs[0], topK=accs[1],
                                  batch_time=time.time() - tic, data_time=data_time)
                self.meter.update_score(output)
                tbar.set_description(
                    'Val: {}/{}| {}'.format(epoch, i, self.meter))
                tic = time.time()
        return self.meter.top1.avg, self.meter.losses.avg

    def fit_ood(self):
        val_accu, val_loss = self.inference_ood(0)
        print(f'Validation accuracy: {val_accu}')
        scores = self.meter.scores
        scores = np.asarray(scores)
        np.save(os.path.join(self.args.out_dir, f'{Path(self.args.val_file).stem}.npy'), scores)


