import glob
import argparse
import glob
import re

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.PathoGen import PathoGen
from utils import *


def summarize_metrics(metric_list, name):
    arr = np.array(metric_list)
    avg = arr.mean()
    std_val = arr.std(ddof=1) if len(arr) > 1 else 0.0
    max_val = arr.max()
    min_val = arr.min()
    print(f"{name} → Avg: {avg:.4f}, Std: {std_val:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description='MIL main parameters')

    # General params.
    parser.add_argument('--experiment_name', type=str, default='F', help='experiment name')
    parser.add_argument('--MIL_model', type=str, default='PathoGen',
                        choices=['ABMIL', 'CLAM',  'DSMIL', 'TransMIL', 'MaxPooling',
                                 'MeanPooling', ],
                        help='MIL model to use')
    parser.add_argument('--metric2save', type=str, default='f1',
                        choices=['acc', 'f1', 'auc', 'acc_auc', 'f1_auc', 'loss'],
                        help='metrics to save best model')
    parser.add_argument('--device_ids', type=str, default=0, help='gpu devices for training')
    parser.add_argument('--seed', type=int, default=3721, help='random seed')
    parser.add_argument('--fold', type=int, default=1, help='fold number')
    parser.add_argument(
        '--fold_list',
        type=int,
        nargs='+',
        default=[1],
        # default=[4],
        help='List of fold indices to run'
    )
    parser.add_argument('--num_classes', type=int, default=4, help='classification number')
    parser.add_argument('--dataset', type=str, default='PathoGen', help='classification number')

    # Progressive pseudo bag augmentation params.
    parser.add_argument('--split_data', action='store_false', help='use data split')
    parser.add_argument('--search_rate', type=int, default=10, help='search rate')
    parser.add_argument('--sample_rate', type=int, default=3, help='subset sample rate')

    # MIL training params.
    parser.add_argument('--epochs', type=int, default=200, help='MIL epochs to train in each round')
    parser.add_argument('--patience', type=int, default=20, help='MIL epochs to early stop')
    parser.add_argument('--lr_patience', type=int, default=8, help='MIL epochs to adjust lr')
    parser.add_argument('--max_lr', type=float, default=1e-4, help='MIL max learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='MIL min learning rate')

    # dir params.
    parser.add_argument('--csv_dir', type=str,
                        default=r'./csv/total',
                        help='csv dir to split data')

    parser.add_argument('--feat_dir', type=str,
                        default=r'./data',
                        help='train/val/test dir for features')

    parser.add_argument('--ckpt_dir', type=str,
                        default=r'./ckpt/F',
                        help='dir to save models')
    parser.add_argument('--test', action='store_false', help='use test dataset')
    parser.add_argument('--logger_dir', type=str,
                        default='./logger',
                        help='tensorboard dir')
    args = parser.parse_args()
    return args




class MILDataset(Dataset):
    def __init__(self, split, feat_dir,clinical_xlsx):
        self.slide_ids = [item[0] for item in split]

        self.texts = [item[1] for item in split]
        self.labels = [item[2] for item in split]

        self.feat_dir = feat_dir
        self.feat_files = self.get_feat()


        self.clinical_df = pd.read_excel(clinical_xlsx)

    def get_labels(self):
        return self.labels

    def get_feat(self):
        feat_files = {}
        for slide_id in self.slide_ids:
            feat_paths = glob.glob(os.path.join(self.feat_dir, str(slide_id) + '.pt*'))
            slide_feats = []
            for feat_path in feat_paths:
                slide_feats.append(feat_path)
            feat_files[slide_id] = slide_feats
        return feat_files


    def __getitem__(self, idx):
        slide_name = self.slide_ids[idx]





        EGFR_label = self.labels[idx]

        # if EGFR_label == 1.0 or EGFR_label == 2.0:
        #     EGFR_label = 1.0
        # else:
        #     EGFR_label = 0.0


        # ????
        # EGFR_label = 1.0 if EGFR_label > 0 else 0
        # ????
        # text = self.des[slide_name]
        text = self.texts[idx]

        feat_files = self.feat_files[slide_name]
        feats = torch.Tensor()
        for feat_file in feat_files:
            feat = torch.load(feat_file, map_location='cpu')
            try:
                feat = torch.from_numpy(feat)
            except:
                pass
            feats = torch.cat((feats, feat), dim=0)

        sample = {'slide_id': slide_name, 'feat': feats, 'EGFR_label': EGFR_label, 'text': text}
        return sample

    def __len__(self):
        return len(self.slide_ids)



def MIL_train_epoch(fold, epoch, model, optimizer, loader, criterion, device, num_classes):
    model.train()
    loss_all = 0.
    EGFR_logits = torch.Tensor()

    EGFR_labels = torch.Tensor()



    with tqdm(total=len(loader)) as pbar:
        for _, sample in enumerate(loader):
            optimizer.zero_grad()
            slide_id, feat, EGFR_label, text = sample['slide_id'], sample['feat'], sample['EGFR_label'], sample['text']
            if len(feat[0]) == 0:
                pbar.update(1)
                continue
            feat = feat.to(device)

            EGFR_label = EGFR_label.to(device)



            pred,_ = model(feat, text)
            loss = criterion(pred, EGFR_label.long())



            # calculate metrics
            EGFR_logits = torch.cat((EGFR_logits, pred.detach().cpu()), dim=0)


            EGFR_labels = torch.cat((EGFR_labels, EGFR_label.cpu()), dim=0)


            loss_all += loss.detach().item() * len(EGFR_label)
            # loss backward
            loss.backward()
            optimizer.step()

            egfr_acc, egfr_f1, egfr_roc_auc = calculate_metrics(EGFR_logits, EGFR_labels, num_classes)


            lr = optimizer.param_groups[0]['lr']

            pbar.set_description(
                '[Fold:{}, Epoch:{}] lr:{:.5f}, loss:{:.4f}, EGFR: acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                .format(fold, epoch, lr, loss_all / len(EGFR_label), egfr_acc, egfr_roc_auc, egfr_f1))

            pbar.update(1)


        egfr_acc, egfr_f1, egfr_roc_auc = calculate_metrics(EGFR_logits, EGFR_labels, num_classes)


        pbar.set_description(
            '[Fold:{}, Epoch:{}] lr:{:.5f}, loss:{:.4f}, EGFR: acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
            .format(fold, epoch, lr, loss_all / len(EGFR_label), egfr_acc, egfr_roc_auc, egfr_f1))

        pbar.update(1)


    return loss_all / len(
        EGFR_labels), egfr_acc, egfr_f1, egfr_roc_auc


def MIL_pred(fold, model, loader, criterion, device, num_classes, status='Val'):
    model.eval()
    loss_all = 0.
    EGFR_logits = torch.Tensor()

    EGFR_labels = torch.Tensor()


    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, sample in enumerate(loader):
                slide_id, feat, EGFR_label, text = sample['slide_id'], sample['feat'], sample['EGFR_label'], sample['text']
                if len(feat[0]) == 0:
                    pbar.update(1)
                    continue
                feat = feat.to(device)

                EGFR_label = EGFR_label.to(device)


                pred = model(feat, text)


                loss = criterion(pred, EGFR_label.long())


                EGFR_logits = torch.cat((EGFR_logits, pred.detach().cpu()), dim=0)


                EGFR_labels = torch.cat((EGFR_labels, EGFR_label.cpu()), dim=0)


                loss_all += loss.item() * len(EGFR_label)


                egfr_acc, egfr_f1, egfr_roc_auc = calculate_metrics(EGFR_logits, EGFR_labels, num_classes)


                pbar.set_description(
                    '[{} Fold:{}]  loss:{:.4f}, EGFR: acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                    .format(status, fold, loss_all / len(EGFR_label), egfr_acc, egfr_roc_auc, egfr_f1))

                pbar.update(1)


            if status == 'Val':

                egfr_acc, egfr_f1, egfr_roc_auc, egfr_mat = calculate_metrics(EGFR_logits, EGFR_labels, num_classes,
                                                                              confusion_mat=True)

            else:
                egfr_acc, egfr_f1, egfr_roc_auc, per_class_acc,per_class_f1,per_class_auc,egfr_mat = calculate_test_metrics(EGFR_logits, EGFR_labels, num_classes,
                                                                                   confusion_mat=True,


                                                                                   state='egfr', fold=fold)

            pbar.set_description(
                '[{} Fold:{}]  loss:{:.4f}, EGFR: acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                .format(status, fold, loss_all / len(EGFR_label), egfr_acc, egfr_roc_auc, egfr_f1))

            pbar.update(1)

    print()
    print(egfr_mat)


    return loss_all / len(
        EGFR_labels), egfr_acc, egfr_f1, egfr_roc_auc,egfr_mat


if __name__ == '__main__':
    args = get_args()

    # set device
    device = torch.device('cuda:{}'.format(args.device_ids))
    print('Using GPU ID: {}'.format(args.device_ids))

    # set random seed
    set_seed(args.seed)
    print('Using Random Seed: {}'.format(str(args.seed)))

    # set tensorboard
    args.logger_dir = os.path.join(args.logger_dir, args.experiment_name)
    os.makedirs(args.logger_dir, exist_ok=True)
    writer = SummaryWriter(args.logger_dir)
    print('Set Tensorboard: {}'.format(args.logger_dir))

    fold_list = args.fold_list
    EGFR_test_acc_list, EGFR_test_auc_list, EGFR_test_f1_list, EGFR_test_mat_list = [], [], [], []


    for fold in fold_list:
        csv_path = os.path.join(args.csv_dir, 'fold_{}.csv'.format(fold))  # dir to save label
        feat_dir = args.feat_dir


        if args.test:
            train_dataset, val_dataset, test_dataset = return_splits(csv_path=csv_path, test=True)
        else:
            train_dataset, val_dataset = return_splits(csv_path=csv_path, test=False)


        train_dset = MILDataset(train_dataset, feat_dir,args.clinical_xlsx)
        train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
        val_dset = MILDataset(val_dataset, feat_dir,args.clinical_xlsx)
        val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)
        if args.test:
            test_dset = MILDataset(test_dataset, feat_dir,args.clinical_xlsx)
            test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss()

        model_dir = os.path.join(args.ckpt_dir, args.experiment_name)
        os.makedirs(model_dir, exist_ok=True)

        if 'PathoGen' == args.MIL_model:
            model = PathoGen(n_classes=args.num_classes)

        else:
            raise NotImplementedError
        model = model.to(device)
        lr = args.max_lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        model_path = os.path.join(model_dir, '{}_model_{}.pth'.format(args.MIL_model, fold))

        if not os.path.exists(model_path):

            early_stopping = EarlyStopping(model_path=model_path, patience=args.patience, verbose=True,
                                           count_loss=False)
            for epoch in range(args.epochs):
                train_loss, egfr_acc, egfr_f1, egfr_roc_auc = MIL_train_epoch(
                    fold, epoch, model,
                    optimizer, train_loader,
                    criterion, device,
                    args.num_classes)

                val_loss, egfr_acc, egfr_f1, egfr_roc_auc,egfr_mat = MIL_pred(
                    fold, model, val_loader, criterion,
                    device, args.num_classes)
                if args.metric2save == 'acc':
                    counter = early_stopping(epoch, val_loss, model, egfr_acc)
                elif args.metric2save == 'f1':
                    counter = early_stopping(epoch, val_loss, model, egfr_f1)
                elif args.metric2save == 'auc':
                    counter = early_stopping(epoch, val_loss, model, egfr_roc_auc)
                elif args.metric2save == 'acc_auc':
                    counter = early_stopping(epoch, val_loss, model, (egfr_acc + egfr_roc_auc) / 2)
                elif args.metric2save == 'f1_auc':
                    counter = early_stopping(epoch, val_loss, model, (egfr_f1 + egfr_roc_auc) / 2)
                elif args.metric2save == 'loss':
                    counter = early_stopping(epoch, val_loss, model)
                else:
                    raise NotImplementedError


                if early_stopping.early_stop:
                    print('Early Stopping')
                    break
                if counter > 0 and counter % args.lr_patience == 0:
                    if lr > args.min_lr:
                        early_stopping.reset()
                        lr = lr / 10 if lr / 10 >= args.min_lr else args.min_lr
                        for params in optimizer.param_groups:
                            params['lr'] = lr
        pretrained_dict = torch.load(model_path, map_location='cpu')


        model_dict = model.state_dict()


        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}


        model.load_state_dict(pretrained_dict, strict=False)

        if args.test:
            test_loss, egfr_acc, egfr_f1, egfr_roc_auc, egfr_mat= MIL_pred(
                fold, model, test_loader, criterion,
                device, args.num_classes, 'Test')
            draw_metrics(writer, 'Test', args.num_classes, test_loss, egfr_acc, egfr_roc_auc, egfr_mat, egfr_f1, fold)

            EGFR_test_acc_list.append(egfr_acc)
            EGFR_test_auc_list.append(egfr_roc_auc)
            EGFR_test_f1_list.append(egfr_f1)
            EGFR_test_mat_list.append(egfr_mat)



    print(EGFR_test_acc_list)
    print(EGFR_test_auc_list)
    print(EGFR_test_f1_list)
    summarize_metrics(EGFR_test_acc_list, 'Test Accuracy')
    summarize_metrics(EGFR_test_auc_list, 'Test AUC')
    summarize_metrics(EGFR_test_f1_list, 'Test F1')
    print(EGFR_test_mat_list)


