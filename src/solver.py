import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import sys
import torch.optim as optim
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import manifold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import MMIM
from transformer import CMD,DiffLoss,MSE
from tensorboardX import SummaryWriter

# 设置散点形状
maker = ['o', 'o', 'o', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['#0000FF','#FFA500', '#008000', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def vv(feat):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    #print(S_data)
    #print(S_data.shape)  # [num, 3]

    for index in range(3):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
    plt.rcParams.update({'font.size': 20})
    plt.legend(labels=['Modality m$_1$', 'Modality m$_2$','Modlaity m$_3$'], loc='upper right')  # best表示自动分配最佳位置
    #plt.title(name, fontsize=32, fontweight='normal', pad=20)

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        # Training hyperarams
        self.alpha = hp.alpha
        self.beta = hp.beta

        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = model = MMIM(hp)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        # criterion
        if self.hp.dataset == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        else: # mosi and mosei are regression datasets
            #
            self.criterion = criterion = nn.SmoothL1Loss(reduction="mean")
        
        # optimizer
        self.optimizer={}

        if self.is_train:

            main_param = []
            bert_param = []

            for name, p in model.named_parameters():
                # print(name)
                if p.requires_grad:
                    if 'bert' in name:
                        bert_param.append(p)


                    else: 
                        main_param.append(p)
                
                for p in (main_param):
                    if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                        #nn.init.xavier_normal_(p,gain=1)
                        nn.init.kaiming_normal_(p, mode='fan_in')

        '''self.optimizer_mmilb = getattr(torch.optim, self.hp.optim)(
            main_param, lr=self.hp.lr_mmilb, weight_decay=hp.weight_decay_club)'''
        
        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': hp.weight_decay_bert, 'lr': hp.lr_bert},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        ]

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        #self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=hp.when, factor=0.5, verbose=True)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        #optimizer_mmilb = self.optimizer_mmilb
        optimizer_main = self.optimizer_main

        #scheduler_mmilb = self.scheduler_mmilb
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        # entropy estimate interval
        mem_size = 1

        def train(model, optimizer, criterion, stage=1):
            epoch_loss = 0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size
            proc_loss, proc_size = 0, 0
            nce_loss = 0.0
            ba_loss = 0.0
            start_time = time.time()

            left_batch = self.update_batch

            if self.hp.add_va:
                mem_pos_va = []
                mem_neg_va = []

            for i_batch, batch_data in enumerate(self.train_loader):
                text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
                #print(alens)

                # for mosei we only use 50% dataset in stage 1
                if self.hp.dataset == "mosei":
                    if stage == 0 and i_batch / len(self.train_loader) >= 0.5:
                        break
                model.zero_grad()

                with torch.cuda.device(0):
                    text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                    text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                    bert_sent_type.cuda(), bert_sent_mask.cuda()
                    if self.hp.dataset=="ur_funny":
                        y = y.squeeze()
                
                batch_size = y.size(0)

                if stage == 0:
                    y = None
                    mem = None


                preds,pre1,pre2,pre3,pre4,x_shared,y_shared,z_shared,fusion,preds1,x_pre,y_pre,z_pre,x_dif,y_dif,z_dif,xre,yre,zre,x_sin,y_sin,z_sin,t,v,a,x0,y0,z0 = model(text, visual, audio, vlens, alens,
                                                bert_sent, bert_sent_type, bert_sent_mask, y)



                valid = Variable(torch.Tensor(pre1.size()).fill_(1.0), requires_grad=False).cuda()
                fake = Variable(torch.Tensor(pre2.size()).fill_(0.0), requires_grad=False).cuda()
                a_loss=nn.BCELoss()
                G1_loss = a_loss(pre1, valid) + a_loss(pre2, fake)
                G2_loss = a_loss(pre3, valid) + a_loss(pre4, fake)
                b_loss=CMD()
                c_loss=DiffLoss()
                d_loss=nn.SmoothL1Loss()
                f_loss=MSE()
                g_loss=nn.L1Loss()
                y_loss=(4/3)*d_loss(preds,y)
                closs1=(2/3)*d_loss(preds1,y)
                '''closs2 = criterion(x_pre, y)
                closs3 = criterion(y_pre, y)
                closs4 = criterion(z_pre, y)'''
                #dif1=c_loss(x_dif,y_dif)
                #dif2=c_loss(x_dif,z_dif)
                #dif3=c_loss(z_dif,y_dif)
                '''lrecx=f_loss(xre,visual)
                lrecy=f_loss(yre,text)
                lrecz=f_loss(zre,acoustic)'''
                dif1=c_loss(x_dif,y_dif)
                dif2=c_loss(z_dif,x_dif)
                dif3=c_loss(y_dif,z_dif)
                '''l1=b_loss(x_s,y_s,3)
                l2 = b_loss(z_s, y_s, 3)
                l3 = b_loss(x_s, z_s, 3)'''

                l4 = b_loss(x_shared,y_shared, 2)
                l5 = b_loss(y_shared, z_shared, 2)
                l6 = b_loss(y_shared, fusion, 2)
                #loss=y_loss+(G1_loss+G2_loss)/4+closs1+(l4+l5+l6)/3
                loss = y_loss + closs1 + (l4 + l5 + l6) /3+(dif1+dif2+dif3)/4

                loss.backward()


                
                left_batch -= 1
                if left_batch == 0:
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                    optimizer.step()
                
                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                #nce_loss += nce.item() * batch_size
                #ba_loss += (-H - lld) * batch_size

                if i_batch == 5 and i_batch > 0:
                #if i_batch  == 50 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    avg_nce = nce_loss / proc_size
                    avg_ba = ba_loss / proc_size
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, 'TASK+BA+CPC' if stage == 1 else 'Neg-lld',
                        avg_loss, avg_nce, avg_ba))
                    proc_loss, proc_size = 0, 0
                    nce_loss = 0.0
                    ba_loss = 0.0
                    start_time = time.time()
                    com=torch.cat((y0,x0),dim=0).detach().cpu()
                    com=torch.cat((com,(z0).detach().cpu()),dim=0)
                    #print(feat.size())
                    label_test1 = [0 for index in range(210)]
                    label_test2 = [1 for index in range(210)]
                    label_test3 = [2 for index in range(210)]

                    label_test = np.array(label_test1 + label_test2 + label_test3)
                    #print(label_test)
                    #print(label_test.shape)

                    fig = plt.figure(figsize=(10, 10))

                    plotlabels(vv(com), label_test, '(a)')

                    fig.savefig('/home/hj/sentimosei/Multimodal-Infomax-main/src/f_sup/pic-{}.pdf'.format(epoch ))

                    
            return epoch_loss / self.hp.n_train

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0
        
            results = []
            truths = []

            with torch.no_grad():
                for i_batch, batch_data in enumerate(loader):
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data

                    '''
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch
                    #print(text.size(),y.size(),batch.size())'''

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        #print(vision.size())
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                        if self.hp.dataset == 'iemocap':
                            y = y.long()
                    
                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)

                    # we don't need lld and bound anymore
                    #preds,_, _, _, _ = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)
                    preds, pre1, pre2, pre3, pre4,x_shared,y_shared,z_shared,fusion,preds1,x_pre,y_pre,z_pre,x_dif,y_dif,z_dif,xre,yre,zre,visual,text,acoustic,t,v,a,x0,y0,z0 = model(text, vision, audio, vlens, alens,
                                                          bert_sent, bert_sent_type, bert_sent_mask, y)
                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.SmoothL1Loss()

                    total_loss += criterion(preds, y).item()* batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

        best_valid = 1e8
        best_test = 1e8
        best_mae = 1e-8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            # maximize likelihood
            if self.hp.contrast:
                train_loss = train(model, optimizer_mmilb, criterion, 0)

            # minimize all losses left
            train_loss = train(model, optimizer_main, criterion, 1)

            val_loss, _, _ = evaluate(model, criterion, test=False)
            test_loss, results, truths = evaluate(model, criterion, test=True)
            
            end = time.time()
            duration = end-start
            scheduler_main.step(val_loss)    # Decay learning rate by validation loss

            # validation F1
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                # update best validation
                patience = self.hp.patience
                best_valid = val_loss

                # for ur_funny we don't care about
                if self.hp.dataset == "ur_funny":
                    accu=eval_humor(results, truths, True)


                if self.hp.dataset in ["mosei_senti", "mosei"]:
                    accu=eval_mosei_senti(results, truths, True)

                elif self.hp.dataset == 'mosi':
                    accu=eval_mosi(results, truths, True)
                elif self.hp.dataset == 'iemocap':
                        accu=eval_iemocap(results, truths)
                if accu >= best_mae:
                    best_epoch = epoch
                    best_mae = accu
                    best_results = results
                    best_truths = truths
                    print(f"Saved model at pre_trained_models/MM.pt!")
                    save_model(self.hp, model)
            else:
                patience -= 1
                if patience == 0:
                    break

        print(f'Best epoch: {best_epoch}')
        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(best_results, best_truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(best_results, best_truths, True)
        elif self.hp.dataset == 'iemocap':
            eval_iemocap(results, truths)       
        sys.stdout.flush()