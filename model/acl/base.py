import abc
import torch
import os.path as osp
import os
import pandas
from dataloader.dataloader import *
from copy import deepcopy
import time


from utils.utils import (
    ensure_path,
    Averager, Timer, count_acc, cal_auxIndex
)


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = self.set_up_datasets()
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] #* args.sessions
        self.sess_acc_dict = {}
        self.result_list = [args]


    def set_up_datasets(self):
       
        if self.args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
            import dataloader.s2s.s2s as Dataset
        elif self.args.dataset == 'nsynth-100':
            import dataloader.nsynth.nsynth as Dataset
        elif self.args.dataset == 'librispeech':
            import dataloader.librispeech.librispeech as Dataset
        self.args.Dataset=Dataset
    
    def save_better_model(self, va, net_dict, session, save_model_dir=None):
        # save better model
        if (va * 100) >= self.trlog['max_acc'][session]:
            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
            self.trlog['max_acc_epoch'] = net_dict['epoch']
            if save_model_dir is None:
                save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
            if self.args.debug:
                # torch.save(dict(params=self.model.state_dict()), save_model_dir)
                # torch.save(net_dict['optimizer'].state_dict(), os.path.join(self.args.save_path, 'optimizer_best.pth'))
                pass
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('********A better model is found!!**********')
            print('Saving model to :%s' % save_model_dir)
        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                            self.trlog['max_acc'][session]))
    def record_info(self, va, vl, net_dict, res_dict, start_time, epochs):
        self.trlog['val_loss'].append(vl)
        self.trlog['val_acc'].append(va)
        lrc = net_dict['scheduler'].get_last_lr()[0]
        self.result_list.append(
            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                net_dict['epoch'], lrc, res_dict['tl'], res_dict['ta'], vl, va))
        self.trlog['train_loss'].append(res_dict['tl'])
        self.trlog['train_acc'].append(res_dict['ta'])
        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
            net_dict['epoch'], lrc, res_dict['tl'], res_dict['ta'], vl, va))
        print('This epoch takes %d seconds' % (time.time() - start_time),
                '\nstill need around %.2f mins to finish' % (
                        (time.time() - start_time) * (epochs - net_dict['epoch']) / 60))
    
    def data_init(self, data_dict, session):
        # for base model
        self.model.load_state_dict(self.best_model_dict)
        self.model = self.replace_base_fc(get_dataset_for_data_init(self.args), self.model)
        best_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
        print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
        self.best_model_dict = deepcopy(self.model.state_dict())
        torch.save(dict(params=self.model.state_dict()), best_model_dir)

        if self.args.debug:
            # torch.save(dict(params=self.model.state_dict()), best_model_dir)
            pass
    
    def replace_base_fc(self, trainset, model):
        num_base_class = self.args.stdu.num_tmpb if self.args.tmp_train else self.args.num_base
        model = model.eval()
        assert len(set(trainset.targets)) == num_base_class
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                num_workers=8, pin_memory=True, shuffle=False)
        # trainloader.dataset.transform = transform
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                data = model.module.get_mel(data)
                
                embedding, _ = model(data)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []
        for class_index in range(num_base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        proto_list = torch.stack(proto_list, dim=0)

        model.module.fc.mu.data[:num_base_class] = proto_list

        return model 
       
    def pretty_output(self, save=True, print_output=True):
        final_out_dict = {}
        final_out_dict['cur_acc'] = []
        final_out_dict['base_Acc'] = []
        final_out_dict['novel_Acc'] = []
        final_out_dict['Both_ACC'] = []
        for k, v in self.sess_acc_dict.items():
            final_out_dict['cur_acc'].append(v['cur_acc'])
            final_out_dict['base_Acc'].append(v['base_acc'])
            if 'novel_acc' in v:
                final_out_dict['novel_Acc'].append(v['novel_acc'])
            else:
                final_out_dict['novel_Acc'].append(None)
            final_out_dict['Both_ACC'].append(v['all_acc'])
        cpi, msr_overall, acc_aver_df, ar_over = cal_auxIndex(final_out_dict)
        pd = final_out_dict['Both_ACC'][0] - final_out_dict['Both_ACC'][-1]
        indexes = {'PD':pd, 'CPI':cpi, 'AR':ar_over, 'MSR':msr_overall}
        indexes_df = pandas.DataFrame.from_dict(indexes, orient='index')
        final_df = pandas.DataFrame(final_out_dict)
        final_df = final_df.T
        # pretty output
        pandas.set_option('display.max_rows', None)
        pandas.set_option('display.max_columns', None)
        pandas.set_option('display.width', None)
        pandas.set_option('display.max_colwidth', None)
        if save:
            excel_fn = os.path.join(self.args.save_path, "output.xlsx")
            print("save output at ", excel_fn)
            writer = pandas.ExcelWriter(excel_fn)
            final_df.to_excel(writer, sheet_name='final_df')
            acc_aver_df.to_excel(writer, sheet_name='final_df', startrow=7)
            indexes_df.to_excel(writer, sheet_name='final_df', startrow=13)
            indexes_df.T.to_excel(writer, sheet_name='final_df', startrow=20)
            writer.save()

        output = f"\nreslut on {self.args.dataset}, method {self.args.project}\
                    \n{self.args.save_path}\
                    \n****************************************Pretty Output********************************************\
                    \n{final_df}\
                    \n===> Comprehensive Performance Index(CPI) v2: {cpi}\n===> PD: {pd}\
                    \n===> Memory Strock Ratio(MSR) Overall: {msr_overall}\n===> Amnesia Rate(AR): {ar_over}\
                    \n===> Acc Average: \n{acc_aver_df}\
                    \n***********************************************************************************************"
        if print_output:
            print(output)
        return output, acc_aver_df, final_df

    @abc.abstractmethod
    def train(self):
        pass