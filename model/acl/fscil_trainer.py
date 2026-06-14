import os
import time
import numpy as np
from utils.utils import ensure_path
from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
import torch
from .helper import *
from dataloader.dataloader import get_base_dataloader_meta, get_new_dataloader, get_task_specific_testloader, get_testloader, get_novel_testloader, get_pretrain_dataloader
import torch.distributions.normal as normal
import matplotlib.pyplot as plt
from dataloader.sampler import  SupportsetSampler
from .standard_train_helper import standard_base_train, standard_test


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.set_up_datasets()
        
        self.cov_mats, self.base_cov_mats = [], []
        self.proto_list = []
        

        self.model = MYNET(self.args, mode=self.args.network.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.old_model = MYNET(self.args, mode=self.args.network.base_mode)
        self.old_model = nn.DataParallel(self.old_model, list(range(self.args.num_gpu)))
        self.old_model = self.old_model.cuda()

        if self.args.s0_model_dir is not None:
            print('Loading init parameters from: %s' % self.args.s0_model_dir)
            self.best_model_dict = torch.load(self.args.s0_model_dir)['params']
           
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())
        

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.optimizer.decay)
        
        if self.args.scheduler.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler.step, gamma=self.args.scheduler.gamma)
        elif self.args.scheduler.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.scheduler.milestones,
                                                             gamma=self.args.scheduler.gamma)
        elif self.args.scheduler.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs.epochs_base)
        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            if "stand" in self.args.prefix:
                trainset, valset, trainloader, valloader = get_pretrain_dataloader(self.args)
            else:  
                trainset, valset, trainloader, valloader = get_base_dataloader_meta(self.args)
            
        else:
            trainset, valset, trainloader, valloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, valloader

    def standard_train(self, pretrained=False):
        session = 0
        data_dict = {}
        data_dict['train_set'], data_dict['valset'], data_dict['trainloader'], data_dict['valloader'] \
            = get_pretrain_dataloader(self.args)
        data_dict['testset'], data_dict['testloader'] = get_testloader(self.args, session)

        net_dict = {}
        print('==> Classes for this standard train stage:\n', np.unique(data_dict['train_set'].targets))

        net_dict['optimizer'], net_dict['scheduler'] = self.get_optimizer_base()

        """****************train and val*************************"""
        tsl, tsa = 0, 0
        if not pretrained:
            for epoch in range(self.args.epochs.epochs_pre):
                std_start_time = time.time()
                tl, ta = standard_base_train(self.args, self.model, data_dict['trainloader'], net_dict['optimizer'], net_dict['scheduler'], epoch)
                net_dict['epoch'] = epoch
                res_dict = {'tl': tl, 'ta': ta}
                tsl, tsa, acc_dict, cls_sample_count = standard_test(self.args, self.model, data_dict['testloader'], epoch, session)
                # set save path
                save_model_path = os.path.join(self.args.save_path, f'std_train{self.args.epochs.epochs_pre}_max_acc.pth')
                self.save_better_model(tsa, net_dict, session, save_model_path)
                self.record_info(tsa, tsl, net_dict, res_dict, std_start_time, self.args.epochs.epochs_pre)
                net_dict['scheduler'].step()

            """****************record on best model*************************"""
            stage_flag = "Standard(not load)"
            self.result_list.append('{} standard train stage, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        stage_flag, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
            print('{} test loss={:.3f}, test acc={:.3f}'.format(stage_flag, tsl, tsa))

        """****************data init and test again*************************"""
        if self.args.strategy.data_init:
            #data init and replace the model
            self.data_init(data_dict, session)
            self.model.module.mode = 'avg_cos'
            tsl, tsa, acc_dict, cls_sample_count = standard_test(self.args, self.model,data_dict['testloader'], 0, session)
            # self.sess_acc_dict[f'sess {session}'] = acc_dict
            if (tsa * 100) >= self.trlog['max_acc'][session]:
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('The NEW(after data init) best test acc of standard train stage={:.3f}'.format(self.trlog['max_acc'][session]))


        self.result_list.append(f"==> Standard train stage: Best epoch:{self.trlog['max_acc_epoch']}, \
            Best acc:{self.trlog['max_acc']}")
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)
    
    def train(self):
        
        conloss = []
        totalloss = []
        epochs= []
        oploss = []
        celoss = []
        
        args = self.args
        start_time = time.time()
        # init train statistics
        self.result_list = [args]
        # if self.args.s0_model_dir is None:
        #     self.standard_train()
            
        
        if args.start_session == 0:
            session = 0
            train_set, trainloader, valloader = self.get_dataloader(session=0)
            self.model.load_state_dict(self.best_model_dict)
            best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            print('new classes for this session:\n', np.unique(train_set.targets))
            optimizer, scheduler = self.get_optimizer_base()
            for epoch in range(args.epochs.epochs_base):
               
                start_time = time.time()
                tl, ta, tcont_1, te,  tcont_2, ttotal, all_embeddings, all_labels = stand_base_train(self.model, trainloader, optimizer, scheduler, epoch, args)            
                
                totalloss.append(tl)
                epochs.append(epoch)
                celoss.append(tl)
                # test model with all seen class
                tsl, tsa, da,  _, acc_dict = test_agg(self.model, valloader, epoch, args, session)
                self.sess_acc_dict[f'sess {session}'] = acc_dict

                # save better model
                if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    self.trlog['max_acc_epoch'] = epoch
                    
                    # Select 10 random classes
                    # unique_classes = torch.unique(all_labels)
                    # selected_classes = torch.randperm(len(unique_classes))[:15]
                    # selected_classes_labels = unique_classes[selected_classes]
                    # print("**************************",selected_classes_labels)
                    save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    print('********A better model is found!!**********')
                    print('Saving model to :%s' % save_model_dir)
                print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                    self.trlog['max_acc'][session]))

                self.trlog['train_loss'].append(tl)
                self.trlog['train_acc'].append(ta)
                self.trlog['test_loss'].append(tsl)
                self.trlog['test_acc'].append(tsa)
                lrc = scheduler.get_last_lr()[0]
                self.result_list.append(
                    'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                        epoch, lrc, tl, ta, tsl, tsa))
                print('epoch:%03d,lr:%.4f,training_total_loss:%.5f, training_mixup_loss_1:%.5f,training_ce_loss:%.5f, training_kl_loss_2:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                        epoch, lrc, tl, tcont_1,te,tcont_2,ta, tsl, tsa))
                
                print('This epoch takes %d seconds' % (time.time() - start_time),
                        '\nstill need around %.2f mins to finish this session' % (
                                (time.time() - start_time) * (args.epochs.epochs_base - epoch) / 60))
                scheduler.step()
    

            self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
            
            if args.strategy.data_init:

                print("Updating old class with class means ")
                self.model.load_state_dict(self.best_model_dict)
                self.model = replace_base_fc(train_set, self.model, args)
                
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                session0_best_model_dict = deepcopy(self.model.state_dict())
                torch.save(dict(params=self.model.state_dict()), best_model_dir)

                self.model.module.mode = 'avg_cos'
                tsl, tsa, da, _ , acc_dict = test_agg(self.model, valloader, 0, args, session)

                self.sess_acc_dict[f'sess {session}'] = acc_dict
                if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                self.model.load_state_dict(self.best_model_dict)


        acc_array = np.zeros([self.args.test_times, 4, self.args.num_session], dtype=float)   #[100, 4, 10]
        for i in range(self.args.test_times):
            tmp_acc_df = self.evaluate(session0_best_model_dict)
            acc_array[i] = tmp_acc_df.to_numpy()
        acc_array = acc_array.mean(0)


        print("\n\n\nFinal result:")
        self.result_list.append("\n\n\nFinal result:")
        cpi, msr_overall, acc_aver_df, ar_over = cal_auxIndex_from_numpy(acc_array)
        pd = acc_array[3][0] - acc_array[3][-1]
        indexes = {'PD':pd, 'CPI':cpi, 'AR':ar_over, 'MSR':msr_overall}
        indexes_df = pandas.DataFrame.from_dict(indexes, orient='index')
        final_df = pandas.DataFrame(acc_array)
        pandas.set_option('display.max_rows', None)
        pandas.set_option('display.max_columns', None)
        pandas.set_option('display.width', None)
        pandas.set_option('display.max_colwidth', None)

        excel_fn = os.path.join(self.args.save_path, "output.xlsx")
        print("save output at ", excel_fn)
        writer = pandas.ExcelWriter(excel_fn)
        final_df.to_excel(writer, sheet_name='final_df')
        acc_aver_df.to_excel(writer, sheet_name='final_df', startrow=7)
        indexes_df.to_excel(writer, sheet_name='final_df', startrow=13)
        indexes_df.T.to_excel(writer, sheet_name='final_df', startrow=20)
        writer.close()

        output = f"\nresult on {self.args.dataset}, method {self.args.project}\
                    \n{self.args.save_path}\
                    \n****************************************Pretty Output********************************************\
                    \n{final_df}\
                    \n===> Comprehensive Performance Index(CPI) v2: {cpi}\n===> PD: {pd}\
                    \n===> Memory Strock Ratio(MSR) Overall: {msr_overall}\n===> Amnesia Rate(AR): {ar_over}\
                    \n===> Acc Average: \n{acc_aver_df}\
                    \n***********************************************************************************************"
        print(output)
        self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)
        
        
        
        
    def evaluate(self, session0_best_model_dict):
        
        # criterion = SupConLoss(temperature=0.07)
        opt_embeddings = []
        opt_labels = []
                 
        args = self.args
        self.Q=torch.zeros(512,self.args.num_all)
        self.G=torch.zeros(512,512)
        for session in range(0, args.num_session):

            train_set, trainloader, valloader = self.get_dataloader(session)
            test_set, testloader = get_testloader(self.args, session)
            best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')

            if session == 0:  # load base class train img label
                self.model.load_state_dict(session0_best_model_dict)
                
                print('test classes for this session:\n', np.unique(test_set.targets))

            else:  # incremental learning sessions
                
                print("training session: [%d]" % session)
                previous_class = (args.num_base + (session-1) * args.way) 
                present_class = (args.num_base + session * args.way) 

                self.model.module.mode = self.args.network.new_mode
                self.model.eval()        
                
                self.model.train()
                for parameter in self.model.module.parameters():
                    parameter.requires_grad = False
                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                for parameter in self.model.module.fc.parameters():
                    parameter.requires_grad = True
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(name)

                optimizer = torch.optim.SGD(self.model.parameters(),lr=self.args.lr.lr_new, momentum=0.9, dampening=0.9 , weight_decay=0)
                support_data, support_label, cur_proto = self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                if self.args.train_proto:
                    print('Started fine tuning')
                    T = 2
                    beta = 0.25
                    alpha = 2
                    gamma = 0.7
                    
                    best_epoch = 0
                    best_loss = float('inf') 
                    
                    with torch.enable_grad():
                        for epoch in range(self.args.epochs.epochs_new):
                            inputs, label = support_data, support_label
                            
                            inputs = self.model.module.get_mel(inputs)
                            logits, feature,  _ = self.model(inputs)
                            
                            protos = args.proto_list
                            #print("*********************************protos.shape*********************^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^6",protos.shape)
                            
                            indexes = torch.randperm(protos.shape[0])
                            protos = protos[indexes]
                            temp_protos = protos.cuda()
                            #print("*********************************temp_protos.shape*********************^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^6",temp_protos.shape)
                            num_protos = temp_protos.shape[0] 
                            
                            label_proto = torch.arange(previous_class).cuda()
                            label_proto = label_proto[indexes]
                            
                            temp_protos = torch.cat((temp_protos,feature))
                            label_proto = torch.cat((label_proto,label))
                            logits_protos = self.model.module.fc(temp_protos) # True
                            ############################
                            embeddings = feature.unsqueeze(1)
                            loss_proto = nn.CrossEntropyLoss()(logits_protos[:num_protos,:present_class], label_proto[:num_protos]) * args.lamda_proto 
                            loss_ce = nn.CrossEntropyLoss()(logits_protos[num_protos:, :present_class], label_proto[num_protos:] ) * (1 - args.lamda_proto)
                            
                            optimizer.zero_grad()
                            
                            loss = loss_proto + loss_ce 
                            loss.backward()
                            optimizer.step()
                            
                            if loss < best_loss:
                                best_loss = loss
                                best_epoch = epoch
                                optimized_logits = self.model.module.fc(temp_protos) # resulting feature space after training with contrastive loss
                                
                                opt_embeddings.append(optimized_logits.detach().cpu())
                                opt_labels.append(label_proto.detach().cpu())
                                
                            print('Epoch: {}, Loss_CE: {}, Loss proto:{}, Loss: {}'.format(epoch, loss_ce, loss_proto, loss))
           
                    print("*****************best epoch for the new session is:==== epoch *********************",best_epoch)
                    
                    best = f"\n*****************best epoch for the new session is:==== epoch *********************{best_epoch}"
                    self.result_list.append({best})    
                else:
                    pass
               
            ################  Printing performance metrics ##################
            self.model.module.mode = self.args.network.new_mode
            self.model.eval()
            


            tsl, tsa, da, tsa_agg, acc_dict = test_agg(self.model, testloader, 0, args, session, print_numbers=True, save_pred=True)
            self.sess_acc_dict[f'sess {session}'] = acc_dict
            
            with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                if session == args.num_session - 1: 
                    result_file.write("\t".join([f"{acc:.4f}" for _, acc in da.items()]))

                    result_file.write("\n")
            
            print('Overall cumulative accuracy: {}, after agg: {}'.format(tsa*100, tsa_agg*100))
            self.result_list.append('Current Session {}, overall cumulative accuracy: {}, after agg: {}'.format(session, tsa*100, tsa_agg*100))            
            testset, novel_testloader = get_novel_testloader(self.args, session)
            tsl_novel, tsa_novel, da, tsa_agg_novel, acc_dict= test_agg(self.model, novel_testloader, 0, args, session)
            print('Novel classes cumulative accuracy: {}, after agg: {}'.format(tsa_novel*100, tsa_agg_novel*100))
            self.result_list.append('Current Session {}, novel classes cumulative accuracy: {}, after agg: {}'.format(session, tsa_novel*100, tsa_agg_novel*100))
            hm = 0
            hm_agg = 0
            hm_agg_stoc_agg = 0
            for j in range(0,session+1):
                testset, specific_testloader = get_task_specific_testloader(self.args, j)
                tsl, tsa, da, tsa_agg, acc_dict = test_agg(self.model, specific_testloader, 0, args, session)
                if session ==0:
                    tsa_base = tsa
                    tsa_agg_base = tsa_agg
                print('session: {} test accuracy: {}, after agg: {}'.format(j, tsa * 100, tsa_agg*100))
                self.result_list.append('session: {} test accuracy: {}, after agg: {}'.format(j, tsa * 100, tsa_agg*100))
                hm += 1/((tsa+0.0000000001)*100)
                hm_agg += 1/((tsa_agg+0.00000001)*100)
            if session>0:  
                print('Task wise Harmonic mean is : {}, agg: {}'.format((session+1)/hm, (session+1)/hm_agg))
                self.result_list.append('Task wise Harmonic mean is : {}, agg: {}'.format((session+1)/hm, (session+1)/hm_agg))
            ###################################################################
            
            ############ Update protos and save features #####################
            if session == 0:
                update_sigma_protos_feature_output(trainloader, train_set, self.model, args, session)
            else:
                update_sigma_novel_protos_feature_output(support_data, support_label, self.model, args, session)

            print('protos, radius', args.proto_list.shape, args.radius)
            self.result_list.append('protos {}, radius {}'.format(args.proto_list.shape, args.radius))
        
            self.model.module.mode = self.args.network.new_mode
            output, acc_df, final_df = self.pretty_output(save=False, print_output=False)
            save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)
                
       
            
        self.result_list.append(output)
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), self.result_list)
        ##################################################################                
        return final_df


    def target2onehot(self,targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
        return onehot
    

    def set_save_path(self):
        mode = self.args.network.base_mode + '-' + self.args.network.new_mode
        if self.args.strategy.data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        self.args.save_path = self.args.save_path + str(self.args.prefix) +'_'+  str(self.args.iter) + '_iter_' +str(self.args.train_proto) + '_train_proto_' + str(self.args.lamda) + '_lamda_'+str(self.args.peta) + '_peta_'+ str(self.args.alpha) + '_alpha_' \
                                + 'lamda_proto_' +str(self.args.lamda_proto) + 'way_'+str(self.args.way)+\
                                    'shot_' + str(self.args.shot) + '_%s/' % self.args.Method  
        if self.args.scheduler.schedule == 'Milestone':
            mile_stone = str(self.args.scheduler.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs.epochs_base, self.args.lr.lr_base, mile_stone, self.args.scheduler.gamma, self.args.batch_size_base,
                self.args.optimizer.momentum)
        elif self.args.scheduler.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs.epochs_base, self.args.lr.lr_base, self.args.scheduler.step, self.args.scheduler.gamma, self.args.batch_size_base,
                self.args.optimizer.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.network.temperature)

        if 'ft' in self.args.network.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr.lr_new, self.args.epochs.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('/', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
