from .Network import MYNET
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from .losses import SupConLoss
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


def symmetric_kl_divergence(feat1, feat2):
    """
    计算两个特征分布之间的对称KL散度损失
    Args:
        feat1: 特征1 (bs, dim)
        feat2: 特征2 (bs, dim)
    Returns:
        loss: 对称KL散度标量值
    """
    # 将特征转换为概率分布（沿dim维度做softmax）
    p = F.softmax(feat1, dim=-1)
    q = F.softmax(feat2, dim=-1)
    
    # 计算KL(p||q)
    kl_pq = F.kl_div(
        input=q.log(),       # 注意kl_div的输入顺序
        target=p,            # target是参考分布
        reduction='batchmean', 
        log_target=False
    )
    
    # 计算KL(q||p)
    kl_qp = F.kl_div(
        input=p.log(),
        target=q,
        reduction='batchmean',
        log_target=False
    )
    
    # 对称KL散度
    loss = (kl_pq + kl_qp) / 2.0
    return loss

def RCNN(X_n, params,model):  # (5, 21, 3, 224, 224)
    # X_n = X_n.reshape(params.way,(params.shot+params.query),3,-1,128)
    
    # N, S, C, H, W = X_n.size()
    p = np.random.rand()
    K = [1, 3, 5, 7, 11, 15]
    if p > params.prob:
        k = K[np.random.randint(0, len(K))]
        Conv = nn.Conv2d(3, 3, kernel_size=k, stride=1, padding=k//2, bias=False)
        Conv = Conv.to('cuda:0')
        nn.init.xavier_normal_(Conv.weight)
        
        X_n = Conv(X_n)
        X_n = Conv(X_n)
    return X_n.detach()



def Max_phase(model, X_n,train_label, args):
    # criterion = SupConLoss(temperature=0.2)
    X_n = X_n.detach()
    X_n = X_n.cuda()
    # criterion = SupConLoss(temperature=0.07)
    
    opt = optim.SGD([X_n.requires_grad_()], lr=args.gen_lr)
    model.eval()
    init_features = None
    
    for i in range(args.gen_epochs):
        opt.zero_grad()
        logits, last_features,  feature_inter = model(X_n)
        # if i == 0:
        #     init_features = last_features.clone().detach()  # (105, 512)
        class_loss = F.cross_entropy(logits, train_label)
        adv_loss =  - class_loss
        adv_loss.backward()
        opt.step()
    return X_n.detach()

def Max_phase_our(model, X_n,train_label, args):
    # criterion = SupConLoss(temperature=0.2)
    X_n = X_n.detach()
    X_n = X_n.cuda()
    # criterion = SupConLoss(temperature=0.07)
    
    opt = optim.SGD([X_n.requires_grad_()], lr=args.gen_lr)
    model.eval()
    init_features = None
    
    for i in range(args.gen_epochs):
        opt.zero_grad()
        logits, last_features,  feature_inter = model(X_n)
        if i == 0:
            init_features = last_features.clone().detach()  # (105, 512)
        KL_loss = symmetric_kl_divergence(init_features, last_features)
        class_loss = F.cross_entropy(logits, train_label)
        adv_loss =  -class_loss -  KL_loss
        adv_loss.backward()
        opt.step()
    return X_n.detach()

def ata_base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    te = Averager()
    ta = Averager()
    treg = Averager()
    tcont_1 = Averager()
    tcont_2 = Averager()
    
    ttotal = Averager()
    
    all_embeddings = []
    all_labels = []
    
    
    criterion = SupConLoss(temperature=0.07)
    
    for i, (mels,train_label) in enumerate(trainloader):
        
        
        # mels = torch.cat([mels[0], mels[1]], dim=0)
        if torch.cuda.is_available():
            mels = mels.cuda(non_blocking=True)
            train_label = train_label.cuda(non_blocking=True)
        bsz = train_label.shape[0]
            
        # data, train_label = [_.cuda() for _ in batch]
        lamda=args.lamda
        peta= args.peta
        alpha=args.alpha
        
        mels = model.module.get_mel(mels)
        mels_hat = mels.clone().detach()
        mels_hat = RCNN(mels, args, model)
        mels_hat = Max_phase(model, mels_hat, train_label, args)
        mels_adv = mels_hat.clone().detach()
        model.train()
        optimizer.zero_grad()
        mels = torch.cat((mels,mels_adv))
        logits, embedding, feature_inter = model(mels) # True
        loss = F.cross_entropy(logits, train_label.tile(2))
        loss_ce = loss.item()
        # contrast_loss_tensor_1 = criterion(embedding[:bsz, :args.num_base], train_label)
        # contrast_loss_1 = contrast_loss_tensor_1.item()
        contrast_loss_1 = 0
        # contrast_loss_tensor_2 = criterion(embedding[bsz:, :args.num_base], train_label)
        # contrast_loss_2 = contrast_loss_tensor_2.item()
        contrast_loss_2 = 0
        acc = count_acc(logits, train_label.tile(2))
        
        KL_loss = symmetric_kl_divergence(embedding[:bsz,:], embedding[bsz:,:])
        
        total_loss = lamda * loss + alpha * KL_loss 
        # total_loss = lamda*loss  + peta *contrast_loss_1 + alpha * contrast_loss_2
        total_loss = loss 

        lrc = scheduler.get_last_lr()[0]
        
        tcont_1.add(contrast_loss_1)
        tcont_2.add(KL_loss.item())
        
        te.add(loss_ce)
        ttotal.add(total_loss.item())
        
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()
        optimizer.step()
        
        _, embedding, _ = model(mels) # resulting feature space after training with contrastive loss
        
        all_embeddings.append(embedding.detach().cpu())
        all_labels.append(train_label.detach().cpu())

        
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    tl = tl.item()
    ta = ta.item()
    tcont_1 = tcont_1.item()
    tcont_2 = tcont_2.item()
    
    ttotal = ttotal.item()
    te = te.item()
    
    return tl, ta, tcont_1, te, tcont_2, ttotal, all_embeddings, all_labels


def our_base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    te = Averager()
    ta = Averager()
    treg = Averager()
    tcont_1 = Averager()
    tcont_2 = Averager()
    
    ttotal = Averager()
    
    all_embeddings = []
    all_labels = []
    
    
    criterion = SupConLoss(temperature=0.07)
    
    for i, (mels,train_label) in enumerate(trainloader):
        
        
        # mels = torch.cat([mels[0], mels[1]], dim=0)
        if torch.cuda.is_available():
            mels = mels.cuda(non_blocking=True)
            train_label = train_label.cuda(non_blocking=True)
        bsz = train_label.shape[0]
            
        # data, train_label = [_.cuda() for _ in batch]
        lamda=args.lamda
        peta= args.peta
        alpha=args.alpha
        
        mels = model.module.get_mel(mels)
        mels_hat = mels.clone().detach()
        mels_hat = RCNN(mels, args, model)
        mels_hat = Max_phase(model, mels_hat, train_label, args)
        mels_adv = mels_hat.clone().detach()
        model.train()
        optimizer.zero_grad()
        mels = torch.cat((mels,mels_adv))
        logits, embedding, feature_inter = model(mels) # True
        labels = train_label.tile(2)
        loss = F.cross_entropy(logits,labels )
        # contrast_loss_tensor_1 = criterion(embedding[:bsz, :args.num_base], train_label)
        # contrast_loss_1 = contrast_loss_tensor_1.item()
        contrast_loss_1 = 0
        # contrast_loss_tensor_2 = criterion(embedding[bsz:, :args.num_base], train_label)
        # contrast_loss_2 = contrast_loss_tensor_2.item()
        contrast_loss_2 = 0
        # total_loss = lamda*loss  + peta *contrast_loss_1 + alpha * contrast_loss_2
        if epoch > args.iter: 
    
            # pit 
            bs,c,f,t = mels.shape
            samples_per_cls = (args.episode.episode_shot + args.episode.episode_query) *2
            pit_mels = mels.view(samples_per_cls, args.episode.episode_way, c,f,t)
            pit_label = labels.view(samples_per_cls, args.episode.episode_way)
            #
            base_data = pit_mels[:, :args.episode.base, :,:,:].reshape(-1, c,f,t)
            base_feat, _ = model.module.encode(base_data)
            base_lb = pit_label[:, :args.episode.base].reshape(-1)
        
            syn_new_data = pit_mels[:, args.episode.base:,:,:,:].view(samples_per_cls, 2,args.episode.syn_new,c,f,t)
            syn_new_label = pit_label[:, args.episode.base:].view(samples_per_cls, 2,args.episode.syn_new)
            lam = np.random.beta(args.pit_mixup_alpha, args.pit_mixup_alpha)
            mixed_data = lam * syn_new_data[:, 0, :, :,:, :] + (1 - lam) * syn_new_data[:, 1, :, :,:, :]
            mixed_data = mixed_data.reshape(-1, c,f,t)
            syn_new_label_ori = syn_new_label[:, 0, :].reshape(-1) # 
            syn_new_label_aux = syn_new_label[:, 1, :].reshape(-1)
            mixed_feat, _ = model.module.encode(mixed_data)
            syn_proto = mixed_feat.view(samples_per_cls, args.episode.syn_new, -1)[:args.episode.episode_shot, :, :].mean(0)
            # 
            picked_new_cls=torch.Tensor(np.random.choice(args.num_all-args.num_base,5,replace=False) + args.num_base).long()
            # start_cls = np.random.choice(args.num_all-args.num_base,1,replace=False) + args.num_base
            # start_cls = start_cls if start_cls < args.num_all - args.episode.syn_new else args.num_all - args.episode.syn_new - 1
            # picked_new_cls=torch.Tensor(np.arange(start_cls, start_cls + args.episode.syn_new)).long().cuda()
            novel_mask=torch.Tensor(np.zeros((args.num_all - args.num_base, model.module.num_features)))
            novel_mask[picked_new_cls - args.num_base, :] = syn_proto.cpu()
            model.module.fc.mu.data[args.num_base:, :] = novel_mask.cuda()
            base_logits = model.module.fc(base_feat)
            mixed_logits = model.module.fc(mixed_feat[args.episode.episode_shot *2 * args.episode.syn_new:, :])
            syn_lbs = torch.tile(picked_new_cls, (args.episode.episode_query * 2, )).cuda()
            # ce_loss = F.cross_entropy(base_logits, base_lb)
            # pit_loss = F.cross_entropy(mixed_logits, syn_lbs) 
            
            logits_ = torch.cat([base_logits, mixed_logits], dim=0)
            labels = torch.cat([base_lb, syn_lbs], dim=0)
            pit_loss = F.cross_entropy(logits_, labels) 
            acc = count_acc(logits_, labels)
        else:
            acc = count_acc(logits, labels)
            
            # loss = 0
        # lam = np.random.beta(args.pit_mixup_alpha, args.pit_mixup_alpha)
        
        # index = torch.randperm(bsz*2, device=mels.device) 

        # mixed_data = lam * mels + (1 - lam) * mels[index]
        
        # y_a, y_b = labels, labels[index]
        # mixup_loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
     
        KL_loss = symmetric_kl_divergence(embedding[:bsz,:], embedding[bsz:,:])
        
        total_loss =  loss 
        
        mixup_loss = 0
        lrc = scheduler.get_last_lr()[0]
        
        tcont_1.add(loss.item())
        tcont_2.add(loss.item())
        
        te.add(loss.item())
        ttotal.add(total_loss.item())
        
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()
        optimizer.step()
        
        _, embedding, _ = model(mels) # resulting feature space after training with contrastive loss
        
        all_embeddings.append(embedding.detach().cpu())
        all_labels.append(train_label.detach().cpu())

        
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    tl = tl.item()
    ta = ta.item()
    tcont_1 = tcont_1.item()
    tcont_2 = tcont_2.item()
    
    ttotal = ttotal.item()
    te = te.item()
    
    return tl, ta, tcont_1, te, tcont_2, ttotal, all_embeddings, all_labels

def ce_scl_base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    te = Averager()
    ta = Averager()
    treg = Averager()
    tcont_1 = Averager()
    tcont_2 = Averager()
    ttotal = Averager()
    all_embeddings = []
    all_labels = []   
    criterion = SupConLoss(temperature=0.07)
    
    for i, (mels,train_label) in enumerate(trainloader):
          
        if torch.cuda.is_available():
            mels = mels.cuda(non_blocking=True)
            train_label = train_label.cuda(non_blocking=True)
        bsz = train_label.shape[0]
            
        # data, train_label = [_.cuda() for _ in batch]
        lamda=args.lamda
        peta= args.peta
        alpha=args.alpha
        
        model.train()
        optimizer.zero_grad()
        mels = model.module.get_mel(mels)
        
        logits, embedding, feature_inter = model(mels) # True
        loss = F.cross_entropy(logits, train_label)
        loss_ce = loss.item()
        contrast_loss_tensor_1 = criterion(embedding[:bsz, :], train_label)
        contrast_loss_1 = contrast_loss_tensor_1.item()
        
        # contrast_loss_tensor_2 = criterion(embedding[bsz:, :args.num_base], train_label)
        # contrast_loss_2 = contrast_loss_tensor_2.item()
        contrast_loss_2 = 0
        acc = count_acc(logits, train_label)
        total_loss = lamda*loss  + peta *contrast_loss_tensor_1 

        lrc = scheduler.get_last_lr()[0]
        
        tcont_1.add(contrast_loss_1)
        tcont_2.add(contrast_loss_2)
        
        te.add(loss_ce)
        ttotal.add(total_loss.item())
        
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()
        optimizer.step()
        
        _, embedding, _ = model(mels) # resulting feature space after training with contrastive loss
        
        all_embeddings.append(embedding.detach().cpu())
        all_labels.append(train_label.detach().cpu())

        
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    tl = tl.item()
    ta = ta.item()
    tcont_1 = tcont_1.item()
    tcont_2 = tcont_2.item()
    
    ttotal = ttotal.item()
    te = te.item()
    
    return tl, ta, tcont_1, te, tcont_2, ttotal, all_embeddings, all_labels


def stand_base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    te = Averager()
    ta = Averager()
    treg = Averager()
    tcont_1 = Averager()
    tcont_2 = Averager()
    
    ttotal = Averager()
    
    all_embeddings = []
    all_labels = []
        
    for i, (mels,train_label) in enumerate(trainloader):
        if torch.cuda.is_available():
            mels = mels.cuda(non_blocking=True)
            train_label = train_label.cuda(non_blocking=True)
        mels = model.module.get_mel(mels)
        model.train()
        optimizer.zero_grad()
        logits, embedding,  feature_inter = model(mels) # True
        loss = F.cross_entropy(logits, train_label)
        loss_ce = loss.item()
        contrast_loss_1 = 0
        contrast_loss_2 = 0
        acc = count_acc(logits, train_label)
        total_loss = 0.2 * loss         
        tcont_1.add(contrast_loss_1)
        tcont_2.add(contrast_loss_2)
        te.add(loss_ce)
        ttotal.add(total_loss.item())
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()
        optimizer.step()
        _, embedding, _ = model(mels) # resulting feature space after training with contrastive loss
        all_embeddings.append(embedding.detach().cpu())
        all_labels.append(train_label.detach().cpu())
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    tl = tl.item()
    ta = ta.item()
    tcont_1 = tcont_1.item()
    tcont_2 = tcont_2.item()
    ttotal = ttotal.item()
    te = te.item()
    
    return tl, ta, tcont_1, te, tcont_2, ttotal, all_embeddings, all_labels

def base_train_pit(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    te = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(trainloader):
        data, train_label = [_.cuda() for _ in batch]
        # 为方便操作reshape
        
        samples_per_cls = args.episode.episode_shot + args.episode.episode_query
        data = data.view(samples_per_cls, args.episode.episode_way, -1)
        train_label = train_label.view(samples_per_cls, args.episode.episode_way)
        audio_samples = data.size(-1)
        #
        base_data = data[:, :args.episode.base, :].reshape(-1, audio_samples)
        base_data = model.module.get_mel(base_data)
        
        base_feat, _ = model.module.encode(base_data)
        base_lb = train_label[:, :args.episode.base].reshape(-1)
        # 将10个新类分成两组mixup合成新类
        
        syn_new_data = data[:, args.episode.base:, :].view(samples_per_cls, 2,args.episode.syn_new,-1)
        syn_new_label = train_label[:, args.episode.base:].view(samples_per_cls, 2,args.episode.syn_new,-1)
        lam = np.random.beta(args.pit_mixup_alpha, args.pit_mixup_alpha)
        mixed_data = lam * syn_new_data[:, 0, :, :] + (1 - lam) * syn_new_data[:, 1, :, :]
        mixed_data = mixed_data.reshape(-1, audio_samples)
        syn_new_label_ori = syn_new_label[:, 0, :].reshape(-1) # 
        syn_new_label_aux = syn_new_label[:, 1, :].reshape(-1)
        mixed_data = model.module.get_mel(mixed_data)
        mixed_feat, _ = model.module.encode(mixed_data)
        syn_proto = mixed_feat.view(samples_per_cls, args.episode.syn_new, -1)[:args.episode.episode_shot, :, :].mean(0)
        # 
        picked_new_cls=torch.Tensor(np.random.choice(args.num_all-args.num_base,5,replace=False) + args.num_base).long()
        # start_cls = np.random.choice(args.num_all-args.num_base,1,replace=False) + args.num_base
        # start_cls = start_cls if start_cls < args.num_all - args.episode.syn_new else args.num_all - args.episode.syn_new - 1
        # picked_new_cls=torch.Tensor(np.arange(start_cls, start_cls + args.episode.syn_new)).long().cuda()
        novel_mask=torch.Tensor(np.zeros((args.num_all - args.num_base, model.module.num_features)))
        novel_mask[picked_new_cls - args.num_base, :] = syn_proto.cpu()
        model.module.fc.mu.data[args.num_base:, :] = novel_mask.cuda()
        base_logits = model.module.fc(base_feat)
        mixed_logits = model.module.fc(mixed_feat[args.episode.episode_shot * args.episode.syn_new:, :])
        syn_lbs = torch.tile(picked_new_cls, (args.episode.episode_query, )).cuda()
        logits_ = torch.cat([base_logits, mixed_logits], dim=0)
        labels = torch.cat([base_lb, syn_lbs], dim=0)
        total_loss =  F.cross_entropy(logits_, labels) 
        acc = count_acc(logits_, labels)
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    #treg = treg.item()
    tcont_1 = 0
    tcont_2 = 0
    
    return tl, ta, tl, tcont_1, tcont_2

def base_train_mixup(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    te = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(trainloader):
        data, train_label = [_.cuda() for _ in batch]
        # 为方便操作reshape
        # one_hot_gt_label = F.one_hot(train_label, num_classes=args.num_all)
        lam = np.random.beta(args.pit_mixup_alpha, args.pit_mixup_alpha)
        batch_size = data.size(0)
        index = torch.randperm(batch_size, device=data.device) 

        mixed_data = lam * data + (1 - lam) * data[index]
        # mixed_gt_label = lam * one_hot_gt_label + (
        #     1 - lam) * one_hot_gt_label[index]
        y_a, y_b = train_label, train_label[index]
        # input = torch.cat((data,mixed_data))
        # labels = torch.cat((one_hot_gt_label,mixed_gt_label))
        # int_labels = torch.argmax(mixed_gt_label, dim=1)
        
        input = model.module.get_mel(mixed_data)
        logits, embedding,  feature_inter = model(input) # True
        total_loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
        total_loss = total_loss *0.2
        acc = count_acc(logits, train_label)
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    #treg = treg.item()
    tcont_1 = 0
    tcont_2 = 0
    
    return tl, ta, tl, tcont_1, tcont_2

def replace_base_fc(trainset, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
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

    for class_index in range(args.num_base):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.mu.data[:args.num_base] = proto_list

    return model


def update_sigma_protos_feature_output(trainloader, trainset, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    
    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            data = model.module.get_mel(data)
            
            _,embedding,  _ = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    if session == 0:
        
        for class_index in range(args.num_base):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        
        args.radius = np.sqrt(np.mean(radius)) 
        args.proto_list = torch.stack(proto_list, dim=0)
    else:
        for class_index in  np.unique(trainset.targets):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            feature_class_wise = embedding_this.numpy()
            cov = np.cov(feature_class_wise.T)
            radius.append(np.trace(cov)/64)
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
        args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)



def update_sigma_novel_protos_feature_output(support_data, support_label, model, args, session):
    
    model = model.eval()
    
    embedding_list = []
    label_list = []

    with torch.no_grad():
        data, label = support_data, support_label
        #model.module.mode = 'encoder'
        data = model.module.get_mel(data)
        
        _,embedding,  _= model(data)

        embedding_list.append(embedding.cpu())
        label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    radius = []
    assert session > 0
    for class_index in  support_label.cpu().unique():

        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        feature_class_wise = embedding_this.numpy()
        cov = np.cov(feature_class_wise.T)
        radius.append(np.trace(cov)/64)
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim =0)


def test_agg(model, testloader, epochs, args, session, print_numbers=False, save_pred=False): 
 
    test_class = args.num_base + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_agg = Averager()
    va_agg_stochastic_agg = Averager()
    num_stoch_samples = 10

    da = DAverageMeter()
    ca = DAverageMeter()
    pred_list = []
    label_list = []
    per_class_acc_list = []
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            data, test_label = [_.cuda() for _ in batch]
            data = model.module.get_mel(data)

            logits, features,  _ = model(data)
            logits = logits[:, :test_class]
            pred = torch.argmax(logits, dim=1)
            if session == args.num_session - 1:
                pred_list.append(pred)
                label_list.append(test_label)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            per_cls_acc, cls_sample_count = count_per_cls_acc(logits, test_label)
            da.update(per_cls_acc)
            ca.update(cls_sample_count)
            
        vl = vl.item()
        va = va.item()
        va_agg = va
        da = da.average()
        ca = ca.average()
        
        acc_dict = acc_utils(da, args.num_base, args.num_session, args.way, session)
        

    if print_numbers:
        print(acc_dict)
    return vl, va, da, va_agg, acc_dict