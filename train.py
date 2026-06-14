import argparse
import importlib
from utils.utils import *
import yaml
import logging
MODEL_DIR=None
DATA_DIR = '/data/datasets/The_NSynth_Dataset'

PROJECT='acl'

def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_proto', type=bool, default=True)
    parser.add_argument('-respre', type=bool, default=False)    
    parser.add_argument('-test', type=bool, default=False)    
    parser.add_argument('-num_groups', type=int, default=64)
    
    parser.add_argument('-seq_sample', type=bool, default=True)
    parser.add_argument('-s0_model_dir', type=str, default=None)
    
    parser.add_argument('-test_times', type=int, default=1)
    parser.add_argument('-shift_weight', type=float, default=0.1)
    parser.add_argument('-iter', type=int, default=0)

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='l2n',
                        choices=['nsynth-100', 'nsynth-200', 'nsynth-300', 'nsynth-400', 'librispeech', 'esc', 'FMC', 'l2n','n2l','l2n2f','l2f2n','l2n2f2n','l2n2f2e'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-save_path', type=str, default='')
    parser.add_argument('-config', type=str, default="") 
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-eta', type=float, default=0.9)
        
    parser.add_argument('-lamda_proto', type=float, default=0.6)
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=5)
    parser.add_argument('-num_session', type=int, default=10)
    
    # parser.add_argument('-num_session', type=int, default=7)
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-pretrain', type=bool, default=True)
    parser.add_argument('-calibration', type=bool, default=False)
    parser.add_argument('-ridge', type=bool, default=False)
    # parser.add_argument('--max_lr', default=0.1, type=float, help='max_lr')
    parser.add_argument('--gen_lr', default=0.001,type=float, help='max_lr')
    
    parser.add_argument('--gen_epochs', default=20, type=int, help='')
    
    parser.add_argument('--prob', default=0, type=float, help='probability of using original Images')
    parser.add_argument('--adv_train', type=bool, default=True)
    parser.add_argument('-a', type=float, default=0, help='')
    parser.add_argument('-b', type=float, default=2, help='')
        
    parser.add_argument('-lamda', type=float, default=0, help='celoss')
    parser.add_argument('-peta', type=float, default=0, help='proto_loss')
    # parser.add_argument('-gamma', type=float, default=1, help='supcon loss of generate data')
    parser.add_argument('-gama', type=float, default=0.1, help='dist')
    parser.add_argument('-lambda_saliency', type=float, default=1, help='dist')
    
    parser.add_argument('-sigma', type=float, default=10, help='domain loss of max')
    
    
    parser.add_argument('--epsilon', '-e', type=float, default=0.01, 
        help='maximum perturbation of adversaries')
    # parser.add_argument('--alpha', '-a', type=float, default=0.01, 
    #     help='movement multiplier per iteration when generating adversarial examples')
    parser.add_argument('--k', '-k', type=int, default=40, 
        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
        help='the type of the perturbation (linf or l2)')
    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
        help='what behavior want to do: train | valid | test | visualize')
    parser.add_argument('--n_eval_step', type=int, default=50, 
        help='number of iteration per one evaluation')
    # about training
    parser.add_argument('-gpu', default='2')
    # print(parser.parse_args())
    args, unknown = parser.parse_known_args()
    args = parser.parse_args()
    with open(args.config, 'r') as config:
        cfg = yaml.safe_load(config) 
    cfg = cfg['train']
    cfg.update(vars(args))
    # args = argparse.Namespace(**cfg)
    args = dict2namespace(cfg)
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(100)]) + "\n")
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(100)]) + "\n")
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(50)]) + "\n")
    with open("per_cls_"+args.dataset+".txt", "a") as result_file:
                
                    result_file.write("Class\tAccuracy\n")
                    result_file.write("\t".join([str(cls_idx) for cls_idx in range(10)]) + "\n")


    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()