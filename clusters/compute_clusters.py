import os, torch
import torch.backends.cudnn as cudnn

import config
from cluster_utils import compute_clusters
from utils import generate_logger, print_and_log
from cca_utils import compute_cca, get_projection
from dataset_utils import get_dataloader, get_flat_features

SEED=12345

def main():

    args = config.get_args()

    # set-up file structures and create logger
    log, args.results_dir = generate_logger(args, args.results_dir)
    print_and_log(log, args)
    
    # basic/cuda set-up
    torch.manual_seed(SEED)
    if args.gpu>=0:
        assert args.gpu <= torch.cuda.device_count()
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(args.gpu)
        cudnn.enabled = True
        cudnn.benchmark = True
    else:
        print_and_log(log, 'on cpu. you should probably use gpus with --gpu=GPU_idx')
 
    ####################################### load datasets  #####################################
    print_and_log(log, '-' * 100)
    train_loader = get_dataloader('train', args)
    val_loader = get_dataloader('val', args, shared_dictionary=train_loader.dataset.dictionary)
    print_and_log(log, '-' * 100)
    
    ####################################### do CCA ##################################### 

    # flatten features for CCA
    if 'QA_full' in args.cca: # CCA on full dataset
        flat_train_features = get_flat_features(train_loader, args)
        if args.cca == 'QA_full_trainval':
            flat_val_features = get_flat_features(val_loader, args)
            flat_train_features = [ torch.cat((flat_train_features[i], flat_val_features[i]), dim=0) for i in range(len(flat_val_features)) ]
    elif 'QA_human' in args.cca: # CCA on H_t subset (i.e. subset with human relevance scores)
        flat_train_features = get_flat_features(train_loader, args, human_set_only=True)
        if args.cca == 'QA_human_trainval':
            flat_val_features = get_flat_features(val_loader, args, human_set_only=True)
            flat_train_features = [ torch.cat((flat_train_features[i], flat_val_features[i]), dim=0) for i in range(len(flat_val_features)) ]
   
    # do CCA
    lambdas, proj_mtxs = compute_cca(flat_train_features, k=args.k)
    
    # get train projections using learned weights
    train_projections = [get_projection(v, mtx, lambdas, args.p) for (v, mtx) in zip(flat_train_features, proj_mtxs) ]
    proj_train_mus = [ proj.mean(dim=0).view(-1,1) for proj in train_projections ]
    del flat_train_features, train_projections
   
    # compute clusters
    compute_clusters(train_loader, lambdas, proj_mtxs, proj_train_mus, args, log)
    compute_clusters(val_loader, lambdas, proj_mtxs, proj_train_mus, args, log)
            
if __name__ == '__main__':
    main()
