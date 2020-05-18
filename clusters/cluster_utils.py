import os
import json
import torch
import numpy as np

import utils
from dataset_utils import get_mask
from cca_utils import mean_center, get_projection, topk_corr
from sklearn.cluster import MeanShift, AgglomerativeClustering

def compute_clusters(loader, lambdas, proj_mtxs, proj_train_mus, args, log):
 
    torch.autograd.set_grad_enabled(False)
    utils.print_and_log(log, 'getting clusters on {:} set...'.format(loader.dataset.mode))

    # set up meters and buffers
    meters = utils.init_meters(args)
    clusters = []
    
    # move to gpu 
    proj_mtxs = utils.send_to_device(proj_mtxs, args.gpu)
    proj_train_mus = utils.send_to_device(proj_train_mus, args.gpu)
    lambdas = utils.send_to_device(lambdas, args.gpu)
    
    b1 = proj_mtxs[0]
    b2 = proj_mtxs[1]
        
    N = len(loader.dataset)
    D = 1 if args.eval_set == 'human' else args.D
    E = loader.dataset.dictionary.emb_size
    avg_fn = loader.dataset.get_avg_embedding
    for i, batch in enumerate(loader):

        mask = get_mask(batch['in_human_set'], args.eval_set == 'human')

        if isinstance(mask, torch.Tensor): 
            bsz = mask.sum(dim=1).gt(0).sum()
            batch = utils.send_to_device(batch, args.gpu)
        
            # compute avg embeddings for candidate answers
            emb_candidates = avg_fn(batch['answer_options_ids'][mask].view(-1, 100, args.S), batch['answer_options_length'][mask].view(-1, 100))
            emb_candidates = emb_candidates.view(bsz, D, 100, E) 
            # project candidate answers to joint space using b1
            proj_candidates = get_projection(emb_candidates.view(-1, E), b1, lambdas, args.p)
            proj_candidates = mean_center(proj_candidates, proj_train_mus[0]) # center by projected train answer mean
            proj_candidates = proj_candidates.view(bsz, D, 100, args.k) 
        
            # compute avg embedding for answer
            emb_answer = avg_fn(batch['answers_ids'][mask].view(bsz,-1,args.S), batch['answers_length'][mask].view(bsz,-1))
            # project answer to joint space using b1
            proj_answer = get_projection(emb_answer.view(-1, E), b1, lambdas, args.p)
            proj_answer = mean_center(proj_answer, proj_train_mus[0]) # center by projected train answer mean
            proj_answer = proj_answer.view(bsz, D, 1, args.k)

            corrs, _ = topk_corr(proj_candidates, proj_answer, k=100, dim=3)
            gtidxs = batch['gtidxs'][mask].view(bsz, -1)
            human_scores = batch['answer_options_scores'][mask].view(bsz,-1,100)

            meters, in_cluster_idxs = compute_cluster_stats(meters, corrs, gtidxs, cluster_method=args.cluster_method, \
                                                            human_scores=human_scores if args.eval_set == 'human' else None)
            clusters.extend( format_cluster(in_cluster_idxs, batch, mask, loader.dataset) )
            utils.log_iteration_stats(log, meters, i+1, len(loader)) 
        
    torch.autograd.set_grad_enabled(True)
    
    utils.print_and_log(log, '-' * 100) 
    clusters_path = save_clusters(clusters, args, loader.dataset.mode)
    utils.print_and_log(log, 'clusters saved to ' + clusters_path)
    utils.print_and_log(log, '-' * 100) 
    utils.save_meters(meters, args.results_dir)
    utils.print_and_log(log, utils.stringify_meters(meters))

def format_cluster(in_cluster_idxs, batch, mask, dataset):

    candidates = batch['answer_options']
    questions = batch['questions']
    answers = batch['answers']

    batch_of_clusters = []
    b_counter=0
    for b in range(mask.size(0)):
        d_counter=0
        for d in range(mask.size(1)):
            if mask[b][d]: 
                item = {'image_id' : int(batch['image_ids'][b])}
                item['round_id'] = d
                item['in_cluster_idxs'] = in_cluster_idxs[b_counter][d_counter].cpu().numpy().tolist()
                item['refs'] = [dataset.all_answers[c] for c in candidates[b][d][in_cluster_idxs[b_counter][d_counter]]]
                item['question'] = dataset.all_questions [ questions[b][d] ]
                item['gt_answer'] = dataset.all_answers[ answers[b][d] ]
            
                batch_of_clusters.append(item)
                d_counter+=1
                b_counter+=1

    return batch_of_clusters
 
def save_clusters(clusters, args, mode):
    clusters_path = os.path.join(args.results_dir, 'refs_{:}_{:}_{:}.json'.format(args.cluster_method, args.eval_set, mode))
    with open(clusters_path, 'w') as f:
        json.dump(clusters, f)
    return clusters_path
 
def compute_cluster_stats(meters, corrs, gt_idxs, cluster_method, human_scores):
   
    corrs_size = corrs.size()
    corrs = corrs.view(-1, corrs_size[-1])
    np_corrs = corrs.cpu().numpy()
    gt_idxs = gt_idxs.view(-1)

    to_list = lambda x: x.view(-1).cpu().numpy().tolist()
    to_np = lambda x: x.cpu().numpy()

    if cluster_method == 'H': # cluster = human_scores > 0 AND GT answer
        
        in_cluster_idxs = (human_scores > 0).view_as(corrs)
        in_cluster_idxs.scatter_(1, gt_idxs.unsqueeze(-1), 1)

    elif cluster_method == 'M': # cluster = meanshift
        
        one_hot = torch.zeros(corrs.size()).type_as(corrs).byte()
        one_hot.scatter_(1, gt_idxs.unsqueeze(-1), 1) 
        corrs_nogt = np.ma.MaskedArray(to_np(corrs), to_np(one_hot))
        
        clustering = [ MeanShift().fit(c.compressed().reshape(-1,1)) for c in corrs_nogt ]
        best_cluster_labels = [ np.argmax(c.cluster_centers_) for c in clustering ]
        in_cluster_idxs = [ c.labels_ == best_cluster_labels[i] for i, c in enumerate(clustering)]
        in_cluster_idxs = torch.Tensor([np.insert(c, gt_idxs[i].item(), 1) for i,c in enumerate(in_cluster_idxs)]).type_as(gt_idxs).byte()

    elif cluster_method == 'G': # cluster = agglomerative (n=5)
       
        n_clusters=5
        one_hot = torch.zeros(corrs.size()).type_as(corrs).byte()
        one_hot.scatter_(1, gt_idxs.unsqueeze(-1), 1) 
        corrs_nogt = np.ma.MaskedArray(to_np(corrs), to_np(one_hot))

        clustering = [ AgglomerativeClustering(n_clusters=n_clusters).fit(c.compressed().reshape(-1,1)) for c in corrs_nogt ]
        best_cluster_labels = []
        for i,c in enumerate(clustering):
            cluster_means = [ np.mean(corrs_nogt[i].compressed()[c.labels_ == n]) for n in range(n_clusters) ]
            best_cluster_labels.append( np.argmax(cluster_means) )
        in_cluster_idxs = [ c.labels_ == best_cluster_labels[i] for i, c in enumerate(clustering)]
        in_cluster_idxs = torch.Tensor([np.insert(c, gt_idxs[i].item(), 1) for i,c in enumerate(in_cluster_idxs)]).type_as(gt_idxs).byte()
         
    elif cluster_method == 'S': # cluster = sigma / range of std dev

        one_hot = torch.ones(corrs.size()).type_as(corrs).byte()
        one_hot.scatter_(1, gt_idxs.unsqueeze(-1), 0) 
        corrs_nogt = corrs[ one_hot ].view(corrs.size(0), 99)
         
        stds = torch.std(corrs_nogt, dim=1, keepdim=True)
        best_corrs = torch.max(corrs_nogt, dim=1, keepdim=True)[0]
        in_cluster_idxs = corrs_nogt >= (best_corrs - stds).expand_as(best_corrs)
        in_cluster_idxs = torch.Tensor([np.insert(c, gt_idxs[i].item(), 1) for i,c in enumerate(to_np(in_cluster_idxs)) ]).type_as(gt_idxs).byte()

    # find clusters of size 1, and exclude from std calculation
    cluster_sizes = torch.sum(in_cluster_idxs.float(), dim=1)
    valid_std_corrs = corrs[ cluster_sizes > 1 ]
    valid_std_idxs = in_cluster_idxs [ cluster_sizes > 1 ]

    # update meters
    if valid_std_corrs.nelement() > 0:
        meters['cluster_std'].update( [ torch.std( valid_std_corrs[i, idx] ).item() for i, idx in enumerate(valid_std_idxs) ] )
    meters['cluster_mean'].update( [ torch.mean( corrs[i, idx] ).item() if idx.any() else 0 for i, idx in enumerate(in_cluster_idxs) ] )
    meters['cluster_size'].update( to_list(cluster_sizes) )
    meters['gt_in_cluster'].update(  to_list(in_cluster_idxs.gather(1, gt_idxs.view(-1, 1)).float().mul_(100)) )

    # if eval_set == 'human', compute overlap stats with human scores
    if isinstance(human_scores, torch.Tensor):
        meters = compute_cluster_overlap_stats(meters, human_scores, in_cluster_idxs, gt_idxs)
    
    return meters, in_cluster_idxs.view(corrs_size)

def compute_cluster_overlap_stats(meters, human_scores, in_cluster_idxs, gt_idxs):
   
    # H = set of answer candidates with human scores unioned with ground-truth answer
    # C = generated candidate cluster

    H_mask = (human_scores > 0).view(-1, in_cluster_idxs.size(-1)) # -1 x 100
    H_mask.scatter_(1, gt_idxs.unsqueeze(-1), 1)
    H_size = H_mask.float().sum(dim=1) # | H | 
    C_size = in_cluster_idxs.float().sum(dim=1) # | C |
        
    # intersection of C and H
    intersection = H_mask.mul(in_cluster_idxs).float()
    intersection_size = intersection.sum(dim=1) # | H ∩ C |
    # union of C and H
    union = ((H_mask + in_cluster_idxs) > 0).float()
    union_size = union.sum(dim=1) # | H ∪ C |
    
    to_list = lambda x: x.mul_(100).cpu().numpy().tolist()
    if intersection_size.nelement() > 0:
        # | H ∩ C | / | H |
        meters['recall'].update( to_list(intersection_size.div( H_size )) )
        # | H ∩ C | / | C |
        meters['precision'].update( to_list(intersection_size.div( C_size )) )
        # | H ∩ C | / | H ∪ C |
        meters['IOU'].update( to_list(intersection_size.div(union_size)) )

    return meters
