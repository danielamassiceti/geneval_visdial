import os
import sys
import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Canonical Correlation Analysis for Visual Dialogue')
    # data
    parser.add_argument('--dataset_root', dest='dataset_root', required=True, help='Path to VisDial root folder.')
    parser.add_argument('--version', dest='version', default='1.0', help='VisDial version (default: 1.0).')
    parser.add_argument('--results_dir', dest='results_dir', default='./results', help='Results directory (default: ./results).')
    parser.add_argument('--fast_text_model', dest='fast_text_model', default='./models/fasttext.wiki.en.bin', \
                        help='Pre-trained word embedding model (default:./models/fasttext.wiki.en.bin).')
    parser.add_argument('--S', dest='S', default=16, type=int, help='Maximum sequence length (default: 16).')
    parser.add_argument('--D', dest='D', default=10, type=int, help='Dialogue exchanges per image (default: 10).')

    # CCA
    parser.add_argument('--cca', dest='cca', default='QA_full_train', choices=['QA_full_train', 'QA_full_trainval', 'QA_human_train', 'QA_human_trainval'], \
            help='Data to compute CCA (default: QA_human_trainval).')
    parser.add_argument('--p', type=float, default=1.0, help='Eigenvalue exponent for CCA (default: 1.0).')
    parser.add_argument('--k', type=int, default=300, help='Joint projection dimensionality for CCA (default: 300).')

    # experiment
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size (default: 128).')
    parser.add_argument('--workers', '-j', dest='workers', default=4, type=int, metavar='N', help='Number of workers to load data (default: 4).')
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int, help='GPU ID (default: -1 (cpu)).')
    
    # clustering
    parser.add_argument('--cluster_method', dest='cluster_method', default='S', type=str, choices = ['S', 'M', 'G', 'H'], help='Clustering method (default: S).')
    parser.add_argument('--eval_set', dest='eval_set', default='full', type=str, choices = ['full', 'human'], help='Train and val set to compute clusters for (default: full).')
    
    args = parser.parse_args()
    args.dataset_path = os.path.join(args.dataset_root, str(args.version))
    
    return args
