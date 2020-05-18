import os
import sys
import torch
from datetime import datetime

def send_to_device(input_batch, device):

    if device>=0:
        if isinstance(input_batch, dict):
            for k, tensor in input_batch.items():
                if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                    input_batch[k] = tensor.to('cuda:' + str(device))
        elif isinstance(input_batch, list):
            for k, tensor in enumerate(input_batch):
                if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                    input_batch[k] = tensor.to('cuda:' + str(device))
        elif isinstance(input_batch, torch.Tensor):
            if not input_batch.is_cuda:
                input_batch = input_batch.to('cuda:' + str(device))
    else:
        if isinstance(input_batch, dict):
            for k, tensor in input_batch.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    input_batch[k] = tensor.to('cpu')
        elif isinstance(input_batch, list):
            for k, tensor in enumerate(input_batch):
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    input_batch[k] = tensor.to('cpu')
        elif isinstance(input_batch, torch.Tensor):
            if not input_batch.is_cuda:
                input_batch = input_batch.to('cpu')
    
    return input_batch

def stringify_meters(meters):
    s = ''
    for k,v in meters.items():
        s += '{name}: {meter:6.4f} (std: {std:6.4f})\t'.format(name=k, meter=v.avg, std=v.compute_std())

    return s + '\n'

def log_iteration_stats(log, meters, iteration, all_iterations):
    s = ''
    s += 'Batch [{:d}/{:d}]'.format(iteration, all_iterations)
    for k,v in meters.items():
        s += '\t{name}: {meter.val:6.4f} ({meter.avg:6.4f})'.format(name=k, meter=v)

    print_and_log(log, s)

def init_meters(args):

    meters = {}
    if args.eval_set == 'human':
        meters['precision'] = AverageMeter()
        meters['recall'] = AverageMeter()
        meters['IOU'] = AverageMeter()
    meters['gt_in_cluster'] = AverageMeter()
    meters['cluster_size'] = AverageMeter()
    meters['cluster_mean'] = AverageMeter()
    meters['cluster_std'] = AverageMeter()

    return meters

def save_meters(meters, save_path):

    for k,v in meters.items():
        filename = os.path.join(save_path + '_' + k + '.meter')
        if not os.path.exists(filename):
            torch.save(v, filename)

def generate_logger(opt, results_dir):

    results_dir = os.path.join(results_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) 
    os.makedirs(results_dir)
    logfile_path = os.path.join(results_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return logfile, results_dir

def print_and_log(log_file, message):
    print(message)
    log_file.write(str(message) + '\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if not isinstance(val, list):
            val = [val]
        self.values.extend(val)
        self.val = val[-1] 
        self.sum += sum(val)
        self.count += len(val)
        self.avg = self.sum / float(self.count)

    def compute_std(self):
        if self.count > 1:
            return torch.std( torch.Tensor(self.values) )
        else:
            return 0

