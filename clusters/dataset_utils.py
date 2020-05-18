import sys
import utils
import torch
from datasets import VisualDialogDataset
import torchvision.transforms as transforms

def build_dataset(mode, args, shared_dictionary=None, with_options=True):
    
    normalize = transforms.Normalize(mean=[0.4711, 0.4475, 0.4080], std=[0.1223, 0.1221, 0.1450]) #visdial 
    transform = transforms.Compose([    transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])
    
    dataset = VisualDialogDataset(mode, args, with_options, transform)
    dataset.load_dictionary(shared_dictionary)
    dataset.load_data()

    return dataset

def get_dataloader(mode, args, shared_dictionary=None, with_options=True):

    loader = torch.utils.data.DataLoader(
                    build_dataset(mode, args, shared_dictionary, with_options),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=False)
    
    nelements = len(loader.dataset)
    
    return loader

def get_mask(human_set_mask, human_set_only=True):
    if human_set_only:
        if torch.sum(human_set_mask) == 0:
            return None
        else:
            return human_set_mask
    else:
        return torch.ones_like(human_set_mask)

def get_flat_features(loader, args, human_set_only=False):
  
    print('flattening {:} features...'.format(loader.dataset.mode))

    if human_set_only:
        return get_flat_human_features(loader, args)
    else:
        return get_flat_full_features(loader, args)

def get_flat_human_features(loader, args):
    
    avg_fn = loader.dataset.get_avg_embedding
    E = loader.dataset.dictionary.emb_size
    questions, answers = [], []
    for i, batch in enumerate(loader):

        sys.stdout.write('\r{}/{} --> {:3.1f}%'.format(str(i+1), str(len(loader)), (i+1)/float(len(loader))*100))
        sys.stdout.flush()

        mask = get_mask(batch['in_human_set'])

        if isinstance(mask, torch.Tensor):
            bsz = mask.sum()
            batch = utils.send_to_device(batch, args.gpu)
            human_scores = batch['answer_options_scores'][mask].view(bsz,-1,100)
            cluster_mask = (human_scores > 0)
            cluster_mask.scatter_(2, batch['gtidxs'][mask].view(bsz,-1, 1), 1)
            cluster_sizes = cluster_mask.sum(dim=2).view(bsz)

            emb_question = avg_fn(batch['questions_ids'][mask].view(bsz,-1,args.S), batch['questions_length'][mask].view(bsz,-1)).cpu()
            emb_answer_set = avg_fn(batch['answer_options_ids'][mask].view(-1,100,args.S), batch['answer_options_length'][mask].view(-1,100))
            emb_answer_set = emb_answer_set.view(bsz,-1,100,E)
            emb_cluster_set = emb_answer_set[cluster_mask].cpu()

            batch_idx, counter = 0, 1
            acc_cluster_sizes = torch.cumsum(cluster_sizes, dim=0)
            for emb_answer in emb_cluster_set:
                questions.append(emb_question[batch_idx])
                answers.append(emb_answer)
                if counter == acc_cluster_sizes[batch_idx]:
                    batch_idx += 1
                counter += 1
             
    sys.stdout.write("\n")
    questions = torch.stack(questions)
    answers = torch.stack(answers)
 
    return [ answers.view(-1, E), questions.view(-1, E)]

def get_flat_full_features(loader, args):

    avg_fn = loader.dataset.get_avg_embedding
    E = loader.dataset.dictionary.emb_size
    questions = torch.FloatTensor(loader.dataset.N, args.D, E)
    answers = torch.FloatTensor(loader.dataset.N, args.D, E)
    for i, batch in enumerate(loader):

        sys.stdout.write('\r{}/{} --> {:3.1f}%'.format(str(i+1), str(len(loader)), (i+1)/float(len(loader))*100))
        sys.stdout.flush()

        batch = utils.send_to_device(batch, args.gpu)
        bsz = batch['questions_ids'].size(0)
        questions[i*loader.batch_size:i*loader.batch_size+bsz] = avg_fn(batch['questions_ids'], batch['questions_length']).cpu()
        answers[i*loader.batch_size:i*loader.batch_size+bsz] = avg_fn(batch['answers_ids'], batch['answers_length']).cpu()
        
    sys.stdout.write("\n")
 
    return [ answers.view(-1, E), questions.view(-1, E)]
