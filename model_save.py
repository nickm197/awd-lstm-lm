import torch


def model_save(fn, vocab=None, val_loss=None, config=None):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer, vocab, val_loss, config], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        # model, criterion, optimizer, vocab, val_loss, config = torch.load(f)
        return torch.load(f)