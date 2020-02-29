import torch


def model_save(fn, model, criterion, optimizer, vocab=None,
               val_loss=None, val_ppl=None, config=None, epoch=None):
    state = {'model': model, 'criterion': criterion,
             'optimizer': optimizer, 'vocab':vocab,
             'val_loss': val_loss, 'val_ppl': val_ppl,
             'config': config, 'epoch': epoch}
    with open(fn, 'wb') as f:
        torch.save(state, f)


def model_load(fn):
    # global model, criterion, optimizer
    with open(fn, 'rb') as f:
        # model, criterion, optimizer, vocab, val_loss, config = torch.load(f)
        return torch.load(f)
