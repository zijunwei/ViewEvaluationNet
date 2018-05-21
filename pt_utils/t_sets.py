import torch
from torch.autograd import Variable


# def randImage(image_size=None):
#     if not image_size:
#         image_size = [1, 3, 224, 224]
#         return torch.FloatTensor(*image_size)
#     elif isinstance(image_size, torch.Size):
#         return torch.FloatTensor(image_size)
#     elif isinstance(image_size, (list, tuple)):
#         return torch.FloatTensor(*image_size)



# def randImageBatch(batch_size=1, image_size=None):
#     s_image = randImage(image_size)
#     image_batch = torch.unsqueeze(s_image, 0)
#     if batch_size > 1:
#         # TODO here
#         pass
#     return image_batch


def getOutputSize(input_size, module):
    # if first layer, it should be [1, 3, 224, 224]
    module.eval()
    input_image = torch.FloatTensor(torch.randn(*input_size))
    input_image = Variable(input_image)
    output_image = module(input_image)
    output_size = output_image.size()
    return output_size


def adjust_learning_rate(optimizer, i, init_lr=0.1, every_i=4):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (i // every_i))
    print "Lr at epoch {:d}\t{:.08f}".format(i, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_lr(optimizer, lr=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    # rand_image = randImage()
    # rand_image_batch = randImageBatch()

    print "DBUG"