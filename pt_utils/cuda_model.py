import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

def ifUseCuda(gpu_id, multiGPU):
    return torch.cuda.is_available() and (gpu_id is not None or multiGPU)


def convertModel2Cuda(model, gpu_id, multiGpu):
    use_cuda = torch.cuda.is_available() and (gpu_id is not None or multiGpu)
    if use_cuda:
        if multiGpu:
            if gpu_id is None:  # using all the GPUs
                device_count = torch.cuda.device_count()
                print("Using ALL {:d} GPUs".format(device_count))
                model = nn.DataParallel(model, device_ids=[i for i in range(device_count)]).cuda()
            else:
                print("Using GPUs: {:s}".format(gpu_id))
                device_ids = [int(x) for x in gpu_id]
                model = nn.DataParallel(model, device_ids=device_ids).cuda()


        else:
            torch.cuda.set_device(int(gpu_id))
            model.cuda()

        cudnn.benchmark = True

    return model


def convertModel2Cuda_args(model, args):
    return convertModel2Cuda(model, args.gpu_id, args.multiGpu)