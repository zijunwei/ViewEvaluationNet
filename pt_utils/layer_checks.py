import torch.nn as nn
import numpy as np
from collections import OrderedDict

def checkIfHasNan(np_input):
    if (np.isnan(np_input)).any():
        return True
    else:
        return False


def checkIfHasInf(np_input):
    if (np.isinf(np_input)).any():
        return True
    else:
        return False


def checkIfAllZero(np_input):
    if (np_input == 0).all():
        return True
    else:
        return False


def checkBNValid(net, ifPrint=True, useCuda=False):
    isValid= True
    for id, s_module in enumerate(net.modules()):
        if isinstance(s_module, nn.BatchNorm2d):
            if useCuda:
                running_mean_np = s_module.running_mean.cpu().numpy()
                running_var_np = s_module.running_var.cpu().numpy()
            else:
                running_mean_np = s_module.running_mean.numpy()
                running_var_np = s_module.running_var.numpy()
            if checkIfHasNan(running_mean_np):
                if ifPrint:
                    print "BN # {:d}  running_MEAN has NaN".format(id)
                isValid = False

            if checkIfHasInf(running_mean_np):
                if ifPrint:
                    print "BN # {:d}  running_MEAN has Inf".format(id)
                isValid = False

            if checkIfHasNan(running_var_np):
                if ifPrint:
                    print "BN # {:d}  running_Var has NaN".format(id)
                isValid = False
            if checkIfHasInf(running_var_np):
                if ifPrint:
                    print "BN # {:d}  running_Var has Inf".format(id)
                isValid = False

    return isValid


def getALLBNs(net):
    BNs = OrderedDict()
    for id, s_module in enumerate(net.modules()):
        if isinstance(s_module, nn.BatchNorm2d):
            BNs[id] = s_module
    return  BNs


