import torch
import shutil
import os

def save_ckpt_single_state(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        dir_name = os.path.dirname(filename)
        basefilename = os.path.basename(filename)
        shutil.copyfile(filename, os.path.join(dir_name, 'best-{:s}'.format(basefilename)))


def save_ckpt(param_save_file, net, optim, other=None, is_best=False):
    if other is not None:
        epoch_state = {
            'state_dict': net.state_dict(),
            'optimizer': optim.state_dict(),
            'performance': other
        }
    else:
        epoch_state = {
            'state_dict': net.state_dict(),
            'optimizer': optim.state_dict(),
        }
    save_ckpt_single_state(epoch_state, param_save_file, is_best=is_best)


def save_ckpt_topN_state(state, performance, dir_name, TopNp, TopN=5):
    save_name_pattern = '{:04d}.pth.tar'

    if len(TopNp) == 0:
        TopNp.append(performance)
        N = len(TopNp)
        torch.save(state, os.path.join(dir_name, save_name_pattern.format(N)))
        return TopNp

    for idx, s_TopNp in enumerate(TopNp):
        if performance > s_TopNp:
            for s_idx in range(TopN, idx+1, -1):
                src = os.path.join(dir_name, save_name_pattern.format(s_idx-1))
                dst = os.path.join(dir_name, save_name_pattern.format(s_idx))
                if os.path.isfile(src):
                    shutil.move(src, dst)

            torch.save(state, os.path.join(dir_name, save_name_pattern.format(idx+1)))
            TopNp.insert(idx, performance)
            if len(TopNp)>TopN:
                TopNp = TopNp[0:TopN]
            return TopNp

    if len(TopNp) < TopN:
        TopNp.append(performance)
        N = len(TopNp)
        torch.save(state, os.path.join(dir_name, save_name_pattern.format(N)))
        return TopNp

    return TopNp

if __name__ == '__main__':
    # Test save_ckpt_topN_state:
    import numpy as np
    import py_utils.dir_utils as dir_utils
    np.random.seed(0)

    n_iter = 100
    x_len = 20
    TopN = 10

    for t in range(100):
        x = np.random.rand(10)
        TopNp = []
        dir_name = dir_utils.get_dir('Test')
        for x_i in x:
            epoch_state = {
                'performance': x_i
            }
            TopNp = save_ckpt_topN_state(epoch_state, x_i, dir_name, TopNp, TopN)

        # Check correctness:
        save_name_pattern = '{:04d}.pth.tar'
        saved_x = []
        for idx in range(TopN):
            ckpt = torch.load(os.path.join(dir_name, save_name_pattern.format(idx+1)), map_location=lambda storage, loc: storage)
            saved_x.append(ckpt['performance'])

        x_sorted = - np.sort(-x)
        Flag = True
        for i in range(TopN):
            if x_sorted[i] != saved_x[i]:
                Flag = False
        if Flag:
            print "[{:03d} | {:d}] Pass".format(t, 100)
        else:
            print "[{:03d} | {:d}] Fail".format(t, 100)


    print "DEBUG"
