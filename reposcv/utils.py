import multiprocessing as mp
import os 
from tqdm import tqdm

def get_progress(iterable=None, total=None, desc=None):
    """
    get progress bar
    :param iterable: target to be iterated
    :param total: total length of the progress bar
    :param desc: description of the progress bar
    :return: progress bar
    """
    return tqdm(iterable=iterable, total=total, desc=desc)
    
"""
Ex: calculate_dice(idx, df, axis, return_df=False)
mutiprocess(calculate_dice, range(len(pairs)), df=pairs, return_df=True)
"""
def multiprocess(func, iter_args, workers=4, **kwds):
    """"
    parallel iterate array
    # .apply_async(func[, args[, kwds[, callback[, error_callback]]]])
    func: function to be called for each data, signature (idx, arg) or arg
    args:  array to be iterated
    kwds: keywords argument
    """
    print('...multiprocessing...')
    pool = mp.Pool(processes=workers)
    jobs = [pool.apply_async(func, (arg,), kwds) for arg in iter_args]
    
    """
    .get(): take value of specified key
    """
    results = [j.get() for j in tqdm(jobs)]
    pool.close()
    pool.join()     # wait the process end
    
    return results


"""
set gpu_ids for training, ids start from 0
"""
def set_gpu(*gpu_ids):
 
    assert len(gpu_ids) > 0, 'require at least 1 is=d'
    
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join([str(id) for id in gpu_ids])  

    # print(','.join([str(id) for id in gpu_ids]))





