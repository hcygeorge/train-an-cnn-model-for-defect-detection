import os
import time
import torch
import torch.nn as nn
import logging

#%% Log setting
class Log():
    """Create logger object to output log to file.
    
    Args:
        filename (str): The filname that stores logs.

    Attributes:
        filename (str): The filname that stores logs.
    """
    def __init__(self, filename):
        self.filename = filename
        
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
            print("Create a folder 'logs' under working directory.")
        
    def log(self, message):
        """Output log to file.
        
        Args:
            message (str): The contents of log.
        """
        logger = logging.getLogger(__name__)  # must be __name__, or duplicate log when pytorch workers>0
        # If logger is already configured, remove all handlers
        if logger.hasHandlers():
            logger.handlers = []
        logger.setLevel(logging.INFO)
        # Setting log format
        handler = logging.FileHandler('./logs/{}.log'.format(self.filename))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s', '%m-%d %H:%M')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info(message)

# logging.basicConfig(level=logging.INFO,
#         format='%(asctime)s %(message)s',
#         datefmt='%m-%d %H:%M:%S',
#         filename='./logs/{}.log'.format(trial_info))
#%%
# Sub subgradient descent for L1-norm
def updateBN(model, scale=1e-4, verbose=False, fisrt=1e-4, last=1e-4):
    """Update subgradient descent for L1-norm.
    
    Args:
        model (nn.modules): A Pytorch model.
        scale (float): scaling factor of L1 penalty term.
    """
    for idx, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # if idx == 'features.28':
            #     m.weight.grad.data.add_(fisrt*torch.sign(m.weight.data))
            # if idx == 'features.35':
            #     m.weight.grad.data.add_(fisrt*torch.sign(m.weight.data))
            if idx == 'features.41':
                m.weight.grad.data.add_(last*torch.sign(m.weight.data))
            else:
                m.weight.grad.data.add_(scale*torch.sign(m.weight.data))  # L1

#%%
def savemodel(state, is_best, freq=10, suffix='', verbose=False):
    serial_number = time.strftime("%m%d")
    checkpoint = './model/checkpoint{}_{:s}.pkl'.format(serial_number, suffix)
    bestmodel = './model/bestmodel{}_{:s}.pkl'.format(serial_number, suffix)
        
    if not os.path.exists('./model'):
        os.makedirs('./model')
        print("Create a folder 'model' under working directory.")
        
    if verbose:
        print('Filepaths: {:s}/{:s}'.format(bestmodel, checkpoint))
        
    if is_best:
        torch.save(state, bestmodel)
        return None
    elif (state['epoch'] + 1) % freq == 0:
        torch.save(state, checkpoint)
        return 'Model saved.'
        # print('Model saved.')
        
        
