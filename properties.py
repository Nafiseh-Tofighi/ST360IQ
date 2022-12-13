# config file
from src.config import Config

def config():
    config = Config({ 

        'wandb' : False,
        'wandb_name': 'ST360IQ',

        # device
        'gpu_id': "0",                          # specify GPU number to use
        'num_workers': 4,

        # ViT structure 
        'n_enc_seq': [16,16],                   # input feature map dimension (N = H*W) from backbone
        'n_layer': 14,                          # number of encoder layers
        'd_hidn': 384,                          # input channel of encoder (input: C x N)
        'i_pad': 0,
        'd_ff': 384,                            # feed forward hidden layer dimension
        'd_MLP_head': 1152,                     # hidden layer of final MLP
        'n_head': 6,                            # number of head (in multi-head attention)
        'd_head': 384,                          # channel of each head -> same as d_hidn
        'dropout': 0.1,                         # dropout ratio
        'emb_dropout': 0.1,                     # dropout ratio of input embedding
        'layer_norm_epsilon': 1e-12,
        'n_output': 1,                          # dimension of output
        'batch_size': 10,
        'patch_size': 32,

        # optimization & training parameters
        'n_epoch': 100,                         # total training epochs
        'learning_rate': 1e-4,                  # initial learning rate
        'weight_decay': 0,                      # L2 regularization weight
        'momentum': 0.9,                        # SGD momentum
        'T_max': 3e4,                           # period (iteration) of cosine learning rate decay
        'eta_min': 0,                           # minimum learning rate
        'save_freq': 10,                        # save checkpoint frequency (epoch)
        'val_freq': 5,                          # validation frequency (epoch)
        'train_size': 0.6,
        'val_size' : 0.2,  

        # tangent images
        'tan_data_path':'',                           
        'txt_file_tans': '',                    # list of images in the database(only names)

        # if specific train/test set available
        'split_avail' :True,
        'train_list': '', 
        'test_list': '', 

        # load & save checkpoint
        'snap_path': './weights',               # directory for saving checkpoint
        'checkpoint': None,                     # load checkpoint
        'if_test':True                          # True for run test set right after training
    })
    return config