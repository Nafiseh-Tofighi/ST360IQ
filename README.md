# ST360IQ
Pytorch implementation of "ST360IQ: NO-REFERENCE OMNIDIRECTIONAL IMAGE QUALITY ASSESSMENT WITH SPHERICAL VISION TRANSFORMERS"

# Extracting Tangent images

 First, from [Tangent_Images](https://github.com/Nafiseh-Tofighi/ST360IQ/tree/main/Tangent_Image) install requirements and edit `configuration.py`based on your dataset. Then run The output Tangent images will be saved in **target_path**. More details are available, [here](https://github.com/Nafiseh-Tofighi/ST360IQ/blob/main/Tangent_Image/README.md).
 
Examples of extracted Tangent images.

[<img src="https://github.com/Nafiseh-Tofighi/ST360IQ/blob/main/Images/oiqa_tan%2Bdis.png" width="500"/>](https://github.com/Nafiseh-Tofighi/ST360IQ/blob/main/Images/oiqa_tan%2Bdis.png)

## Train and Test

 - Tangent data of the dataset must be available.
 - Downlod the weights from this [website](https://download.pytorch.org/models/resnet50-0676ba61.pth). Rename the `.pth` file as "resnet50.pth" and put it in the `src` folder.
- Edit parameters in `properties.py` file and run `train.py` for start training.


**Parameters**
- `wandb`: bool.  
True if using wandb tool.
- `wandb_name`: str.  
Name of the wandb project.
- `gpu_id`: str.  
GPU number to use.
- `num_workers`: int.  
The number of available workers.
- `n_enc_seq`: [int, int].  
Input feature map dimension (N = H*W) from the backbone.
- `n_layer`: int.  
Number of encoder layers,
- `d_hidn`: int.  
Input channel of the encoder (input: C x N).
- `i_pad`: int.
- `d_ff`: int.  
Feed-forward hidden layer dimension.
- `d_MLP_head`: int.  
Hidden layer of the final MLP.
- `n_head`: int.  
The number of heads (in multi-head attention).
- `d_head`: int.  
Channel of each head -> same as d_hidn.
- `dropout`: float.  
Dropout ratio.
- `emb_dropout`: float.  
Dropout ratio of input embedding.
- `layer_norm_epsilon`: float.  
- `n_output`: int.  
Dimension of output.
- `batch_size`: int.
- `patch_size`: int.
- `n_epoch`: int.  
Total training epochs.
- `learning_rate`: float.  
Initial learning rate.
- `weight_decay`: float.  
L2 regularization weight.
- `momentum`: float.  
SGD momentum.
- `T_max`: float.  
Period (iteration) of cosine learning rate decay.
- `eta_min`: float.  
Minimum learning rate.
- `save_freq`: int.  
Save checkpoint frequency (epoch).
- `val_freq`: int.  
Validation frequency (epoch).
- `train_size`: float between 0 and 1.
- `val_size`: float between 0 and 1.
Field of view for extracting Tangent images.
- `tan_data_path`: str.  
The direction to the folder contains Tangent images data.
- `txt_file_tans`: str.  
The direction to the text file contains the list of `.pt` Tangent files(extracted at the Tangent image phase).
- `split_avail`: bool.  
True if specific train/test set available.
- `train_list`: str.  
The path to the text file contains the list of train set Tangent images data.
- `test_list`: str.  
The path to the text file contains the list of train set Tangent images data.
- `snap_path`: str.
Directory for saving checkpoint.
- `checkpoint`: str.  
Load checkpoint.
- `if_test`: bool.  
True for run test set right after training is over.
*If the last parameter is available, the test results for all checkpoints will be written in the test_results.txt file.*

## Model overview

The following figure is the overview of the implemented model. The saliency sampling-based image quality assessment model, with the help of the vision transformer, is designed. We first extract tangent viewports based on the salient region of the input image with ERP. The following step involves converting the input viewports into a series of tokens. Finally, a vision transformer encoder is included in addition to the related embeddings.

[<img src="https://github.com/Nafiseh-Tofighi/ST360IQ/blob/main/Images/Overview.png"/>](https://github.com/Nafiseh-Tofighi/ST360IQ/blob/main/Images/Overview.png)
