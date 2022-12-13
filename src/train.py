import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import properties
from src.ViT_model import IQARegression
from src.resnet50 import resnet50_backbone
from trainer import train_epoch, eval_epoch
from util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle
from test import test_epoch

config=properties.config()

# WandB setting
if config.wandb:
    import wandb
    wandb.init(project=config.wandb_name)

# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')


from tan_dataset import IQADataset

# loading train/test scenes
train_scene, val_scene, test_scene = RandShuffle(config)

# data load
train_dataset = IQADataset(
    config=config,
    transform=transforms.Compose(
        [Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]), 
        RandHorizontalFlip(), 
        ToTensor()]),
    scene_list=train_scene
)

test_dataset = IQADataset(
    config=config,
    transform= transforms.Compose([Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]), 
        ToTensor()]
        ),
    scene_list=val_scene
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=config.batch_size, 
    num_workers=config.num_workers, 
    drop_last=True, 
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=config.batch_size, 
    num_workers=config.num_workers, 
    drop_last=True, 
    shuffle=True
)

# create model
model_backbone = resnet50_backbone().to(config.device)

# number of parameters
# print(sum(p.numel() for p in model_backbone.parameters() if p.requires_grad))

model_transformer = IQARegression(config).to(config.device)

# number of parameters
# print(sum(p.numel() for p in model_backbone.parameters() if p.requires_grad))
 
# loss function & optimization
criterion = torch.nn.L1Loss()
params = list(model_backbone.parameters()) + list(model_transformer.parameters())

optimizer = torch.optim.SGD(
    params, 
    lr=config.learning_rate, 
    weight_decay=config.weight_decay, 
    momentum=config.momentum
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=config.T_max, 
    eta_min=config.eta_min
)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
    model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

# make directory for saving weights
if not os.path.exists(config.snap_path):
    os.mkdir(config.snap_path)

# train & validation
for epoch in range(start_epoch, config.n_epoch):
    loss, srcc, plcc, rmse = train_epoch(
        config, 
        epoch, 
        model_transformer, 
        model_backbone, 
        criterion, 
        optimizer, 
        scheduler, 
        train_loader
    )

    if config.wandb:
        wandb.log(
            {'PLCC': plcc, 
            'RMSE': rmse,
            'RMSE': rmse, 
            'loss': loss, 
            'epoch': epoch+1}
        )

    if (epoch+1) % config.val_freq == 0:
        loss, srcc, plcc, rmse = eval_epoch(
            config, 
            epoch, 
            model_transformer, 
            model_backbone, 
            criterion, 
            test_loader
        )

        if config.wandb:
            wandb.log(
                {
                'validation-PLCC':plcc,
                'validation-SRCC': srcc, 
                'validation-RMSE': rmse, 
                'validation-loss': loss, 
                'epoch': epoch+1}
            )

if config.if_test:
    weights_source='./weights'
    result={}

    check_list = [
        os.path.join(weights_source, file) for file in os.listdir(weights_source) 
            if file.endswith('.pth')
        ]

    for check_idx, checkpoint in enumerate(check_list):
        result[f'checkpoint{(check_idx+1)*10}']=test_epoch(checkpoint, test_scene, config.tan_data_path)
    with open('./test_results.txt', 'w') as file:
        file.write(str(result))
