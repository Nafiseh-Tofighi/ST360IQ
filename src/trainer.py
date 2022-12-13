import os
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

""" train model """
def train_epoch(
    config, 
    epoch, 
    model_transformer, 
    model_backbone, 
    criterion, 
    optimizer, 
    scheduler, 
    train_loader
    ):

    losses = []
    model_transformer.train()
    model_backbone.train()

    # input mask (batch_size x len_sqe+1)
    mask_inputs = torch.ones(
        config.batch_size, 
        (config.n_enc_seq[0]*config.n_enc_seq[1])+1).to(config.device
        )

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for data in tqdm(train_loader):
        d_img = data['d_img'].to(config.device)
        source_emb=data['source_emb']
        center=data['center']
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        feat_d_img = model_backbone(d_img)

        # weight update
        optimizer.zero_grad()
        pred = model_transformer(
            mask_inputs, 
            feat_d_img,
            center,
            source_emb
            )         

        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    srcc, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    plcc, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rmse = mean_squared_error(np.squeeze(labels_epoch), np.squeeze(pred_epoch), squared=False)

    print('[train] epoch:%d / loss:%f / PLCC:%4f / SROCC:%4f / RMSE:%4f' % 
                                                    (epoch+1, loss.item(), srcc, plcc, rmse))

    # save weights
    if (epoch+1) % config.save_freq == 0:
        weights_file_name = "epoch%d.pth" % (epoch+1)
        weights_file = os.path.join(config.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_backbone_state_dict': model_backbone.state_dict(),
            'model_transformer_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))

    return np.mean(losses), srcc, plcc, rmse


""" validation """
def eval_epoch(
    config, 
    epoch, 
    model_transformer, 
    model_backbone, 
    criterion, 
    test_loader
    ):

    with torch.no_grad():
        losses = []
        model_transformer.eval()
        model_backbone.eval()

        mask_inputs = torch.ones(
            config.batch_size, 
            (config.n_enc_seq[0]*config.n_enc_seq[1])+1).to(config.device
            )

        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            d_img = data['d_img'].to(config.device)
            center=data['center']
            source_emb=data['source_emb']
            labels = data['score']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

            feat_d_img = model_backbone(d_img)

            pred = model_transformer(mask_inputs, feat_d_img,center, source_emb)

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            loss_val = loss.item()
            losses.append(loss_val)

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        srcc, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        plcc, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = mean_squared_error(np.squeeze(labels_epoch), np.squeeze(pred_epoch), squared=False)

        print('[validation] epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f / RMSE:%4f' % 
                                                            (epoch+1, loss.item(), srcc, plcc, rmse))

        return np.mean(losses), srcc, plcc, rmse
