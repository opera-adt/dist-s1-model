import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from src.dist_model import SpatioTemporalTransformer
from src.utils import nll_gaussian, nll_gaussian_stable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


########################################################

# If you are using the dev version
# os.environ['WITH_CUDA'] = 'true'
# os.environ['DEV_DTYPE'] = 'float32'

max_num_pre_imgs = 10
patch_size = 8
input_size_tf = 16
num_patches = int((input_size_tf / patch_size) ** 2)


train_config = {
    'batch_size': 1,
    'num_epochs': 50,  # ideally 100+
    'learning_rate': 1e-4,
    'seed': 177,
    # StepLR
    'step_size': 25,
    'gamma': 0.1,
}

tf_config = {
    'type': 'transformer',
    'patch_size': patch_size,
    'num_patches': num_patches,
    'data_dim': int(2 * patch_size * patch_size),
    'd_model': 256,  # embedding dimension
    'nhead': 4,
    'num_encoder_layers': 4,
    'dim_feedforward': 768,
    'max_seq_len': max_num_pre_imgs,
    'dropout': 0.2,  # can be between .1 - .3
    'activation': 'relu',
}


TRAIN_PATH = 'training_data/original/train_12813.pt'
TEST_PATH = 'training_data/original/test_3204.pt'

########################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_epoch_tf(dataloader, model, optimizer, device, train=True):
    """Perform one epoch of training by looping through the dataset once."""
    """
    dataloader should have:
    pre_imgs: (batch_size, max_seq_len, channels, height, width)
    target: (batch_size, channels, height, width)
    """

    # Setting models and datasets into train/test mode
    if train:
        model.train()
    else:
        model.eval()

    nll_total = 0
    mse_total = 0

    naive_nll = 0
    naive_mse = 0

    for _, (batch, target) in enumerate(dataloader):
        input_size = 16

        batch = batch.unfold(3, input_size, input_size).unfold(4, input_size, input_size)
        target = target.unfold(2, input_size, input_size).unfold(3, input_size, input_size)

        batch = batch.permute(0, 3, 4, 1, 2, 5, 6).reshape(-1, 10, 2, input_size, input_size)
        target = target.permute(0, 2, 3, 1, 4, 5).reshape(-1, 2, input_size, input_size)

        target = torch.special.logit(target)  # make sure you're doing this once and not 0 or 2 times...
        batch = torch.special.logit(batch)

        ###

        mask_out_idx = torch.randint(0, 9, (1,))  # vary the time window the model sees
        target = target.to(device)

        batch = batch[:, mask_out_idx:, ...]  # "batch" is the pre images
        batch = batch.to(device)

        if train:
            pred_means, pred_logvars = model(batch)

            # new style
            loss = nll_gaussian(pred_means, pred_logvars, target)  # , pi=pi)
            mse_total += F.mse_loss(pred_means, target).detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # likely not necessary for transformer
            optimizer.step()

        else:
            with torch.no_grad():
                # compute baseline based on pre image mean and var
                pre_image_mean = torch.mean(batch, dim=1)
                pre_image_var = torch.var(batch, dim=1)
                pre_image_var += 1e-8  # for numerical stability

                naive_nll += nll_gaussian_stable(pre_image_mean, pre_image_var, target)
                naive_mse += F.mse_loss(pre_image_mean, target).item()

                # get prediction
                pred_means, pred_logvars = model(batch)

                loss = nll_gaussian(pred_means, pred_logvars, target)  # , pi=pi)
                mse_total += F.mse_loss(pred_means, target).item()

        nll_total += loss.item()

    nll_average = nll_total / len(dataloader)  # average NLL per sequence (sample)
    mse_average = mse_total / len(dataloader)

    naive_nll_average = naive_nll / len(dataloader)
    naive_mse_average = naive_mse / len(dataloader)

    if epoch % 10 == 0:
        print('Train \n' if train else 'Test \n')
        print(f'NLL: {nll_average:.6f}')
        if not train:
            print(f'Naive NLL: {naive_nll_average:.6f}\n')

        print(f'MSE: {mse_average:.6f}')
        if not train:
            print(f'Naive MSE: {naive_mse_average:.6f}')
        print(' ')

    return nll_average, mse_average  # had nll average first


pi = torch.FloatTensor([np.pi]).to(device)
train_dataset = torch.load(TRAIN_PATH, weights_only=False)
test_dataset = torch.load(TEST_PATH, weights_only=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = SpatioTemporalTransformer(tf_config)
print('Number of parameters: ', model.num_parameters())

now = datetime.now()
now = now.strftime('%m-%d-%Y %H:%M')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = SpatioTemporalTransformer(tf_config).to(device)
print('Number of parameters: ', model.num_parameters())
model_type = 'tf'


# Initialize optimizer (default using ADAM optimizer)
optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

scheduler = StepLR(optimizer, step_size=train_config['step_size'], gamma=train_config['gamma'])

training_notes = {
    'trainsetsize': len(train_dataset),
    'bursts': 'all',
    'total_params': model.num_parameters(),
    'step size': train_config['step_size'],
    'gamma': train_config['gamma'],
    'other': 'summer model, aurora data',
}

train_loss_vec = []
train_mse_vec = []
test_loss_vec = []
test_mse_vec = []


for epoch in range(1, train_config['num_epochs'] + 1):
    # if epoch % 10 == 0:
    print('--- EPOCH [{}/{}] --- \n'.format(epoch, train_config['num_epochs']))

    epoch_start_time = time.time()

    train_loss, train_mse_loss = run_epoch_tf(train_loader, model, optimizer, device, train=True)
    test_loss, test_mse_loss = run_epoch_tf(test_loader, model, optimizer, device, train=False)

    print('train loss: ', train_loss)
    print('test loss: ', test_loss)
    print('train MSE: ', train_mse_loss)
    print('test MSE: ', test_mse_loss)
    print('time: ', (time.time() - epoch_start_time) / 60, ' minutes')
    print('\n')

    train_loss_vec.append(train_loss)
    train_mse_vec.append(train_mse_loss)
    test_loss_vec.append(test_loss)
    test_mse_vec.append(test_mse_loss)

    if epoch % 5 == 0 and epoch != 0:
        # Save model

        folder_path = 'model_weights'
        file_name = f'{tf_config["type"]}_{now}.pth'
        full_path = f'{folder_path}/{file_name}'

        torch.save(model.state_dict(), full_path)

        # Plot Negative Log Likelihood Loss
        plt.figure()
        plt.plot(np.arange(1, len(train_loss_vec) + 1), train_loss_vec)
        plt.plot(np.arange(1, len(test_loss_vec) + 1), test_loss_vec)
        plt.legend(['Train', 'Test'])
        plt.title('Negative Log Likelihood Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'figs/nll_loss_curve_{now}.png', dpi=200, bbox_inches='tight', pad_inches=0)

        # Start a new figure before plotting MSE
        plt.figure()
        plt.plot(np.arange(1, len(train_mse_vec) + 1), train_mse_vec)
        plt.plot(np.arange(1, len(test_mse_vec) + 1), test_mse_vec)
        plt.legend(['Train', 'Test'])
        plt.title('Mean Squared Error Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'figs/mse_loss_curve_{now}.png', dpi=200, bbox_inches='tight', pad_inches=0)

    scheduler.step()
