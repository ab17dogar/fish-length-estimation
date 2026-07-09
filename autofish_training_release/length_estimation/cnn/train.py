import torch
from torchvision import transforms, datasets, models
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset, sampler
import torch.nn as nn
import os
import pandas as pd
from skimage import io
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import configparser
import argparse
import json
import time

from Model import Model
from FishLengthDataset import FishLengthDataset


def setup_dataloaders(config, num_workers=7):
    gt_path = config.get("Config", "MASKS_GT")
    pred_path = config.get("Config", "MASKS_PREDICTION", fallback=None)


    # Training param
    cache_images = config.getboolean("Training", "CACHE_IMAGES", fallback=False)
    
    # Pre-processing / formatting
    crop_to_bbox = config.getboolean("Preprocessing", "CROP_TO_BBOX", fallback=False)
    masking_type = config.get("Preprocessing", "MASKING_TYPE", fallback=None)
    
    # Input to the model
    model_input_bbox = config.getboolean("Model", "MODEL_INPUT_BBOX", fallback=False)
    model_input_plane = config.getboolean("Model", "MODEL_INPUT_PLANE", fallback=False)
    normalize_bbox = config.getboolean("Model", "NORMALIZE_BBOX", fallback=False)
    
    # Datasets from each folder
    data = {'train':FishLengthDataset(gt_path=gt_path,
                                      pred_path=pred_path,
                                      use_caching=cache_images,
                                      groups=json.loads(config.get("Config", "TRAIN_GROUPS")), 
                                      crop_to_bbox=crop_to_bbox,
                                      model_input_bbox=model_input_bbox,
                                      normalize_bbox=normalize_bbox,
                                      model_input_plane=model_input_plane,
                                      masking_type=masking_type,
                                      augment_color=config.getboolean("Augmentation", "AUGMENT_COLOR_TRAIN_IMAGES", fallback=False),
                                      augment_crop=config.getboolean("Augmentation", "AUGMENT_CROP_TRAIN_IMAGES", fallback=False)),
            'val':FishLengthDataset(gt_path=gt_path,
                                    pred_path=pred_path,
                                    use_caching=cache_images,
                                    groups=json.loads(config.get("Config", "VAL_GROUPS")), 
                                    crop_to_bbox=crop_to_bbox,
                                    model_input_bbox=model_input_bbox,
                                    normalize_bbox=normalize_bbox,
                                    model_input_plane=model_input_plane,
                                    masking_type=masking_type)}

    # Dataloader iterators
    batch_size = config.getint("Training", "BATCH_SIZE")
    dataloaders = {'train': DataLoader(data['train'], batch_size=batch_size,
                                       pin_memory=False,
                                       num_workers=num_workers, shuffle=True),
                   'val': DataLoader(data['val'], batch_size=batch_size,
                                     pin_memory=False,
                                     num_workers=num_workers, shuffle=False)}
    return dataloaders


def train(model,criterion,optimizer,train_loader,valid_loader,output_dir,
          max_epochs_stop=3,n_epochs=20,print_every=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            #if train_on_gpu:
            data, target = [d.cuda() for d in data], target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * batch_size

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

            
        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Prepare val image dir
            val_img_dir = os.path.join(output_dir, "val_images")
            if not os.path.exists(val_img_dir):
                os.makedirs(val_img_dir)

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for iv, (data, target) in enumerate(valid_loader):
                    # Tensors to gpu
                    #if train_on_gpu:
                    data, target = [d.cuda() for d in data], target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * batch_size

                    # Save val examples
                    # only for first 5 batches
                    if(iv < 5):
                        img = transforms.ToPILImage()(data[0][0])
                        plt.figure()
                        len_est = output[0].cpu().detach().numpy()[0]
                        len_gt = target[0].cpu().detach().numpy()[0]
                        plt.title(f"Length gt: {len_gt} vs est: {len_est}")
                        plt.imshow(img)
                        plt.savefig(os.path.join(val_img_dir, f"val-epoch{epoch}-batch{iv}.png"))
                        plt.close()


                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                history.append([train_loss, valid_loss])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )


                # Save model no matter what
                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

                # Save the model every 10th epoch
                if(epoch % 10 == 0):
                    torch.save(model.state_dict(), os.path.join(output_dir, f"model-epoch{epoch}.pt"))
                    
                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), os.path.join(output_dir, "model-lowest-val.pt"))
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt")))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss'
                            ])
                        return model, history

        # Save/plot train/val losses
        train_loss = [h[0] for h in history]
        val_loss = [h[1] for h in history]

        plt.figure()
        plt.plot(train_loss, label="train")
        plt.plot(val_loss, label="val")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss.png"))


        with open(os.path.join(output_dir, "train_loss.csv"), "w") as f:
            f.write("\n".join([str(s) for s in train_loss]))

        with open(os.path.join(output_dir, "val_loss.csv"), "w") as f:
            f.write("\n".join([str(s) for s in val_loss]))




    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f}')
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss'])
    return model, history



if __name__ == '__main__':
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    arguments = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(arguments.config)

    # Set random seed
    torch.manual_seed(config.getint("Training", "RANDOM_SEED", fallback=42))

    # Prepare model
    model = Model(bbox_input=config.getboolean("Model", "MODEL_INPUT_BBOX"),
                  plane_input=config.getboolean("Model", "MODEL_INPUT_PLANE"),
                  model_size=config.get("Model", "MODEL_SIZE", fallback=None),
                  freeze_backend=config.getboolean("Model", "FREEZE_BACKEND", fallback=True),
                  model_name=config.get("Model", "MODEL_BACKEND")).cuda() #get_pretrained_model()

    # Setup loss and optimizer
    loss_func = config.get("Training", "LOSS", fallback="l1")
    if(loss_func == "smooth-l1"):
        criterion = nn.SmoothL1Loss()
    elif(loss_func == "l1"):
        criterion = nn.L1Loss()
    elif(loss_func == "l2"):
        criterion = nn.MSELoss()
    print(f"loss: {criterion}")
    optimizer = optim.Adam(model.parameters())

    # Following weights will be trained
    print("following layers will be trained:")
    for p in optimizer.param_groups[0]['params']:
        if p.requires_grad:
            print(p.shape)


    # Setup output dir
    output_dir = config.get("Config", "OUTPUT_DIR")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.popen(f"cp {arguments.config} {output_dir}")
            
    # Setup dataloaders
    batch_size=config.getint("Training", "BATCH_SIZE")
    dataloaders = setup_dataloaders(config)

    # Test dataloaders   
    for split in ["train", "val"]:
    
        trainiter = iter(dataloaders[split])

        for i in range(10):
            data, target = next(trainiter)

            # Plot images for debugging
            img = transforms.ToPILImage()(data[0][0])
            plt.figure()
            plt.imshow(img)
            meta_data = [f"{d[0]}" for d in data[1:]]
            plt.title(f"input: {meta_data} {target[0]}", wrap=True)
            output_path = os.path.join(output_dir, "dataloaders", split,
                                       f"debug-{split}-dataloader{i}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()

    # Run training
    model, history = train(model,
                           criterion,
                           optimizer,
                           dataloaders['train'],
                           dataloaders['val'],
                           output_dir=output_dir,
                           max_epochs_stop=9000,
                           n_epochs=config.getint("Training", "NUM_EPOCHS"),
                           print_every=1)
