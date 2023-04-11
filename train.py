import sys, os
import torch

from utils import data, train, lr_scheduler
from utils.losses import UNetLoss
from models.model import NeoUnet
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Params():
    ### path ###
    train_fp = "data/train"
    root = "data/done"
    test_fp = "data/test"
    save_model_path = ""

    ### color ###
    mask_value = {
        "polyp": [255, 255, 255],
        "neoplastic": [255, 0, 0],
        "non-neoplastic": [0, 255, 0],
        "undefined": [255, 255, 0]
    }

    ### training size ###
    height = 352    # P.10(448、352、256)
    weight = 352

    ### training params ###
    lr = 1e-3
    min_lr = 0
    batch_size = 4
    num_epoch = 20
    warmup_epoch = 5
    weight_decay = 1e-4


def main(argv):
    ### Read data ###
    train_image_fps = data.read_data_file(Params.train_fp)
    train_dataset = data.Dataset(
        train_image_fps,
        mask_values = Params.mask_value,
        augmentation = data.get_train_augmentation(
            height = Params.height,
            width = Params.weight
        ),
        preprocessing = data.get_preprocessing(
            data.preprocess_input
        ),
    )

    ### Create model ###
    model = NeoUnet()
    loss = UNetLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size = Params.batch_size,
        shuffle = True,
        num_workers = 4
    )

    ### training ###
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = float(Params.lr),
        weight_decay = float(Params.weight_decay),
        momentum = 0.9
    )

    scheduler = lr_scheduler.CosineAnnealingWarmupLR(optimizer,
                                                     T_max = Params.num_epoch - Params.warmup_epoch,
                                                     warmup_epochs = Params.warmup_epoch,
                                                     eta_min = float(Params.min_lr))

    train_epoch = train.TrainEpoch(
        model,
        loss = loss,
        metrics = [],
        optimizer = optimizer,
        device = "cuda",
        verbose = True
    )

    for i in range(0, Params.num_epoch):
        current_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch: {} - Learning Rate {}".format(i, current_lr))

        train_logs = train_epoch.run(train_loader)

        print("Save model {}".format(Params.save_model_path))
        torch.save(model.state_dict(), Params.save_model_path)

        scheduler.step()


if __name__ == "__main__":
    main(sys.argv)

    
    