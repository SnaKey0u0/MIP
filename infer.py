import sys, os
import torch
import cv2
import time
import numpy as np
from tqdm import tqdm

from utils import data
from utils.losses import UNetLoss
from models.model import BlazeNeo
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Params():
    ### path ###
    train_fp = "/home/zzy/Desktop/111-2/MIS/project/data/train/"
    test_fp = "/home/zzy/Desktop/111-2/MIS/project/data/test"
    save_model_path = ""
    model_path = ""
    save_result_path = ""

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
    test_image_fps = data.read_data_file(Params.train_fp)
    test_dataset = data.Dataset(
        test_image_fps,
        mask_values = Params.mask_value,
        augmentation = data.get_valid_augmentation(
            height = Params.height,
            width = Params.weight
        ),
        preprocessing = data.get_preprocessing(
            data.preprocess_input
        ),
    )

    ### Create model ###
    model = BlazeNeo()
    model.load_state_dict(torch.load(Params.model_path), strict=False)
    model.to(torch.device("cuda"))
    model.eval()

    ### infer ###
    for i in tqdm(range(0, len(test_image_fps))):
        fn = test_image_fps[i].split("/")[-1].split(".")[0]
        image, label = test_dataset[i]
        image = torch.from_numpy(image).to("cuda").unsqueeze(0)

        # UNet, HarDNetMSEG
        with torch.no_grad():
            predict = model(image)
        # BlazeNeo
        # with torch.no_grad():
        #     _, predict = model(image)
        # PraNet
        # with torch.no_grad():
        #     _, _, _, predict = model(image)
        predict = torch.argmax(
            predict, dim=1, keepdims=True).squeeze().data.cpu().numpy()
        neo_predict = (predict == 0).astype(np.float)
        non_predict = (predict == 1).astype(np.float)
        output = np.zeros(
            (predict.shape[0], predict.shape[1], 3)).astype(np.uint8)
        output[neo_predict > 0] = [0, 0, 255]
        output[non_predict > 0] = [0, 255, 0]

        # NeoUNet
        # with torch.no_grad():
        #     _, _, _, predict = model(image)
        # neo_predict = predict[:, [0], :, :]
        # non_predict = predict[:, [1], :, :]

        # neo_predict = torch.sigmoid(neo_predict).squeeze().data.cpu().numpy()
        # non_predict = torch.sigmoid(non_predict).squeeze().data.cpu().numpy()

        output = np.zeros(
            (predict.shape[-2], predict.shape[-1], 3)).astype(np.uint8)
        output[(neo_predict > non_predict) * (neo_predict > 0.5)] = [0, 0, 255]
        output[(non_predict > neo_predict) * (non_predict > 0.5)] = [0, 255, 0]

        saved_path = os.path.join(Params.save_result_path, '{}.png'.format(fn))
        cv2.imwrite(saved_path, output)


if __name__ == "__main__":
    main(sys.argv)

    
    