import os
import time
import numpy as np
import time
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import source
import segmentation_models_pytorch as smp
import argparse
import warnings
from source.mit_unet.network_mit_unet import Net
warnings.filterwarnings("ignore")

class NamedDataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, name=""):
        super(NamedDataParallel, self).__init__(module, device_ids, output_device, dim)
        self.name = name

def data_loader(args):
    # get all image paths with ".tif" and "/labels/" in the path
    train_pths = [f for f in Path(args.train_data_root).rglob("*.tif")]
    # val_pths = [f for f in Path(args.val_data_root).rglob("*.tif")]
    # Shuffle the paths to randomize the selection
    # random.shuffle(img_pths)
    # # split data: 90% training and 10% validation
    # split_idx= int(0.9 * len(img_pths))
    # convert paths to strings (if needed)
    train_pths = [str(f) for f in train_pths]
    # val_pths = [str(f) for f in val_pths]
    
    # print("Total samples      :", len(img_pths))
    print("Training samples   :", len(train_pths))
    # print("Validation samples :", len(val_pths))

    trainset = source.dataset.Dataset(train_pths, classes=args.classes, size=args.crop_size, train=True)
    # validset = source.dataset.Dataset(val_pths, classes=args.classes, train=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # return train_loader, valid_loader
    return train_loader

def train_model(args, model, optimizer, criterion, metric, device):
    # get dataset loaders
    train_data_loader = data_loader(args)
    
    # create folder to save model
    os.makedirs(args.save_model, exist_ok=True)
    model_name = f"SAR_Pesudo_{model.name}_s{args.seed}_{criterion.name}"

    # max_score = 0
    train_hist = []
    # valid_hist = []
    for epoch in range(args.n_epochs):
        print(f"\nEpoch: {epoch + 1}")

        logs_train = source.runner.train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            dataloader=train_data_loader,
            device=device,
        )

        # logs_valid = source.runner.valid_epoch(
        #     model=model,
        #     criterion=criterion,
        #     metric=metric,
        #     dataloader=val_data_loader,
        #     device=device,
        # )

        train_hist.append(logs_train)
        # valid_hist.append(logs_valid)
        # score = logs_valid[metric.name]

        torch.save(model.state_dict(), os.path.join(args.save_model, f"{model_name}.pth"))
        print("Model saved in the folder : ", args.save_model)
        print("Model name is : ", model_name)
     
            
def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # using UNet with EfficientNet-B4 backbone
    model = Net(pretrained=True).cuda(device=0)
    
    # count parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters: ", params)

    classes_wt = np.ones([len(args.classes)+1], dtype=np.float32)
    criterion = source.losses.CEWithLogitsLoss(weights=classes_wt)
    metric = source.metrics.IoU2()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
    if torch.cuda.device_count() > 1:
        print("Number of GPUs :", torch.cuda.device_count())
        model = NamedDataParallel(model, name="DataParallel")
        optimizer = torch.optim.Adam(
            [dict(params=model.module.parameters(), lr=args.learning_rate)]
        )
    
    print("Number of epochs   :", args.n_epochs)
    print("Number of classes  :", len(args.classes)+1)
    print("Batch size         :", args.batch_size)
    print("Device             :", device)
               
    # training model
    train_model(args, model, optimizer, criterion, metric, device)
    
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--n_epochs', default=50)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--num_workers', default=8)
    parser.add_argument('--crop_size', default=512)
    parser.add_argument('--learning_rate', default=0.0001)  
    parser.add_argument('--classes', default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--train_data_root', default="/kaggle/input/dfc25-track1-trainval/train/labels")
    # parser.add_argument('--val_data_root', default="dataset/val/sar_images")
    parser.add_argument('--save_model', default="weight")
    parser.add_argument('--save_results', default="results")
    args = parser.parse_args()
    
    start = time.time()
    main(args)
    end = time.time()
    print('Processing time:', end - start)
