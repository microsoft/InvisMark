from argparse import ArgumentParser
import logging
import sys

import train
import configs

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path, batch_size, num_workers=8):
    dataset = dset.ImageFolder(root=path, transform=transforms.Compose([
                            transforms.ToTensor(), # scale image pixels from [0, 255] to [0, 1] values
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [-1, 1]
                            ]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    

def main(args, device):
    torch.manual_seed(42)
    cfg = configs.ModelConfig(
        lr = args.lr,
        num_epochs = args.num_epochs,
        ckpt_path = args.ckpt_path,
        saved_ckpt_path = args.saved_ckpt_path,
        log_dir = args.log_dir,
        num_encoded_bits=args.num_bits,
        wm_repeats = args.wm_repeats,
        image_shape = (args.image_size, args.image_size),
        batch_size = args.batch_size,
        beta_epochs = args.beta_epochs,
        beta_max = args.beta_max,
    )

    train_data = load_dataset(args.train_path, args.batch_size)
    # use batch=1 to be compatible with different image resolution w/o resizing.
    eval_data = load_dataset(args.eval_path, 1, num_workers=0)

    wm_model = train.Watermark(cfg, device=device)
    wm_model.train(train_data, eval_data, args.saved_ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./runs/")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts/")
    parser.add_argument("--saved_ckpt_path", type=str)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_bits", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta_max", type=float, default=40.)
    parser.add_argument("--beta_epochs", type=int, default=20)
    
    torch.cuda.empty_cache()
    command_args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(command_args, device)
