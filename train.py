import os
import torch.optim as optim
import datetime
import sys
import warnings
import ssl
import torch

ssl._create_default_https_context = ssl._create_unverified_context
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model import AAAF
from utility.image_process import *
from utility.evaluate import batch_PSNR
from utility.init import init_train
from loss import get_loss
from config import cfg

def train_model(model, train_loader, args):

    MSE_Loss, SSIM_Loss = get_loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)

    prev_time = time.time()
    print(f"Starting training")
    for epoch in range(0, args.epochs):
        for i, batch in enumerate(train_loader):
            img = Variable(batch[0]).cuda()
            gt = Variable(batch[1]).cuda()

            optimizer.zero_grad()
            img_enhance = model(img)

            loss_pixel = MSE_Loss(img_enhance, gt)
            loss_ssim = -SSIM_Loss(img_enhance, gt)

            ssim_value = -loss_ssim.item()
            loss_ssim = 1 + loss_ssim

            loss = args.lambda_pixel * loss_pixel + args.lambda_ssim * loss_ssim

            loss.backward(retain_graph=True)
            optimizer.step()

            batches_done = epoch * len(train_loader) + i
            batches_left = args.epochs * len(train_loader) - batches_done
            out_train = torch.clamp(img_enhance, 0., 1.)
            psnr_train = batch_PSNR(out_train, gt, 1.)
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()


        scheduler.step()

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [PSNR: %.2f] [SSIM: %.4f] [Loss: %.4f] [pixel_loss: %f] [ssim_loss: %f]   ETA: %s\n"
                % (
                    epoch, args.epochs, i + 1, len(train_loader),
                    psnr_train, ssim_value, loss.item(), loss_pixel.item(),loss_ssim.item(),
             time_left
                )
            )

            if (epoch % args.checkpoint_save_interval == 0):
                    torch.save(model.state_dict(),
                               os.path.join(args.model_save_dir, f"LightUNet_{epoch}.pth"))


def training(model, data_set):
    config = cfg(data_set)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
    torch.set_default_tensor_type(torch.FloatTensor)

    train_loader, Model = init_train(config, model)

    train_model(Model, train_loader, config)


if __name__ == '__main__':
    training(AAAF, "UIEB")
