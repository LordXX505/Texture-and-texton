import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from util import get_output_dir,set_seed,set_cuda,set_gpu,\
    copy_source,to_named_dict,setup_logging,\
    plot_stats,save_images
import torch.optim as optim
from ops import weights_init_xavier, DecodeBlock

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--gpuid', default='0', help='GPU ids of running')
    parser.add_argument('--datapath', type=str, default='./Image/')
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--img_channels', default=3, type=int)
    parser.add_argument('--channel_max', default=512, type=int)
    parser.add_argument('--batch_size', default=11, type=int)
    parser.add_argument('--ninterp', default=10, type=int)
    parser.add_argument('--nz', type=int, default=32, help='size of the latent z')
    parser.add_argument('--l_steps', type=int, default=45, help='number of langevin steps')
    parser.add_argument('--l_step_size', type=float, default=0.1, help='stepsize of langevin')
    parser.add_argument('--sigma', type=float, default=0.5, help='prior of llhd, annealing parameter')
    parser.add_argument('--prior_sigma', type=float, default=1, help='prior of z')
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--gamma', default=0.996, help='lr decay')
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--n_epochs', default=5000, type=int)
    parser.add_argument('--n_printout', type=int, default=1, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=500, help='plot each n epochs')
    parser.add_argument('--n_ckpt', type=int, default=1000, help='save ckpt each n epochs')
    parser.add_argument('--n_stats', type=int, default=1000, help='stats each n epochs')
    parser.add_argument('--channel_base', type=int, default=2048, help='stats each n epochs')
    return parser.parse_args()


# class GenNet(nn.Module):
#     ####################################################
#     # Define the structure of generator,
#     # ops.py defines the basic block you may use
#     ####################################################
#     def __init__(self, args):
#         super().__init__()
#         self.fc = nn.Linear(in_features=args.nz, out_features=64*8*4*4, bias=True)
#         self.fc_op = nn.Sequential(
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
#         )
#         self.convt1 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, padding=2),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.convt2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, padding=2),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.convt3 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.convt4 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.convt5 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2),
#         )
#         self.tanh = nn.Tanh()
#
#     def decode(self, z, flag):
#         z = self.fc(z)
#         z = z.view(-1, 512, 4, 4)
#         z = self.fc_op(z)
#         if flag:
#             print(z[0, 0, :, :])
#         z = self.convt1(z)
#         z = self.convt2(z)
#         z = self.convt3(z)
#
#         z = self.convt4(z)
#         # print(z)
#         if flag:
#             print(z[0, 0, :, :])
#         z = self.convt5(z)
#         if flag:
#             print(z[0, 0, :, :])
#         z = self.tanh(z)
#         return z
#
#     def forward(self, z, flag=False):
#         flag = False
#         return self.decode(z, flag)

class GenNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.global_iter = 0
        self.img_channels = args.img_channels
        self.img_resolution = args.img_size
        self.img_resolution_log2 = int(np.log2(args.img_size))
        self.decode_block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        # 4:8:...:resolution
        decode_channels_dict = {res: min(args.channel_base // res, args.channel_max) for res in
                                self.decode_block_resolutions[:-1]}
        # 4*  :8*  :...(res//4)*64:(res//2)*32
        for res in self.decode_block_resolutions[:-1]:
            in_channels = decode_channels_dict[res]
            out_channels = decode_channels_dict[res * 2] if res != self.decode_block_resolutions[-2] else args.img_channels
            is_last = (res == self.decode_block_resolutions[-2])
            block = DecodeBlock(in_channels, out_channels, resolution=res, is_last=is_last)
            setattr(self, f'decode_b{res}', block)
        self.fc_d1 = nn.Linear(args.nz, 4 * 4 * decode_channels_dict[4])
        self.decode_channels_dict = decode_channels_dict
        self.flatten = nn.Flatten()

    def decode(self, z, flag=False):
        x = F.relu(self.fc_d1(z)).view(z.shape[0], self.decode_channels_dict[4], 4, 4)
        for res in self.decode_block_resolutions[:-1]:
            block = getattr(self, f'decode_b{res}')
            x = block(x)
        return x

    def forward(self, z, flag=False):
        return self.decode(z)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.Gnet = GenNet(args)
        self.Gnet.apply(weights_init_xavier)

    def sample_langevin(self, z, x, Gnet, args):
        mse = nn.MSELoss(reduction='sum')
        z = z.clone().detach()
        z.requires_grad = True
        for i in range(args.l_steps):
            x_hat = Gnet(z)
            Gnet.zero_grad()
            loss = 1.0 / (2.0 * args.sigma * args.sigma) * mse(x_hat, x)
            loss.backward()
            z.data = z.data - 0.5 * args.l_step_size * args.l_step_size * (z.grad + 1.0 / (
                    args.prior_sigma * args.prior_sigma) * z.data)
            z.data += torch.randn_like(z).data
        return z.detach()


    def forward(self, z, x):
        return self.sample_langevin(z, x, self.Gnet, self.args)


def predict(args, output_dir, device='cpu'):
    output_dir = os.path.join('./output', output_dir)
    path = os.path.join(output_dir, 'ckpt/best_model.pth')
    model = Model(args)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    z = 25 * torch.randn([81, args.nz])
    # index = np.ceil(np.random.rand(2)*args.nz).astype(np.int16)
    # k = 0
    # for i in np.linspace(-40, 40, num=9, endpoint=False):
    #     for j in np.linspace(-40, 40, num=9, endpoint=False):
    #         z[k][index[0]] = torch.tensor(data=[i])
    #         z[k][index[1]] = torch.tensor(data=[j])
    #         k += 1


    re_img = model.Gnet(z)
    save_images(re_img,
                os.path.join(output_dir, 'syn.png'), nrow=9)

def train(args, output_dir,device,logger):
    # Prepare training data
    dataset = datasets.ImageFolder(root=args.datapath,
                                  transform=transforms.Compose([
                                      transforms.Resize([args.img_size,args.img_size]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, **kwargs)
    model = Model(args).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=[args.beta1, args.beta2], eps=1e-8, weight_decay=args.gamma)
    red_z = torch.randn([args.batch_size, args.nz]).to(device)
    eps = args.l_step_size
    loss_max = np.inf
    best_index = 0
    loss_list = []
    for epoch in range(args.n_epochs+1):
        for train_data, label in train_loader:
            train_data = train_data.to(device)
            model.train()
            red_res = model(red_z, train_data)

            red_restruct_img = model.Gnet(Variable(red_res.data, requires_grad=False))
            loss = torch.mean(torch.sum(1. / eps / eps / 2 * torch.pow((train_data - red_restruct_img), 2), dim=[1, 2, 3]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().numpy())
            red_z = red_res.clone().detach()
            model.eval()
            if epoch > 0 and epoch % args.n_printout == 0:
                logger.info('{:>5d} loss={:>18.2f}'.format(epoch, loss))
            if epoch > 0 and epoch % args.n_stats == 0:
                logger.info(model.state_dict())
            # --------------------------------------------------------------------------------------------------------------------------
            red_restruct_img = model.Gnet(Variable(red_z.data, requires_grad=False), flag=True)

            if epoch > 0 and epoch % args.n_plot == 0:
                save_images(red_restruct_img,
                            os.path.join(output_dir, "samples/red_restructed_img_{0}.png".format(str(epoch))), nrow=3)
                random_z = 10 * torch.randn(81, args.nz).to(device)
                random_img = model.Gnet(Variable(random_z.data, requires_grad=False))
                save_images(random_img,
                            os.path.join(output_dir, "samples/random_restructed_img_{0}.png".format(str(epoch))), nrow=9)
            if loss < loss_max:
                loss_max = loss
                best_index = epoch
                torch.save(model.state_dict(),
                           os.path.join(output_dir, "ckpt/" + 'best_model.pth'.format(epoch)))
            if epoch > 0 and epoch % args.n_ckpt == 0:
                logger.info(red_z)
                torch.save(model.state_dict(),
                           os.path.join(output_dir, "ckpt/"+ 'descriptor_{}.pth'.format(epoch)))
    logger.info('best_index:{:>5d}'.format(best_index))
    plt.plot(range(2000), loss_list[:2000], 'b')
    plt.savefig(os.path.join(output_dir, './loss.png'))
    plt.show()
    ####################################################
    # Define the learning process. Train the model here.
    # Print the loss term at each epoch to monitor the
    # training process. You may use setup_logging() and
    # save_images() in ./util.py. At each log_step,
    # save the reconstructed images and synthesized images,
    # and record model ckpt.pth.
    ####################################################

def main():
    exp_id = os.path.splitext(os.path.basename(__file__))[0] + '32'
    test = 1
    output_dir = get_output_dir(exp_id, test)
    args = parse_args()
    args = to_named_dict(args)

    set_seed(args.seed)
    set_cuda(deterministic=args.gpu_deterministic)
    set_gpu(args.gpuid)
    device = torch.device('cuda:{}'.format(args.gpuid) if torch.cuda.is_available() else 'cpu')
    args.device = device

    if test == 1:
        copy_source(__file__, output_dir)

        logger = setup_logging('main', output_dir, console=True)
        logger.info(args)

    train(args, output_dir, device, logger)
    # predict(args, os.path.join(exp_id, '2022-12-18-17-21-11'), device)


if __name__ == '__main__':
    main()