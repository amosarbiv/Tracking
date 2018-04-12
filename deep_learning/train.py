# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from pathlib import Path
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import models, transforms

from tqdm import tqdm


def to_variables(*tensors, cuda=None, **kwargs):
    if cuda is None:
        cuda = torch.cuda.is_available()

    variables = []
    for t in tensors:
        if cuda:
            t = t.cuda()
        variables.append(Variable(t, **kwargs))

    return variables


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(1)  # pause a bit so that plots are updated


class VotDatasetByImage(Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# class VotDatasetByImage(Dataset):
#     def __init__(self, path):
#         path = isinstance(path, Path) or Path(path)
#         self.videos = sorted([video for video in path.iterdir() if video.is_dir()])
#
#         self.images = []
#         for video in self.videos:
#             self.images.extend(sorted(video.rglob('*.jpg')))
#
#         labels = [np.genfromtxt(video / 'groundtruth.txt', delimiter=',') for video in self.videos]
#         self.labels = np.concatenate(labels, axis=0).astype(np.float32)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         transform = transforms.Compose([
#             transforms.Rescale((224, 224)),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x / 255.),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#         image = Image.open(self.images[idx])
#
#         factor = 224. / np.array(image.size)
#         label = self.labels[idx]
#         label[::2] /= factor[1]
#         label[1::2] /= factor[0]
#
#         return transform(image), label


class TrackerNet(nn.Module):
    def __init__(self):
        super(TrackerNet, self).__init__()
        self.squeeze = models.squeezenet1_1(pretrained=True)
        self.fc = nn.Linear(1000, 8)

        # self.squeeze = models.resnet18(pretrained=True)
        # num_ftrs = self.squeeze.fc.in_features
        # self.squeeze.fc = nn.Linear(num_ftrs, 8)


    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc(x)
        return x


def train(model, optimizer, data_loader, summary_writer, epoch):
    model.train()

    avg_stats = defaultdict(float)
    for batch_i, (x, y) in tqdm(enumerate(data_loader), desc='training'):
        x, y = to_variables(x, y, cuda=args.cuda)
        optimizer.zero_grad()

        y_ = model(x).view(-1, 8)
        loss = F.mse_loss(y_, y)

        loss.backward()
        optimizer.step()

        avg_stats['loss'] += loss.data[0]

    str_out = '[train] {}/{} '.format(epoch, args.epochs)
    for k, v in avg_stats.items():
        avg = v / len(data_loader)
        summary_writer.add_scalar(k, avg, epoch)
        str_out += '{}: {:.6f}  '.format(k, avg)

    print(str_out)


def test(model, data_loader, summary_writer, epoch):
    model.eval()

    avg_stats = defaultdict(float)
    for batch_i, (x, y) in tqdm(enumerate(data_loader), desc='testing'):
        x, y = to_variables(x, y, cuda=args.cuda)

        y_ = model(x).view(-1, 8)
        loss = F.mse_loss(y_, y)

        avg_stats['loss'] += loss.data[0]

    str_out = '[test ] {}/{} '.format(epoch, args.epochs)
    for k, v in avg_stats.items():
        avg = v / len(data_loader)
        summary_writer.add_scalar(k, avg, epoch)
        str_out += '{}: {:.6f}  '.format(k, avg)

    print(str_out)
    return avg_stats['loss']


def main():
    train_images = np.load(args.train_data_path)['images']
    train_labels = np.load(args.train_data_path)['labels']
    test_images = np.load(args.test_data_path)['images']
    test_labels = np.load(args.test_data_path)['labels']

    print('loaded data')

    # imshow(train_images)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        VotDatasetByImage(train_images, train_labels),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        VotDatasetByImage(test_images, test_labels),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    log_dir = Path(args.log_dir)
    train_writer = SummaryWriter(log_dir=(log_dir / 'train'))
    test_writer = SummaryWriter(log_dir=(log_dir / 'test'))

    model = TrackerNet()

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')

    torch.save(model.state_dict(), str(log_dir / 'last_model.checkpoint'))

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, train_writer, epoch)
        loss = test(model, test_loader, test_writer, epoch)

        torch.save(model.state_dict(), str(log_dir / 'last_model.checkpoint'))

        if loss < best_loss:
            torch.save(model.state_dict(), str(log_dir / 'best_model.checkpoint'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', required=True)
    parser.add_argument('--test-data-path', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
