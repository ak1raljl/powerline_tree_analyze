import argparse
import os
import torch
import datetime
import sys
from pathlib import Path
import importlib
import numpy as np
import provider
from tqdm import tqdm
from data.eclair_npy_dataset import EclairNpyDataset, ECLAIR_CLASSES

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = ECLAIR_CLASSES
class2label = {cls: i for i, cls in enumerate(seg_classes)}
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes)}

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Training with NPY Dataset')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save logs and models')
    parser.add_argument('--npoints', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--block_size', type=float, default=20.0, help='Block size in meters [default: 20.0]')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='Sample rate [default: 1.0]')
    parser.add_argument('--npy_dir', type=str, default='data/eclair/npy_preprocessed', help='Path to NPY dataset root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading [default: 4]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving checkpoints (in epochs)')
    parser.add_argument('--pin_memory', action='store_true', help='Enable pin memory for faster GPU transfer')
    parser.add_argument('--split_ratio', type=float, default=0.1, help='Validation split ratio [default: 0.1]')
    return parser.parse_args()

def main(args):
    args = parse_args()

    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    log_dir = Path('./logs/')
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath('eclair_npy_seg')
    log_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        log_dir = log_dir.joinpath(time_str)
    else:
        log_dir = log_dir.joinpath(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    # checkpoint_dir = log_dir.joinpath('checkpoints/')
    # checkpoint_dir.mkdir(exist_ok=True)

    NUM_CLASSES = len(ECLAIR_CLASSES)

    train_dataset = EclairNpyDataset(
        npy_dir=args.npy_dir,
        split='train',
        num_points=args.npoints,
        block_size=args.block_size,
        sample_rate=args.sample_rate,
        transform=None,
        split_ratio=args.split_ratio
    )

    val_dataset = EclairNpyDataset(
        npy_dir=args.npy_dir,
        split='val',
        num_points=args.npoints,
        block_size=args.block_size,
        sample_rate=args.sample_rate,
        transform=None,
        split_ratio=args.split_ratio
    )

    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False
    )

    valDataLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        drop_last=True,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False
    )

    weights = torch.Tensor(train_dataset.label_weights).cuda()

    model = importlib.import_module(args.model)
    classifier = model.get_model(NUM_CLASSES).cuda()
    criterion = model.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    try:
        checkpoint_path = log_dir / 'best_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path))
            start_epoch = checkpoint['epoch'] + 1
            classifier.load_state_dict(checkpoint['model_state_dict'])
            print(f"Use pretrain model: epoch {start_epoch}")
        else:
            print('No existing model, starting training from scratch...')
            classifier = classifier.apply(weights_init)
    except Exception as e:
        print(f'loading checkpoint failed: {e}')
        print('No existing model, starting training from scratch...')
        classifier = classifier.apply(weights_init)

    # optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    learning_rate_clip = 1e-5
    momentum_original = 0.1
    momentum_decay = 0.5
    momentum_decay_step = args.step_size

    global_epoch = start_epoch
    best_iou = 0.0

    for epoch in range(start_epoch, args.epoch):
        print('Epoch %d (%d/%s)' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (0.7  ** (epoch // args.step_size)), learning_rate_clip)
        print('Learning rate: %f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = momentum_original * (momentum_decay  ** (epoch // momentum_decay_step))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum: %.4f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        classifier.train()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_seen = 0

        for batch_idx, (points, target) in enumerate(tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9, desc='training')):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.random_scale_point_cloud(points[:, :, :3])
            points[:, :, :3] = provider.shift_point_cloud(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, batch_label, trans_feat, weights)

            # Calculate training accuracy
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label.cpu().data.numpy())
            total_correct += correct
            total_seen += (args.batch_size * args.npoints)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += points.shape[0]

        avg_loss = total_loss / len(trainDataLoader)
        print('Training mean loss: %f' % (avg_loss))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % args.save_freq == 0:
                save_path = log_dir / f'epoch_{epoch}_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, str(save_path))
                print(f'\nSaving checkpoint: {save_path}')

        # Validation
        with torch.no_grad():
            num_batches = len(valDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            label_weights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier.eval()
            print('Epoch %d validation...' % (epoch + 1))
            for batch_idx, (points, target) in enumerate(tqdm(valDataLoader, total=len(valDataLoader), smoothing=0.9)):
                cur_batch_size, NUM_POINT, _ = points.size()
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred_flat = seg_pred.contiguous().view(-1, NUM_CLASSES)
                cur_pred_val = seg_pred.cpu().data.numpy()

                batch_label = target.cpu().data.numpy()
                target_flat = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred_flat, target_flat, trans_feat, weights)
                loss_sum += loss
                cur_pred_val = np.argmax(cur_pred_val, 2)

                correct = np.sum((cur_pred_val == batch_label))
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                label_weights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((cur_pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((cur_pred_val == l) | (batch_label == l)))

            label_weights = label_weights.astype(np.float32) / np.sum(label_weights.astype(np.float32))
            iou_per_class = []
            for l in range(NUM_CLASSES):
                if total_iou_deno_class[l] > 0:
                    iou = total_correct_class[l] / float(total_iou_deno_class[l])
                    iou_per_class.append(iou)
                else:
                    iou_per_class.append(np.nan)
            valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
            mIoU = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0

            print('Validation mean loss: %f' % (loss_sum / num_batches))
            print('Validation accuracy: %f' % (total_correct / float(total_seen)))
            print('Validation mIoU: %f' % mIoU)
            print('Per class IoU:')
            print(f'{"ID":>3} {"Class Name":<20} {"Weight":>8} {"IoU":>8}')
            for l in range(NUM_CLASSES):
                class_name = seg_classes[l]
                weight = label_weights[l]
                if total_iou_deno_class[l] > 0:
                    iou = total_correct_class[l] / float(total_iou_deno_class[l])
                    if not np.isnan(iou):
                        print(f'{l:3d} {class_name:20s} {weight:8.4f} {iou:8.4f} ')
                    else:
                        print(f'{l:3d} {class_name:20s} {weight:8.4f} {"N/A":>8} ')
                else:
                    print(f'{l:3d} {class_name:20s} {weight:8.4f} {"N/A":>8} ')

            if mIoU > best_iou:
                best_iou = mIoU
                save_path = log_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'mIoU': mIoU,
                }, str(save_path))
                print(f'Best model (mIoU: {best_iou:.4f})')

        global_epoch += 1
        print("-" * 60)

    print('\n' + '=' * 60)
    print(f'Training complete! Best mIoU: {best_iou:.4f}')
    print('=' * 60)

if __name__ == '__main__':
    main(parse_args())
