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
from data.eclairDataloader import EclairDataset, ECLAIR_CLASSES

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
    parser = argparse.ArgumentParser('Training Script')
    parser.add_argument('--model', type=str, default='seg', help='Model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs and models')
    parser.add_argument('--npoints', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--block_size', type=float, default=100.0, help='Block size in meters [default: 100.0]')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='Sample rate [default: 1.0]')
    parser.add_argument('--data_root', type=str, default='data/eclair/', help='Path to ECLAIR dataset root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving checkpoints (in epochs)')
    return parser.parse_args()

def main(args):
    args = parse_args()
    time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    log_dir = Path('./logs/')
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath('eclair_seg')
    log_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        log_dir = log_dir.joinpath(time)
    else:
        log_dir = log_dir.joinpath(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = log_dir.joinpath('checkpoints/')
    checkpoint_dir.mkdir(exist_ok=True)

    NUM_CLASSES = 11

    print("Loading ECLAIR dataset...")
    print(f"Data root: {args.data_root}")

    train_dataset = EclairDataset(
        root_dir=args.data_root,
        split='train',
        num_points=args.npoints,
        block_size=args.block_size,
        sample_rate=args.sample_rate,
        transform=None,
    )

    val_dataset = EclairDataset(
        root_dir=args.data_root,
        split='val',
        num_points=args.npoints,
        block_size=args.block_size,
        sample_rate=args.sample_rate,
        transform=None,
    )

    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    valDataLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True
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
    try:
        checkpoint = torch.load(str(log_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    
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

    global_epoch = 0
    best_iou = 0.0

    for epoch in range(start_epoch, args.epoch):
        print('Epoch %d (%d/%s)' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (0.7 ** (epoch // args.step_size)), learning_rate_clip)
        print('Learning rate: %f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = momentum_original * (momentum_decay ** (epoch // momentum_decay_step))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (args.batch_size * args.npoints)
            loss_sum += loss

        print('Training mean loss: %f' % (loss_sum / num_batches))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % args.save_freq == 0:
            savepath = str(checkpoint_dir) + '/model_' + str(epoch) + '.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, savepath)
            print('Model saved at %s' % savepath)
        
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
            for i, (points, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (args.batch_size * args.npoints)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))

            print('Validation mean loss: %f' % (loss_sum / num_batches))
            print('Validation point avg class IoU: %f' % mIoU)
            print('Validation point accuracy: %f' % (total_correct / float(total_seen)))
            print('Validation point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
            iou_per_class_str = 'IoU per class: '
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (20 - len(seg_label_to_cat[l])),
                    labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6))
            print(iou_per_class_str)
            print('Validation mean loss: %f' % (loss_sum / num_batches))
            print('Validation accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = str(checkpoint_dir) + '/best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, savepath)
                print('Best model saved at %s' % savepath)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)