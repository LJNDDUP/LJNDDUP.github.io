---
layout:       post
title:        Effective Pytorch
subtitle:     
date:         2019-09-23
author:       JAN
header-img: img/post-bg-coffee.jpeg
catalog:      true
tags:
- Coding
---

# 条款01: parser

```python
def parse_command():
	import argparse
	parser = argparse.ArgumentParser(description='JAN')
	parser.add_argument('--resume', default=None, type=str, metavar='PATH',
		help='path to latest checkpoint (default: ./example.checkpoint.pth.tar')
	parser.add_argument('--batch-size', default=4, type=int, 
		help='mini-batch size (default: 4)')
	parser.add_argument('--epochs', default=100, type=int, metavar='N',
		help='number of total epoch to run (default: 100)')
	parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', 
		help='initial learning rate (default: 1e-3)')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
		help='momentum (default: 0.9)')
	parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W',
		help='weight decay (default: 5e-4)')
	parser.add_argument('--dataset', default='kitti', type=str,
		help='dataset used for training, kitti and nyu is available')
	parser.add_argument('--manual-seed', default=1, type=int, 
		help='manaully set random seed (default: 1)')
	parser.add_argument('--gpu', default='0', type=str, 
		help='gpu id (default: '0')')
	parser.add_argument('--print-freq', default=10, type=int, metavar='F', 
		help='print frequence (default: 10)')
	args = parser.parse_args()
	return args
```

# 条款02: output directory

```python
def get_output_directory(args):
	save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
	save_dir_root = os.path.join(save_dir_root, 'result', args.dataset)
	if args.resume:
		runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
		run_id = int(runs[-1].split('_')[-1]) if runs else 0
	else:
		runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
		run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
	save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
	return save_dir
```

# 条款03: save checkpoint

```python
save_checkpoint({
	'args': args,
	'epoch': epoch,
	'model': model.state_dict(),
	'best_result': best_result,
	'optimizer:' optimizer
	}, is_best, epoch, output_directory)
import shutil
def save_checkpoint(state, is_best, epoch, output_directory):
	checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
	torch.save(state, checkpoint_filename)
	if is_best:
		best_filename = os.path.join(output_directory, 'model_best.pth.tar')
		shutil.copyfile(checkpoint_filename, best_filename)
```

# 条款04: color depth map

```python
import matplotlob.pyplot as plt 
camp = plt.cm.jet
def color_depth_map(depth, d_min=None, d_max=None):
	if d_min is None:
		d_min = np.min(depth)
	if d_max is None:
		d_max = np.max(depth)
	depth_relative = (depth - d_min) / (d_max - d_min)
	return 255 * cmap(depth_relative)[:, :, :3]
```

# 条款05: save image

```python
def merge_into_row(input, depth_target, depth_pred):
	rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy), (1, 2, 0))  # (H, W, C)
	depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
	depth_pred_cpu = np.squeeze(depth_pred.cpu().numpy())

	d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
	d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
	depth_target_col = color_depth_map(depth_target_cpu, d_min, d_max)
	depth_pred_col = color_depth_map(depth_pred_cpu, d_min, d_max)
	img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

	return img_merge

def add_row(img_merge, row):
	return np.vstack(img_merge, row)

from PIL import Image
def save_image(img_merge, filename):
	img_merge = Image.fromarray(img_merge.astype('uint8'))
	img_merge.save(filename)

# i: iteration
if i == 0:
	img_merge = utils.merge_into_row(rgb, target, pred)
elif i < 8 * skip and (i % skip) == 0:
	row = utils.merge_into_row(rgb, target, pred)
	img_merge = utils.add_row(img_merge, row)
elif i == 8 * skip:
	filename = os.path.join(output_directory, str(epoch) + '.png')
	utils.save_image(img_merge, filename)
```

# 条款06: depth metircs

```python
import math
import torch
import numpy as np

def log10(x):
	return torch.log(x) / math.log(10)

class Result(object):
	def __init__(self):	
		self.absrel, self.sqrel = 0., 0.	
		self.mse, self.rmse, self.mae = 0., 0., 0.
		self.irmse, self.imae = 0., 0.
		self.delta1, self.delta2, self.delta3 = 0., 0., 0.
		# self.silog = 0.
		self.data_time, self.gpu_time = 0., 0.

	def set_to_worst(self):
		self.absrel, self.sqrel = np.inf, np.inf  
		self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf 
		self.irmse, self.imae = np.inf, np.inf
		self.delta1, self.delta2, self.delta3 = 0., 0., 0.
		# self.silog = np.inf
		self.data_time, self.gpu_time = 0., 0.

	def update(self, absrel, sqrel, mse, rmse, mae, irmse, imae, delta1, delta2, delta3, data_time, gpu_time):
		self.absrel, self.sqrel = absrel, sqrel
		self.mse, self.rmse, self.mae = mse, rmse, mae
		self.irmse, self.imae = irmse, imae
		self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
		# self.silog = silog
		self.data_time, self.gpu_time = data_time, gpu_time

	def evaluate(self, output, target):
		valid_mask = target > 0
		output = output[valid_mask]
		target = target[valid_mask]

		abs_diff = (output - target).abs()
		self.mse = float(abs_diff.pow(2).mean())
		self.rmse = math.sqrt(self.mse)
		self.mae = float(abs_diff.mean())
		self.absrel = float((abs_diff / target).mean())
		self.sqrel = float((abs_diff.pow(2) / target).mean())

		inv_output = 1. / output
		inv_target = 1. / target
		abs_inv_diff = (inv_output - inv_target).abs()
		self.irmse = math.sqrt(abs_inv_diff.pow(2).mean())
		self.imae = float(abs_inv_diff.mean())

		maxRatio = (output / target, target / output).max()
		self.delta1 = float((maxRatio < 1.25).float().mean())
		self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
		self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

		# log_diff = log10(output) - log10(target)
		# self.silog = log_diff.pow(2).mean() - log_diff.sum().pow(2) / log_diff.numel().pow(2)

	class AverageMeter(object):
		def __init__(self):
			self.reset()

		def reset(self):
			self.count = 0.
			self.sum_absrel, self.sum_sqrel = 0., 0.	
			self.sum_mse, self.sum_rmse, self.sum_mae = 0., 0., 0.
			self.sum_irmse, self.sum_imae = 0., 0.
			self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0., 0., 0.
			self.sum_data_time, self.sum_gpu_time = 0., 0.

		def update(self, result, data_time, gpu_time, n=1):
			self.count += n
			self.sum_absrel += n*result.absrel 
			self.sum_sqrel += n*result.sqrel 
			self.sum_mse += n*result.mse 
			self.sum_rmse += n*result.rmse 
			self.sum_mae += n*result.mae 
			self.sum_irmse += n*result.irmse 
			self.sum_imae += n*result.imae
			self.sum_delta1 += n*result.delta1
			self.sum_delta2 += n*result.delta2
			self.sum_delta3 += n*result.delta3
			self.sum_data_time += n*result.data_time
			self.sum_gpu_time += n*result.gpu_time

		def average(self):
			avg = Result()
			avg.update(
				self.sum_absrel / self.count, self.sum_sqrel / self.count,
				self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
				self.sum_irmse / self.count, self.sum_imae / self.count,
				self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
				self.sum_data_time / self.count, self.sum_gpu_time / self.count)
			return avg
```

# 条款07: set random seed

```python
import random
import numpy as np
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)
random.seed(args.manual_seed)
```

# 条款08: resume

```python
if args.resume:
	assert os.path.isfile(args.resume), \
		'=> no checkpoint found at {}'.format(args.resume)
	print('=> loading checkpoint {}'.format(args.resume))
	checkpoint = torch.load(args.resume)
	start_epoch = checkpoint['epoch'] + 1
	best_result = checkpoint['best_result']
	optimizer = checkpoint['optimizer']
	model = get_models(args.dataset, pretrained=True)
	model.load_state_dict(checkpoint['model'])
	print('=> loaded checkpoint (epoch {})'.format(checkpoint['epoch']))
	# clear memory
	del checkpoint
	# del model_dict
	torch.cuda.empty_cache()
else:
	print('=> creating model')
	model = get_models(args.dataset, pretrained=True)
	print('=> model created')
	start_epoch = 0
	train_params = [{'params': model.get_1x_lr_params, 'lr': args.learning_rate},
		{'params': model.get_10x_lr_params, 'lr': args.learning_rate * 10}]
	optimizer = torch.optim.SGD(train_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	# you can use DataParallel() whether you use multi-gpus or not
	model = nn.DataParallel(model).cuda()
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)
# ...
# scheduler.step(result.absrel)
```

# 条款09: config file

```python
config_txt = os.path.join(output_directory, 'config.txt')
if not os.path.exists(config_txt):
	with open(config_txt, 'w') as f_w:
		args_ = vars(args)
		args_str = ''
		for k, v in args_.items():
			args_str += str(k) + ':' + str(v) + ',\t\n'
		f_w.write(args_str)
```

# 条款10：create log

```python
import socket
from datetime import datetime
from tensorboardX import SummaryWriter
log_path = os.path.join(output_directory, 'logs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
os.makedirs(log_path)
logger = SummaryWriter(log_path)
# ...
# logger.close()
```

# 条款11: record the learning rate

```python
for i, param_group in enumerate(optimizer.param_groups):
	lr = float(param_group['lr'])
	logger.add_scalar('LR/lr_' + str(i), lr, epoch)
```

# 条款12: best file

```python
best_txt = os.path.join(output_directory, 'best.txt')
# define your best
is_best = result.rmse < best_result.rmse 
if is_best:
	best_result = result
	with open(best_txt, 'w') as f_w:
		f_w.write('epoch: {}, absrel: {:.3f}...'.format(epoch, best_result.absrel, ...))
	if img_merge is not None:
		img_filename = os.path.join(output_directory, 'best.png')
		save_image(img_merge, img_filename)
```

# 条款13: record time

```python
import time
start = time.time()
# load data
torch.cuda.synchronize()
data_time = time.time() - start 
start = time.time()
# forward and backward
torch.cuda.synchronize()
gpu_time = time.time() - start
```

# 条款14: torch.autograd.detect_anomaly()

```python
with torch.autograd.detect_anomaly():
	# forward
	# backward
```
