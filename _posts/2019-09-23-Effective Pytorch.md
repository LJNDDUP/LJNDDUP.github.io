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

# 条款2: output directory

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

# 条款3: save checkpoint

```python
import shutil
def save_checkpoint(state, is_best, epoch, output_directory):
	checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
	torch.save(state, checkpoint_filename)
	if is_best:
		best_filename = os.path.join(output_directory, 'model_best.pth.tar')
		shutil.copyfile(checkpoint_filename, best_filename)
```

# 条款4: color depth map

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

# 条款5: save image

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

from PIL import Image
def save_image(img_merge, filename):
	img_merge = Image.fromarray(img_merge.astype('uint8'))
	img_merge.save(filename)
```

# 条款6: depth metircs

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
