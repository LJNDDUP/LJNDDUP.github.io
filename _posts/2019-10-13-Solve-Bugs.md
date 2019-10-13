---
layout:       post
title:        Solve Bugs
subtitle:     Bug，出来玩
date:         2019-10-13
author:       JAN
header-img: img/post-bg-ios9-web.jpg
catalog:      true
tags:
    - coding
---

# RuntimeError: Function "CudnnConvolutionBackward" returned nan values in its 0th output

训练时，整个网络随机初始化，很容易出现nan。这时候需要把学习率调小，直到不出现nan为止，如果一直有nan，那可能是网络实现问题。学习率一般和网络层数呈反比，层数越多，学习率通常要减小。可以先用较小的学习率迭代5000次，然后用这个参数初始化网络，再加大学习率去训练。

其他尝试：
- 数据本身是否存在nan，可以用numpy.any(numpy.isnan(x))去检查一下input和target；
- 图片转换为float，即/255；
- relu和softmax不要连着用，最好将relu改成tanh；
- batchsize选择过小；
- tensorflow有专门的内置调试器来调试此类问题，示例如下：
```python
from tensorflow.python import debug as tf_debug
<meta charset="utf-8">

# 建立原来的Session

sess = tf.Session()

# 用tfdbg的Wrapper包裹原来的Session对象：

sess = tf_debug.LocalCLIDebugWrapperSession(sess)

sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

# 以上为所有需要的代码变动，其余的代码可以保留不变，因为包裹有的sess和原来的界面一致。

# 但是每次执行`sess.run`的时候，自动进入调试器命令行环境。

sess.run(train_op, feed_dict=...)
在tfdbg命令行环境里面，输入如下命令，可以让程序执行到inf或nan第一次出现。
tfdbg> run -f has_inf_or_nan
```
