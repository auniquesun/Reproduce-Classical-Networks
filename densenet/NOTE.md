### 问题
1. 我没有想明白的点是为什么不断增加输入通道，能模拟之前所有层的输入  
2. 现在的问题是 224x224 的输入，经过 7x7 conv(stride=2, padding=1) 怎么变成 112x112 的，我算出来总是比 112x112 小  
3. 问题集中在DenseBlock中间的各层特征图传播维度是什么呢？  
    - 明天重点解决

5. 现在就差跑模型，调bug了，但还是比较多的任务

4. 还有一个问题是训练DenseNet肯定要用多卡，PyTorch如何把数据和模型分布式放到各个GPU上？
    - 一个思路是参考**mmdetection**的方法，这个之前跑过了，算是比较顺利

5. 分布式训练教程，来自[pytorch官方文档](https://pytorch.org/docs/stable/distributed.html#launch-utility)，mmdetection做法就是从这里来的
    
6. 感觉这个代码不对啊
    - 打印日志，应该是整个模型整体打一份，现在是每个GPU打一份
    - loss一直不变
    - precision每个epoch的输出都一样