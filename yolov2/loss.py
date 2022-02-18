import torch
import torch.nn as nn
from anchors import ANCHORS


class Loss(object):
    def __init__(self):
        self.loss_cls = nn.CrossEntropyLoss()

        self.loss_box = nn.L1Loss()

        self.loss_objectness = nn.CrossEntropyLoss()

    def get_total_loss(self, outputs, targets, args):
        """
        dimension of outputs: batch_size x fea_height x fea_width x 125
        dimension of targets: batch_size x dim(annos)

        cls预测的是物体类别概率分布
        box预测的到底是什么？标签：是中心点坐标和长宽，而且经过了归一化
        objectness预测的是是否包含物体

        annos是一个list，一个元素对应一张图的标签
            [
                {
                    'segmentation': [],
                    'area': ...,
                    'iscrowd': ...,
                    'image_id' ...,
                    'bbox': [],
                    'category_id': ...,
                    'id': ...
                },
                ...,
                {
                    ...
                }
            ]

        问题是每个元素是个字典，pytorch里有字典吗？看这里的代码，有涉及到pytorch吗？
            当然有，传进来的outputs, targets都是pytorch的变量吧

        每个grid有5个prediction box，prediction box怎么和ground truth联系起来，没想明白
            1. 现在想明白了，输出feature map的尺寸为SxS，每个位置都有5个prediction box
            2. 而最后的feature map尺寸就是SxS，它正是用来匹配这SxS的网格
            3. ground truth 从标签而来

        弄明白yolov2-coco设置的anchor大小：
            width-height: (10×13),(16×30),(33×23),(30×61),(62×45),(59× 119),(116 × 90),(156 × 198),(373 × 326)
            这个大小应该指的绝对大小
            yolov2里说只用到了5个anchor，到底是哪5个anchor？

        怎么把prediction box和真实box匹配，计算loss

        决定了，就在计算loss时候转换真实标签为同样维度：SxSx5(80+4+1)

        * COCO数据集加载出来的格式，不太适合YOLO直接训练，需要转换成对YOLO友好的
        * 假如说数据格式转换好了，我应该怎么写loss，想象起来比之前简单了
        * 理解了YOLO标签中数值为什么要归一化
        """
        self.total_loss = t.tensor(.0)

        batch_size, fea_height, fea_width, _ = targets.shape

        # 要把target分解，做成和outputs维度一样的数组
        bbox = targets['bbox']
        # coco共有80类，数据集标签中，取值从[1,80]，下标范围 [0,79]
        category_id = targets['category_id']    


        # 都向量化了，不要再用双层循环
        for j in range(fea_height):
            for k in range(fea_width):
                
                for i in range(len(ANCHORS)):

                    pred_objectness = outputs[85*i]
                    pred_bbox = outputs[85*i + 1: 85*i + 5]
                    pred_category = outputs[85*i + 5: 85*(i+1)]

                    tx, ty, tw, th = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]
                    bx = torch.sigmoid(tx) + k*32/args.image_width
                    by = torch.sigmoid(ty) + j*32/args.image_height
                    bw = ANCHORS[i][0]/args.image_width * torch.exp(tw)
                    bh = ANCHORS[i][1]/args.image_height * torch.exp(th)
        
                    loss_cls = self.loss_cls(outputs, category_id-1)
                    loss_box = self.loss_box(outputs, targets)
                    loss_objectness = self.loss_objectness(outputs, targets)

                    self.total_loss += args.lambda_cls * loss_cls + args.lambda_box * loss_box + args.lambda_objectness * loss_objectness

        return self.total_loss