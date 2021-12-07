import torch.nn as nn


class Loss(object):
    def __init__(self):
        self.loss_cls = nn.CrossEntropyLoss()

        self.loss_box = nn.L1Loss()

        self.loss_objectness = nn.CrossEntropyLoss()

    def get_loss(self, outputs, targets, args):
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

        每个grid有5个anchor，预测怎么和anchor联系起来，没想明白
        """

        # coco共有80类，数据集标签中，取值从[1,80]，下标范围 [0,79]
        category_id = targets['category_id']    
        bbox = targets['bbox']

        loss_cls = self.loss_cls(outputs, category_id-1)
        loss_box = self.loss_box(outputs, targets)
        loss_objectness = self.loss_objectness(outputs, targets)

        self.total_loss = args.lambda_cls * loss_cls + args.lambda_box * loss_box + args.lambda_objectness * loss_objectness

        return self.total_loss