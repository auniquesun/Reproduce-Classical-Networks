### 关键点
1. YOLOv2的损失函数由多部分构成，分类、边界框回归、objectness，怎么实现？
    - 我想是要对最终的输出feature map进行分段处理
    - 搞明白每部分的loss具体由哪些部分组成

2. anchor boxes在哪里设置？