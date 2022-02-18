"""
anchors of the dataset
"""

# reference - https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
# 9 anchors of yolov3
# (10×13),(16×30),(33×23),(30×61),(62×45),(59× 119),(116 × 90),(156 × 198),(373 × 326)

# reference - https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg
# 5 anchors of yolov2
ANCHORS = [
    (18.32736, 21.67632),
    (59.98273, 66.00096),
    (106.82976, 175.17888),
    (252.25024, 112.88896),
    (312.65664, 293.38496)
]