# -*- coding: utf-8 -*-
# @Time    : 18-3-23 下午12:22
# @Author  : AaronJny
# @Email   : Aaron__7@163.com


# 内容图片路径
CONTENT_IMAGE = 'images/content.jpg'
# 风格图片路径
STYLE_IMAGE = 'images/style.jpg'
# 输出图片路径
OUTPUT_IMAGE = 'output/output'
# 预训练的vgg模型路径
VGG_MODEL_PATH = 'imagenet-vgg-verydeep-19.mat'
# 图片宽度
IMAGE_WIDTH = 450
# 图片高度
IMAGE_HEIGHT = 300
# 定义计算内容损失的vgg层名称及对应权重的列表
CONTENT_LOSS_LAYERS = [('conv4_2', 0.5),('conv5_2',0.5)]
# 定义计算风格损失的vgg层名称及对应权重的列表
STYLE_LOSS_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
# 噪音比率
NOISE = 0.5
# 图片RGB均值
IMAGE_MEAN_VALUE = [128.0, 128.0, 128.0]
# 内容损失权重
ALPHA = 1
# 风格损失权重
BETA = 500
# 训练次数
TRAIN_STEPS = 3000
