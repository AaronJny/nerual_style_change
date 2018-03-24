# -*- coding: utf-8 -*-
# @Time    : 18-3-23 下午12:22
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import tensorflow as tf
import settings
import models
import numpy as np
import scipy.misc


def loss(sess, model):
    """
    定义模型的损失函数
    :param sess: tf session
    :param model: 神经网络模型
    :return: 内容损失和风格损失的加权和损失
    """
    # 先计算内容损失函数
    # 获取定义内容损失的vgg层名称列表及权重
    content_layers = settings.CONTENT_LOSS_LAYERS
    # 将内容图片作为输入，方便后面提取内容图片在各层中的特征矩阵
    sess.run(tf.assign(model.net['input'], model.content))
    # 内容损失累加量
    content_loss = 0.0
    # 逐个取出衡量内容损失的vgg层名称及对应权重
    for layer_name, weight in content_layers:
        # 提取内容图片在layer_name层中的特征矩阵
        p = sess.run(model.net[layer_name])
        # 提取噪音图片在layer_name层中的特征矩阵
        x = model.net[layer_name]
        # 长x宽
        M = p.shape[1] * p.shape[2]
        # 信道数
        N = p.shape[3]
        # 根据公式计算损失，并进行累加
        content_loss += (1.0 / (2 * M * N)) * tf.reduce_sum(tf.pow(p - x, 2)) * weight
    # 将损失对层数取平均
    content_loss /= len(content_layers)

    # 再计算风格损失函数
    style_layers = settings.STYLE_LOSS_LAYERS
    # 将风格图片作为输入，方便后面提取风格图片在各层中的特征矩阵
    sess.run(tf.assign(model.net['input'], model.style))
    # 风格损失累加量
    style_loss = 0.0
    # 逐个取出衡量风格损失的vgg层名称及对应权重
    for layer_name, weight in style_layers:
        # 提取风格图片在layer_name层中的特征矩阵
        a = sess.run(model.net[layer_name])
        # 提取噪音图片在layer_name层中的特征矩阵
        x = model.net[layer_name]
        # 长x宽
        M = a.shape[1] * a.shape[2]
        # 信道数
        N = a.shape[3]
        # 求风格图片特征的gram矩阵
        A = gram(a, M, N)
        # 求噪音图片特征的gram矩阵
        G = gram(x, M, N)
        # 根据公式计算损失，并进行累加
        style_loss += (1.0 / (4 * M * M * N * N)) * tf.reduce_sum(tf.pow(G - A, 2)) * weight
    # 将损失对层数取平均
    style_loss /= len(style_layers)
    # 将内容损失和风格损失加权求和，构成总损失函数
    loss = settings.ALPHA * content_loss + settings.BETA * style_loss

    return loss


def gram(x, size, deep):
    """
    创建给定矩阵的格莱姆矩阵，用来衡量风格
    :param x:给定矩阵
    :param size:矩阵的行数与列数的乘积
    :param deep:矩阵信道数
    :return:格莱姆矩阵
    """
    # 改变shape为（size,deep）
    x = tf.reshape(x, (size, deep))
    # 求xTx
    g = tf.matmul(tf.transpose(x), x)
    return g


def train():
    # 创建一个模型
    model = models.Model(settings.CONTENT_IMAGE, settings.STYLE_IMAGE)
    # 创建session
    with tf.Session() as sess:
        # 全局初始化
        sess.run(tf.global_variables_initializer())
        # 定义损失函数
        cost = loss(sess, model)
        # 创建优化器
        optimizer = tf.train.AdamOptimizer(1.0).minimize(cost)
        # 再初始化一次（主要针对于第一次初始化后又定义的运算，不然可能会报错）
        sess.run(tf.global_variables_initializer())
        # 使用噪声图片进行训练
        sess.run(tf.assign(model.net['input'], model.random_img))
        # 迭代指定次数
        for step in range(settings.TRAIN_STEPS):
            # 进行一次反向传播
            sess.run(optimizer)
            # 每隔一定次数，输出一下进度，并保存当前训练结果
            if step % 50 == 0:
                print 'step {} is down.'.format(step)
                # 取出input的内容，这是生成的图片
                img = sess.run(model.net['input'])
                # 训练过程是减去均值的，这里要加上
                img += settings.IMAGE_MEAN_VALUE
                # 这里是一个batch_size=1的batch，所以img[0]才是图片内容
                img = img[0]
                # 将像素值限定在0-255，并转为整型
                img = np.clip(img, 0, 255).astype(np.uint8)
                # 保存图片
                scipy.misc.imsave('{}-{}.jpg'.format(settings.OUTPUT_IMAGE,step), img)
        # 保存最终训练结果
        img = sess.run(model.net['input'])
        img += settings.IMAGE_MEAN_VALUE
        img = img[0]
        img = np.clip(img, 0, 255).astype(np.uint8)
        scipy.misc.imsave('{}.jpg'.format(settings.OUTPUT_IMAGE), img)


if __name__ == '__main__':
    train()
