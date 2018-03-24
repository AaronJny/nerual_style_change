# -*- coding: utf-8 -*-
# @Time    : 18-3-23 下午12:20
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import tensorflow as tf
import numpy as np
import settings
import scipy.io
import scipy.misc


class Model(object):
    def __init__(self, content_path, style_path):
        self.content = self.loadimg(content_path)  # 加载内容图片
        self.style = self.loadimg(style_path)  # 加载风格图片
        self.random_img = self.get_random_img()  # 生成噪音内容图片
        self.net = self.vggnet()  # 建立vgg网络

    def vggnet(self):
        # 读取预训练的vgg模型
        vgg = scipy.io.loadmat(settings.VGG_MODEL_PATH)
        vgg_layers = vgg['layers'][0]
        net = {}
        # 使用预训练的模型参数构建vgg网络的卷积层和池化层
        # 全连接层不需要
        # 注意，除了input之外，这里参数都为constant，即常量
        # 和平时不同，我们并不训练vgg的参数，它们保持不变
        # 需要进行训练的是input，它即是我们最终生成的图像
        net['input'] = tf.Variable(np.zeros([1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3]), dtype=tf.float32)
        # 参数对应的层数可以参考vgg模型图
        net['conv1_1'] = self.conv_relu(net['input'], self.get_wb(vgg_layers, 0))
        net['conv1_2'] = self.conv_relu(net['conv1_1'], self.get_wb(vgg_layers, 2))
        net['pool1'] = self.pool(net['conv1_2'])
        net['conv2_1'] = self.conv_relu(net['pool1'], self.get_wb(vgg_layers, 5))
        net['conv2_2'] = self.conv_relu(net['conv2_1'], self.get_wb(vgg_layers, 7))
        net['pool2'] = self.pool(net['conv2_2'])
        net['conv3_1'] = self.conv_relu(net['pool2'], self.get_wb(vgg_layers, 10))
        net['conv3_2'] = self.conv_relu(net['conv3_1'], self.get_wb(vgg_layers, 12))
        net['conv3_3'] = self.conv_relu(net['conv3_2'], self.get_wb(vgg_layers, 14))
        net['conv3_4'] = self.conv_relu(net['conv3_3'], self.get_wb(vgg_layers, 16))
        net['pool3'] = self.pool(net['conv3_4'])
        net['conv4_1'] = self.conv_relu(net['pool3'], self.get_wb(vgg_layers, 19))
        net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
        net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
        net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
        net['pool4'] = self.pool(net['conv4_4'])
        net['conv5_1'] = self.conv_relu(net['pool4'], self.get_wb(vgg_layers, 28))
        net['conv5_2'] = self.conv_relu(net['conv5_1'], self.get_wb(vgg_layers, 30))
        net['conv5_3'] = self.conv_relu(net['conv5_2'], self.get_wb(vgg_layers, 32))
        net['conv5_4'] = self.conv_relu(net['conv5_3'], self.get_wb(vgg_layers, 34))
        net['pool5'] = self.pool(net['conv5_4'])
        return net

    def conv_relu(self, input, wb):
        """
        进行先卷积、后relu的运算
        :param input: 输入层
        :param wb: wb[0],wb[1] == w,b
        :return: relu后的结果
        """
        conv = tf.nn.conv2d(input, wb[0], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv + wb[1])
        return relu

    def pool(self, input):
        """
        进行max_pool操作
        :param input: 输入层
        :return: 池化后的结果
        """
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_wb(self, layers, i):
        """
        从预训练好的vgg模型中读取参数
        :param layers: 训练好的vgg模型
        :param i: vgg指定层数
        :return: 该层的w,b
        """
        w = tf.constant(layers[i][0][0][0][0][0])
        bias = layers[i][0][0][0][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return w, b

    def get_random_img(self):
        """
        根据噪音和内容图片，生成一张随机图片
        :return:
        """
        noise_image = np.random.uniform(-20, 20, [1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3])
        random_img = noise_image * settings.NOISE + self.content * (1 - settings.NOISE)
        return random_img

    def loadimg(self, path):
        """
        加载一张图片，将其转化为符合要求的格式
        :param path:
        :return:
        """
        # 读取图片
        image = scipy.misc.imread(path)
        # 重新设定图片大小
        image = scipy.misc.imresize(image, [settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH])
        # 改变数组形状，其实就是把它变成一个batch_size=1的batch
        image = np.reshape(image, (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3))
        # 减去均值，使其数据分布接近0
        image = image - settings.IMAGE_MEAN_VALUE
        return image


if __name__ == '__main__':
    Model(settings.CONTENT_IMAGE, settings.STYLE_IMAGE)
