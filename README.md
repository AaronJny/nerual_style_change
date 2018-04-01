# 使用VGG19迁移学习实现图像风格迁移

这是一个使用预训练的VGG19网络完成图片风格迁移的项目，使用的语言为python，框架为tensorflow。

给定一张风格图片A和内容图片B，能够生成具备A图片风格和B图片内容的图片C。

下面给出两个示例，风格图片都使用梵高的星夜：

![风格图片](https://raw.githubusercontent.com/AaronJny/nerual_style_change/master/sample/input_style_1.jpg)

**示例1：**

网络上找到的一张风景图片。

内容图片：

![内容图片1](https://raw.githubusercontent.com/AaronJny/nerual_style_change/master/sample/input_content_1.jpg)

生成图片：

![生成图片1](https://raw.githubusercontent.com/AaronJny/nerual_style_change/master/sample/output_1.jpg)


**示例2：**

嗷嗷嗷，狼人嚎叫～

内容图片：

![内容图片2](https://raw.githubusercontent.com/AaronJny/nerual_style_change/master/sample/input_content_2.jpg)

生成图片：

![生成图片2](https://raw.githubusercontent.com/AaronJny/nerual_style_change/master/sample/output_2.jpg)


更多详情请移步博客[https://blog.csdn.net/aaronjny/article/details/79681080](https://blog.csdn.net/aaronjny/article/details/79681080)

----------------------

## 快速开始

**1.下载预训练的vgg网络，并放入到项目的根目录中**

模型有500M+，故没有放到GitHub上，有需要请自行下载。

下载地址：[http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)

**2.选定风格图片和内容图片，放入项目根目录下的`images`文件夹中**

在项目根目录下的`images`文件夹中，有两张图片，分别为`content.jpg`和`style.jpg`，即内容图片和风格图片。

如果只是使用默认图片测试模型，这里可以不做任何操作。

如果要测试自定义的图片，请使用自定义的内容图片`和/或`风格图片替换该目录下的内容图片`和/或`风格图片，请保持命名与默认一致，或者在`settings.py`中修改路径及名称。

**3.开始生成图片**

运行项目中的`train.py`文件，进行训练。在训练过程中，程序会定期提示进度，并保存过程图片。

当训练结束后，保存最终生成图片。

所有生成的图片均保存在项目根目录下`output`文件夹中。

**4.更多设置**

在`settings.py`文件中存在多种配置项，可根据需求进行配置。