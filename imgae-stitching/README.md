# Image Stitching(图像拼接)

*首先，想把不同的图像拼接起来，就得先找到两张图片是否含有相同的部分。简单的像素值匹配显然不饿能实现目的，因为可能有不同尺度或者不同的角度。因此需要用到特征点的检测，在两张图片中找到相同的特征点，即实现匹配。再经过的一定的空间变换就有可能实现图像拼接。*

### 1. 特征点的检测
1.1. 何为特征

![特征的定义](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/9890339.jpg)

1.2. 特征的检测算法

![特征的检测算法](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/62085432.jpg)
在opencv3中均有各个算法的实现及介绍，`python`使用教程请转至[opencv3-python教程](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)

1.3. 特征的匹配算法

有以下两种:

+ 暴力匹配(Brute-Force)算法
+ 基于FLANN的匹配法
实现请见[opencv3-python教程](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)

也可参考`FLANN_Matcher.py`

