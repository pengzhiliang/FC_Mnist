# Image Stitching(图像拼接)

*首先，想把不同的图像拼接起来，就得先找到两张图片是否含有相同的部分。简单的像素值匹配显然不饿能实现目的，因为可能有不同尺度或者不同的角度。因此需要用到特征点的检测，在两张图片中找到相同的特征点，即实现匹配。再经过的一定的空间变换就有可能实现图像拼接。*


### 1. 特征点的检测与匹配
1.1. 何为特征

![特征的定义](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/9890339.jpg)

1.2. 特征的检测算法

![特征的检测算法](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/62085432.jpg)

在opencv3中均有各个算法的实现及介绍，`python`使用教程请转至[opencv3-python教程](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)

此任务我们使用的是SURF：
```python
# 读取图片
img = cv2.imread("image.jpg")
# 转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建surf对象，设置阈值，阈值越高检测到的特征就越少
surf = cv2.xfeatures2d.SURF_create(8000)
keypoints, descriptor = surf.detectAndCompute(gray, None)
```
surf阈值为1000时：

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/58951701.jpg)

surf阈值为10000时：

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/87898005.jpg)

1.3. 特征的匹配算法

有以下两种:

+ 暴力匹配(Brute-Force)算法
+ 基于FLANN的匹配法
实现请见[opencv3-python教程](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)
也可参考`FLANN_Matcher.py`

1.4. FLANN匹配

官网介绍：

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/77042281.jpg)

此次任务我们选择FLANN匹配法：
```python
# 定义FLANN的两个参数
# 采用KTreeIndex处理参数索引
FLANN_INDEX_KDTREE = 0
# 深度默认为5
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# 遍历树次数默认为50
search_params = dict(checks=50)   # or pass empty dictionary
# 实例化一个FLANN匹配
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# 丢弃距离大于0.7的匹配对，能减少90%以上的错误
matchesMask = [[0,0] for i in xrange(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
# 画出
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
```
结果为：

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/38503623.jpg)

至此，我们完成了两幅图像中相同特征(或者说相同部分)的检测，那么接下来就是拼接的过程

###  2. 拼接

2.1.  单应性

建议参考[知乎：单应矩阵](https://zhuanlan.zhihu.com/p/35309172)

简单来说，就是单应性是一个条件，该条件表明当两幅图像中的一幅出现投影畸变是，他们还能彼此匹配，而单应矩阵则表明了变换关系。

2.2. 单应矩阵

opencv计算单应矩阵

```python
# 理论上4个点就能算出单应矩阵，但不够鲁棒（噪声），因此点越多越好
MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # 通过一些相匹配的特征点就能得到两幅图像的单应矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
```

如：

```
Homography is :  
[[  8.82849004e-01   6.29424368e-02   1.73974487e+02]
 [ -8.15172258e-02   9.72706270e-01  -1.20308509e+01]
 [ -2.64191566e-04   4.09411969e-05   1.00000000e+00]]

```

2.3. 变形拼接

一旦我们获得了单应矩阵，也就获得了第一幅图像在第二副图像的视角是应该是呈现什么样子，比如：

图片1

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/8255881.jpg)

图片2

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/13483359.jpg)

第一幅图像在 第二副图像的视角 下呈现：

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/14360779.jpg)

根据匹配得到的坐标偏移就可以拼接得到：

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/69892296.jpg)

同理，三张图片拼接：

![](http://p1i1k4m2v.bkt.gdipper.com/18-10-27/59494248.jpg)

### 3. Refenerce
参考Github:[Python-Multiple-Image-Stitching](https://github.com/kushalvyas/Python-Multiple-Image-Stitching)