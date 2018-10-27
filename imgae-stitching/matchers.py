#-*-coding:utf-8-*-
import cv2
import numpy as np 

class matchers:
	def __init__(self):
		"""
		1. SIFT特征点检测参考：
		https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html#sift-intro
		2. 特征匹配参考：
		https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
		3. Feature Matching + Homography to find Objects参考：
		https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#py-feature-homography
		"""
		self.surf = cv2.xfeatures2d.SURF_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def match(self, i1, i2, direction=None):
		# 参考FLANN_Matcher.py理解
		imageSet1 = self.getSURFFeatures(i1)
		imageSet2 = self.getSURFFeatures(i2)
		if not direction:
			print "Direction : ", direction
		matches = self.flann.knnMatch(
			imageSet2['des'],
			imageSet1['des'],
			k=2
			)
		good = []
		for i , (m, n) in enumerate(matches):
			if m.distance < 0.7*n.distance:
				good.append((m.trainIdx, m.queryIdx))

		if len(good) > 4:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']

			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in good]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in good]
				)

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			return H
		return None

	def getSURFFeatures(self, im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.surf.detectAndCompute(gray, None)
		return {'kp':kp, 'des':des}

def cut(Img):
	H,W,C= Img.shape
	img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

	for h in range(H):
		if sum(img[h,:]) != 0:
			stH = h
			break
	for w in range(W):
		if sum(img[:,w]) != 0:
			stW = w
			break 

	for h in range(stH,H):
		if sum(img[h,:]) == 0:
			rH = h
			break
	for w in range(stW,W):
		if sum(img[:,w]) == 0:
			rW = w
			break 
	return Img[stH:rH,stW:rW,:]