# distracted_drivers

## 侦测走神司机
(Exploration on going)

## 项目背景
侦测走神司机项目（[Distracted Driver Detection][1]）来自 Kaggle，由[State Farm][2]提出。State Farm 使用车内摄像头，对驾驶员驾车时的状态进行了视频记录并截取了视频中的图片作为训练数据，要求通过该数据建立识别出司机是在专注驾驶还是在走神的方法。改项目具有比较大的现实意义，基于该图像识别的车载系统可以提示司机进行更安全、更专注的驾驶，提高行车效率，降低事故率。

该项目是一个典型的图像识别分类问题，所以可以考虑通过深度卷积神经网络算法来解决。著名的卷积神经网络架构[LeNet][3]最早于1998年由Yann LeCun *et al.*提出，在2012年，Alex Krizhevsky *et al.*通过对卷积神经网络进行改进，提出了[AlexNet][4]，应用在ImageNet图像数据上，极大的提升了图像识别的准确率。从这时候，很多成功的深度卷积神经网络架构都被提出，例如著名的[VGGNet][5]、[GoogLeNet][6]。基于这些架构的成功，更多的方法不断被提出，包括[ResNet][7]，[Inception v3][8]，[Inception v4][9]， [Xception][10] ，[ResNeXt][11]等等。这些架构和方法使得卷积神经网络在图像识别问题上超过了其他方法，并提供了大量可以借鉴的想法和经验，是解决该项目的首选算法之一。

我选择这个项目的原因有两个：第一，该项目具有比较强的现实意义，可以实际地帮助解决安全行驶的问题。第二，希望可以通过初次上手实际的Kaggle项目，学习了解更多深度神经网络的知识以及实践方法，为以后使用深度神经网络解决其他问题打下基础。

结合Udacity提供的初步[提示资料][12]，下面从问题描述、输入数据、解决办法、基准模型、评估指标以及设计大纲这几个部分对问题进行阐述。

[1]: https://www.kaggle.com/c/state-farm-distracted-driver-detection
[2]: https://www.statefarm.com/
[3]: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
[4]: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
[5]: https://arxiv.org/abs/1409.1556
[6]: https://arxiv.org/abs/1409.4842
[7]: https://arxiv.org/abs/1512.03385
[8]: https://arxiv.org/abs/1512.00567
[9]: https://arxiv.org/abs/1602.07261
[10]: https://arxiv.org/abs/1610.02357
[11]: https://arxiv.org/abs/1611.05431
[12]: https://github.com/nd009/capstone/tree/master/distracted_driver_detection

## 问题描述
在该项目中，我们会得到一系列司机驾驶时拍摄的视频的截图，截图中展示了司机在驾驶时在做什么，例如发短信、吃东西、打电话等等。项目的目标是预测司机在图片中可能在干什么。该问题属于对于图片的分类问题，这里采用卷积神经网络为主要的算法。作为深度学习的初学者，重头构造自己的卷积神经网络架构并不可取，可以选择借鉴已有的成功网络架构，并尝试首先带入其他经典图像集预先训练好的权重，然后进行finetune。除了使用神经网络进行计算以外，项目还要求对神经网络做出判断的依据进行可视化解释，这就需要对神经网络每一层关注的图像区域进行可视化分析。

## 输入数据
输入数据为按十种状态分类的彩色图片，状态列表为：
- c0：安全驾驶
- c1：右手打字
- c2：右手打电话
- c3：左手打字
- c4：左右打电话
- c5：调收音机
- c6：喝饮料
- c7：拿后边的东西
- c8：整理头发和化妆
- c9：和其他乘客说话

训练数据分别保存在以状态编号命名的文件夹内，这里训练集的主要特点为：
- 图像数据为视频截图，同一位司机在开车中的不同状态都被截图放入了不同的对应文件夹内，故不同状态下的同一司机截图具有一定的时间相关性。
- 不同司机的姓名和对应的文件名保存在.csv文件中。
- test集中的司机跟训练数据中的不同。
- 图片中司机一般位于图片的右侧。
- 测试集的图片数相比训练集来说比较多。
- 分类比较多，且图片场景相对比较类似。


## 解决办法
-  准备进行项目的软硬件环境，使用OpenCV进行图像的读取和处理，TensorFlow以及Keras作为深度学习框架，使用[亚马逊云][19]来进行计算。
-  对数据进行探索，了解该项目数据特征，对数据进行分割处理。可以借鉴[ZFTurbo][13]共享的基础代码，作为开始，实现图片的读取、归类、处理、train-validation分割。
- 学习借鉴已经成熟的卷积深度学习网络结构和方法（如VGG，Xception等），首先采用已有模型，以及预训练权重，transfer learning+开放部分层进行finetune的思路，先跑起来，实现初步的训练，观察结果。可以的话比较几种经典不同方法的效果，发表的模型代码可以参考[deep learning models for keras][15]。模型的特点和方法可以阅读相关论文，以及参阅网上的资料：
	- [DiamonJoy的博客文章][16]
	- [Coursera的CNN课程][17]
	- [CS231n课程][18]
- 根据对训练结果主要先对valid logloss结果进行分析，根据数据特征以及他人的经验，训练数据较少而且图像之间比较类似，容易出现过拟合的现象。根据出现的过拟合或者欠拟合来采用相应的方法进行处理。
- 该数据的训练集数目较少，容易出现过拟合，可能需要尝试对数据进行增强处理，以及根据模型表现学习采用模型融合的方法提高准确度。一些可以参考的文献如下：
	- [Understanding data augmentation for classification: when to warp?][Sebastien]
	- [The Effectiveness of Data Augmentation in Image Classification using DeepLearning][Luis]
- 为了提高最终结果，模型融合也是一种在Kaggle竞赛中经常使用的房，这里将几种模型的预测结果进行融合，采用投票的方式，可以进一步地提高预测的准确度。
	- [模型融合方法][20]
	- [Kaggel ensembling guide][21]
- 最后参考Kaggle上比赛者的一些[经验分享][14]以及Udacity的通关群来讨论借鉴如何进一步提高模型效果。


[13]: https://www.kaggle.com/zfturbo/keras-sample
[14]: https://www.kaggle.com/c/state-farm-distracted-driver-detection/discussion
[15]: https://github.com/fchollet/deep-learning-models
[16]: http://my.csdn.net/u013751160
[17]: https://www.coursera.org/learn/convolutional-neural-networks/home/week/4
[18]: http://cs231n.github.io/convolutional-networks/
[19]: https://zhuanlan.zhihu.com/p/25066187
[20]: http://blog.csdn.net/sinat_29819401/article/details/71191219
[21]: https://mlwave.com/kaggle-ensembling-guide/
[Sebastien]: https://arxiv.org/abs/1609.08764
[Luis]: https://arxiv.org/abs/1712.04621


## 基准模型
这里打算使用[jiao Dong][jiaoDong]在Kaggle上分享的VGG_16方案作为比较的基准模型，他用预训练过的权重为基础直接对VGG_16进行训练，达到了LB=0.238左右的效果。

[jiaoDong]: https://www.kaggle.com/jiaodong/vgg-16-pretrained-loss-0-23800

## 评估指标
根据题目的[评估要求][Evaluation]，这里的评估指标为test集的multi-class logarithmic loss，其计算公式如下：

[equation 1](http://latex.codecogs.com/gif.latex?logloss%3D-%5Cfrac%7B1%7D%7BN%7D%5Csum%5Climits%5E%7BN%7D_%7Bi%3D1%7D%5Csum%5Climits%5E%7BM%7D_%7Bj%3D1%7Dy_%7Bij%7D%5Clog%28p_%7Bij%7D%29)  

其中N是test集中图片的数目，M是图片的标注种类的数目，y_{ij}在图片i属于第j个标注时等于1，其他情况下等于0，p_{ij}表示第i张图片属于标注j的概率。
为了防止\log中极端值的出现，对于p_{ij}可以做如下处理

[equation 2](http://latex.codecogs.com/gif.latex?%5Cmax%28%5Cmin%28p_%7Bij%7D%2C1-10%5E%7B-15%7D%29%2C10%5E%7B-15%7D%29)

[Evaluation]: https://www.kaggle.com/c/state-farm-distracted-driver-detection#evaluation

## 设计大纲
整个项目的初步设计流程如下图所示：

```flow
st=>start: 项目开始
e=>end: 项目结束
op1=>operation: 环境准备
op2=>operation: 数据集分割
op3=>operation: 图像预处理
op4=>operation: 模型选择、Pre-train+finetune，调试梯度下降优化方法、超参数调参、数据增强、模型集成
cond1=>condition: valid-logloss是否达到或超过基准模型？
op5=>operation: 计算test-logloss
op6=>operation: Class Activation Mapping 
op7=>operation: 完成报告，提交审阅
cond2=>condition: 提交项目是否满足要求？
st->op1->op2->op3->op4->cond1
cond1(yes)->op5->op6->cond2
cond1(no)->op4
cond2(no)->op2
cond2(yes)->e
```

- **环境准备**：项目拟在aws云的p2.xlarge实例上，在jupyter-notebook中实现，使用tensorflow-gpu。
- **数据集分割**：原数据集中没有分类训练集和验证集，故需要自行分类，这里需要实现的点有两个：一是将图片以每个司机的名字来分类，保证训练集和验证集中不同时出现同一位司机的不同照片，这样做是因为同一位司机的不同照片其实是取自于同一视频，若这些照片即出现在训练集又出现在验证集，则会造成过拟合。二是实现CV，使得验证集的loss结果稳定可信。在最终计算test logloss之前，仅使用valid logloss对模型进行评估，只有在对模型结果又充分信心后，最优用test logloss进行评价。
- **图像预处理**：这里打算首先使用已有的网络模型，由于图像的预处理方式对于模型计算结果的影响在带入神经网络之前，图片的预处理可以考虑所使用模型论文中的处理方法。例如Vgg需要对各个通道除以相应的均值，Xception则需要归一化到$[-1,1]$区间。
- **模型选择 pre-train+finetune**：考虑到p2.xlarge实例的资源规模，使用vgg-16这种参数很多的模型进行fine-tune训练很可能资源不足，不能开放很多层。故考虑先使用参数较少的、综合了很多新结构的模型，如Xception。如下图所示，这里打算将Xception的Exit Flow下GlobalAveragePooling之后的全连接层修改为自己的形式，首先对于全连接层之前的层读入IMAGENET的预训练权重，并仅训练全连接层。根据训练的结果再试验开放部分Middle Flow层进行训练。
![Alt text](./1516459234426.png)
	- **梯度下降优化方法**：尝试SGD，Adam，RMSprop等梯度下降方法，比较结果的不同。
	-  **超参数调试**：根据结果对学习率、学习率衰减方式、batch size、全连接层的结构进行调整。由于可以预见会出现过拟合的问题，所以需要尝试加入Dropout，正则化，以及减少全连接层复杂度的方法。
	-  **数据增强**：利用keras的[imagedatagenerator]以及自定义的图像变换函数，对数据进行增强，这里打算在训练时，对于每个epoch都使用原始数据+重新进行的数据增强来训练。增强的方法打算使用随机的镜像，剪切，旋转，以及截取右侧图像的方式进行。
	-   **模型集成**：如果需要进一步降低logloss，进行多模型集成的尝试，以进一步提高结果。
- **可视化**：根据[Class Activation Mapping][map]，来对CNN层做出判断时看重的图像元素进行可视化。


[map]: http://cnnlocalization.csail.mit.edu/
[imagedatagenerator]: https://keras.io/preprocessing/image/

