# 文本分类前期调研

[TOC]

## 一、文本分类任务

文本分类是指按照一定的分类体系或标准使用机器对文本集进行自动分类标记的过程。从流程上可分为**文本预处理**、**文本表示**、**特征提取**、**分类器训练**等过程, 其中最关键的步骤是特征提取和分类器训练。

数学方法描述：在有监督学习中,数据集用 $X$ 表示,$X = {(x^1 , y^1 ), · · · , (x^N , y^N )}$,N 为 样本个数。$y^i$ 为样本实例 $x^i$ 的类别标记。其中$x^i$的为一组文本，$y^i$可以是一组标签如词性，也可以是一个标签如文本的类别
$$
x^i = (w_1...w_t...w_T)\\ 
y^i=(p_1...p_t...p_T)
$$
使用机器学习方法－即找到这样一个映射f， 使得$x^{i}->y^{i}$	

## 二、文本分类算法

### 1、传统机器学习

- **knn算法**（k-NearestNeighbor算法）

  每个样本都可以用它最接近的k个邻居来代表，没有训练阶段，数据集事先已有了分类和特征值，待收到新样本后直接进行处理。

- **朴素贝叶斯算法**

  在统计数据的基础上，依据**条件概率**公式，计算当前特征的样本属于某个分类的概率，选最大的概率分类

- **SVM（支持向量机）**

- **随机森林(RF)**

  随机森林(RF)指的是利用多棵决策树对样本进行训练并预测的一种分类器

### 2、深度学习方法

- **fastText**

  把一篇文章中所有的词向量（还可以加上N-gram向量）直接相加求均值，然后过一个单层神经网络来得出最后的分类结果。对于复杂的文本分类任务来说丢失了太多的信息

- **TextCNN**

- **TextRNN**

  单向RNN结构、双向RNN(Bidirection-RNN)、LSTM(长短时记忆网络)、LSTM的变体GRU网络

- **TextRCNN**

- **HAN**

- **DPCNN**

## 三、数据集

### 1、学术论文常用数据集

知网中收录多数有关文本分类的论文均大多使用在下列数据集的实验效果作为成果，**未发现各个算法在这些数据集上得分榜单**。

**（1）英文数据集**

- [**20 newsgroups**](http://qwone.com/~jason/20Newsgroups/)

  20newsgroups数据集是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合。

- [**Reuters (RCV*) Corpuses**](https://keras.io/zh/datasets/#imdb)

  RCV1为路透社新闻数据集, 是英文文本主题分类很常用的实验数据。可使用keras库导入数据集。

- [**IMDB**](https://keras.io/zh/datasets/#imdb)

  IMDB为互联网电影资料库, 是英文情感分析的标准实验数据, 其任务是判断电影评论是正面 (positive) 还是负面 (negative) 的。可使用keras库导入数据集。

- [**AG's corpus of news articles**](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

  AG是由ComeToMyHead超过一年的努力，从2000多不同的新闻来源搜集的超过1百万的新闻文章。

- [**Yahoo**](https://webscope.sandbox.yahoo.com/index.php)

- [**yelp**](https://www.yelp.com/dataset)

- [**DEpebian**](https://wiki.dbpedia.org/develop/datasets/dbpedia-version-2016-10)

- [**WOS数据集**](https://data.mendeley.com/datasets/9rw3vkcfy4/6)

  WOS-11967- 该数据集包含11,967个文档，包含35个类别，其中包括7个父类别。

  WOS-46985- 该数据集包含46,985个文档，包含134个类别，其中包括7个父类别。

  WOS- 5736-该数据集包含5,736个文档，其中11个类别包括3个父类别。

**（2）中文数据集**

- [**复旦大学文本分类语料库**](https://www.kesci.com/home/dataset/5d3a9c86cf76a600360edd04)

  未找到官方说明文档，但可以找到下载链接。

- [**THUCNews**](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)

  THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。

- [**搜狗实验室新闻数据**](http://www.sogou.com/labs/resource/list_news.php)

  来自搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据，提供URL和正文信息。

### 2、其它数据集

其它比赛中或者个人整理所得文本分类数据集。

- [**“达观杯”文本智能处理挑战赛**](<https://www.pkbigdata.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html>)

  此次比赛，达观数据提供了一批长文本数据和分类信息，希望选手动用自己的智慧，结合当下最先进的NLP和人工智能技术，深入分析文本内在结构和语义信息，构建文本分类模型，实现精准分类。目前已结束报名，但仍然可以报名参赛从而下载测试数据。数据集无数量等详细信息。

  **可以看到其他队伍得分与排名**。

- [**2017 知乎看山杯机器学习挑战赛**](<https://biendata.com/competition/zhihu/>)

  根据知乎给出的问题及话题标签的绑定关系的训练数据，训练出对未标注数据自动标注的模型。标注数据中包含 300 万个问题，每个问题有 1 个或多个标签，共计1999 个标签。每个标签对应知乎上的一个「话题」，话题之间存在父子关系。更多信息可以点击链接进入官网查看。目前已结束报名，但仍然可以报名参赛从而下载测试数据。

  **可以看到其他队伍得分与排名。**

- [**Sentiment Analysis on Movie Reviews**](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

  电影评论的情感分析，是 github 上 fudanNLP入门项目文本分类推荐使用的数据集。

  **可以看到其他队伍得分与排名。**

- [**今日头条新闻数据集（非官方）**](https://github.com/fate233/toutiao-text-classfication-dataset)

  共382688条，分布于15个分类中。采集时间：2018年05月。

  **README文档中有未知算法的实验结果。可进行比对**。

## 四、各个算法在数据集表现

**部分**前沿学术论文使用算法及其在部分**常用数据集**上的表现

|                            MODELS                            |  AGNews   |   Yahoo   |  Yelp P.  |  Yelp F.  |  DBPedia  |
| :----------------------------------------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| [LEAM](https://www.aclweb.org/anthology/P18-1216#page=11&zoom=100,0,505) |   92.45   | **77.42** |   95.31   |   64.09   |   99.02   |
|     [Bi-BloSAN](https://openreview.net/pdf?id=H1cWzoxA-)     |   93.32   |   76.28   |   94.56   |   62.13   |   98.77   |
|  [Very Deep CNN](https://www.aclweb.org/anthology/E17-1104)  |   91.27   |   73.43   |   95.72   |   64.26   |   98.71   |
|     [ULMFiT](https://www.aclweb.org/anthology/P18-1031)      | **94.99** |     -     |     -     | **70.02** | **99.20** |
| [DPCNN+ unsupervised embed](https://aclweb.org/anthology/P17-1052) |   93.13   |   76.10   | **97.36** |   69.42   |   99.12   |

​						Test Accuracy on document classification tasks, in percentage. 

最近三年(17--19)有关文本分类的ACL学术论文：

[Joint Embedding of Words and Labels for Text Classification](https://www.aclweb.org/anthology/P18-1216)

[Universal Language Model Fine-tuning for Text Classification](https://www.aclweb.org/anthology/P18-1031)

[Cross-domain Text Classification with Multiple Domains and Disparate Label Sets](https://www.aclweb.org/anthology/P16-1155)

[On-device Structured and Context Partitioned Projection Networks](https://www.aclweb.org/anthology/P19-1368)

[Variational Pretraining for Semi-supervised Text Classification](https://www.aclweb.org/anthology/P19-1590)

[Incorporating Priors with Feature Attribution on Text Classification](https://www.aclweb.org/anthology/P19-1631)

[Hierarchical Transfer Learning for Multi-label Text Classification](https://www.aclweb.org/anthology/P19-1633)

[Towards Unsupervised Text Classification Leveraging Experts and Word Embeddings](https://www.aclweb.org/anthology/P19-1036)

[Towards Scalable and Reliable Capsule Networks for Challenging NLP Applications](https://www.aclweb.org/anthology/P19-1150)

[Towards Integration of Statistical Hypothesis Tests into Deep Neural Networks](https://www.aclweb.org/anthology/P19-1557#page=6&zoom=100,0,428)

[Generative and Discriminative Text Classification with Recurrent Neural Networks](http://pdfs.semanticscholar.org/54ed/f1c695972f2af800d7062ee7657217a4f632.pdf)

## 五、其他

在调研过程中发现的比较好的博客等学习资料，收集起来可供未来参考使用。

- [优质的中文 NLP 工具和资源集合项目——funNLP](<https://blog.csdn.net/fkyyly/article/details/88101838>)
- [有关THUCNews数据集的介绍](https://www.jianshu.com/p/687ecdd3f840)
- [文本分类概述](<https://blog.csdn.net/u014248127/article/details/80774668>)
- [部分文本分类算法集锦](<https://www.kesci.com/home/project/5be7e948954d6e0010632ef2>)
- [文本分类方面一些文章](<http://www.52nlp.cn/category/text-classification>)
- [python中文文本分类](<https://blog.csdn.net/github_36326955/article/details/54891204>) 
- [ACL会议paper](https://aclweb.org/anthology/venues/acl/)