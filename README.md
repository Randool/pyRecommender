# pyRecommender

> 基于知识图谱的推荐系统

参考了[Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation](https://arxiv.org/abs/1901.08907)，是该论文的PyTorch版实现，具体用法还请移步原作者的[git](https://github.com/hwwang55/MKR)

## TODO

### 基于知识图谱的推荐功能
- [ ] 提取知识图谱特征
- [ ] TransE
- [ ] 基于模糊数学的推荐系统
- [ ] 交替学习MKR

### 基于知识图谱的查询功能
- [ ] pyDatalog


---
## 数据格式

数据用json形式组织。数据包括：用户信息、书籍信息、电影信息、音乐信息

### 用户信息
```
{
    "location":                                         // 常驻地
    "username":                                         // 用户名
    "join_time":                                        // 加入时间
    "book_do":       [(book_name, book_url), ...]       // 在看的书
    "book_wish":     [(book_name, book_url), ...]       // 想看的书
    "book_collect":  [(book_name, book_url), ...]       // 看过的书
    "movie_do":      [(movie_name, movie_url), ...]     // 在看的电影
    "movie_wish":    [(movie_name, movie_url), ...]     // 想看的电影
    "movie_collect": [(movie_name, movie_url), ...]     // 看过的电影
    "music_do":      [(music_name, music_url), ...]     // 在听的音乐
    "music_wish":    [(music_name, music_url), ...]     // 想听的音乐
    "music_collect": [(music_name, music_url), ...]     // 听过的音乐
    "attation_url":  [url1, url2, url3, ...]            // 所关注的用户主页url
}
```

### 书籍信息
> 作者、出版社、出版时间、页数、价格、ISBN编号、评分
```
{
    "author":       "作者"
    "publisher":    "出版社"
    "time":         "出版时间"
    "pages":        "页数"
    "price":        "价格"
    "ISBN":         "ISBN编号"
    "score":        "评分"
}
```

### 电影信息
> 导演、编剧、演员、类型、制片国家/地区、语言、上映时间、片长、IMDb链接、评分
```
{
    "director":     "导演"
    "scriptwriter": "编剧"
    "actors":       [act1, act2, ...]
    "type":         [t1, t2, t3, ...]
    "country":      "制片国家/地区"
    "language":     "语言"
    "time":         "上映时间"
    "length":       "片长"
    "IMDb":         "IMDb链接"
    "score":        "评分"
}
```

### 音乐信息
> 歌手、流派、专辑类型、介质、出版者、评分
```
{
    "singer":   "歌手"
    "genre":    "流派"
    "type":     [t1, t2, t3, ...]
    "media":    "介质"
    "author":   "出版者"
    "score":    "评分"
}
```

---
## 算法思路

### 推荐功能
> 特征工程

#### MKR
由于推荐系统(RS)中的物品和知识图谱(KG)中的实体存在重合，因此可以采用多任务学习的框架，讲推荐系统和知识图谱视为两个分离但是相关的任务，进行 __交替学习__。

推荐部分的输入是用户(user_feature)和物品(item_feature)的特征表示，点击率的预估值(predicted_probability)作为输出。知识图谱特征学习部分使用的是三元组的头节点(head)和关系(relation)作为输入，预测的尾节点(tail)作为输出。

推荐系统和知识图谱两者的纽带就是“交叉特征共享单元”(cross-feature-sharing unit)。该单元的目的是让两个模块交换信息，据说这样做是为了让两者获取更多的信息，弥补自身信息稀疏性。

由于该模型存在两个模块的交叉，所以训练的时候首先固定推荐系统模块，训练知识图谱的参数，然后固定知识图谱特征学习模块的参数，训练推荐系统的参数。

推荐系统的训练目的是预测用户点击率，相当于一个二分类问题，使用L2正则项。
```Python
# RS
self.base_loss_rs = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores)
)
self.l2_loss_rs = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
for var in self.vars_rs:
    self.l2_loss_rs += tf.nn.l2_loss(var)
self.loss_rs = self.base_loss_rs + self.l2_loss_rs * args.l2_weight
```

知识图谱特征学习是让预测的tail向量和真实tail向量相近，即目标
$$ \vec{v}_{head} + \vec{v}_{relation} = \vec{v}_{tail} $$
因此首先计算预测的tail和真实tail的内积，经过sigmoid平滑后取相反数，最后加上l2正则项。
```Python
# KGE
self.base_loss_kge = -self.scores_kge
self.l2_loss_kge = tf.nn.l2_loss(self.head_embeddings) + tf.nn.l2_loss(self.tail_embeddings)
for var in self.vars_kge:
    self.l2_loss_kge += tf.nn.l2_loss(var)
self.loss_kge = self.base_loss_kge + self.l2_loss_kge * args.l2_weight
```

实际上是通过随机选择实体替换头实体或者尾实体来构成错误的训练三元组。由于用户“正确列表”是确定的，并且“正确列表”的物品数量一般小于总物体数量，那么只要知道了“正确列表”，那么就可以从“总列表-正确列表”中随机寻找一些物品构成“错误列表”即可，让两列表的数量尽可能相等。

损失函数
$$\text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right) = -x[class] + \log\left(\sum_j \exp(x[j])\right)$$

### 查询功能
> Prolog语言推理

#### pyDatalog

