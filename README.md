# pyRecommender

> 基于知识图谱的推荐系统

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

#### 提取知识图谱特征


#### TransE
使用词向量word2vec，对于给定的三元组$(h,r,t)$，需要学习到
$$ \vec{v}_h + \vec{v}_r = \vec{v}_t $$

定义损失函数
$$ loss = \sum_{(h,r,t)\in{S}} \sum_{(h',r',t')\in{S'_{(h,r,t)}}} [\gamma + d(h+r, t) - d(h'+r', t')]  $$

其中S表示的是正确三元组集合，S'是错误三元组的集合。上式中$\gamma$是一个边际参数，让正确三元组和错误三元组的距离最大化。

有一个问题，错误三元组的数量肯定比较多，那么怎么构建合适的错误三元组？

$$ S'_{(h,r,t)} = \{(h',r,t) | h'\in{E}\} \Cup \{(h,r,t') | t'\in{E}\} $$

实际上是通过随机选择实体替换头实体或者尾实体来构成错误的训练三元组。由于用户“正确列表”是确定的，并且“正确列表”的物品数量一般小于总物体数量，那么只要知道了“正确列表”，那么就可以从“总列表-正确列表”中随机寻找一些物品构成“错误列表”即可，让两列表的数量尽可能相等。

### 查询功能
> Prolog语言推理

#### pyDatalog

