# 📰 Chinese News Classification (BERT)

## 1. 任务介绍

本项目基于 `bert-base-chinese` 实现一个简单的中文新闻标题分类任务。  
通过对新闻标题进行编码，并使用 BERT 进行分类预测，实现多类别文本分类。

---

## 📂 数据集获取

* 🔗 [百度网盘下载](https://pan.baidu.com/s/10XRGQAIKGDI5eWLjmaB9Xg?pwd=1111)
* 🔑 提取码：`1111`

下载后将数据放入：
RawData/
├── train_3k.txt
├── dev_1k.txt
├── test_1k.txt

---

## 🚀 How to run

### 1. 安装依赖
```bash
pip install torch transformers pandas
```

### 2. 准备数据

将数据集放入 RawData/ 文件夹。

### 3. 运行训练
python train_model.py

