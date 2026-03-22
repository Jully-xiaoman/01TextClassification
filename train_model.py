import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
# pandas是表格数据结构，很像excel表格

# 1.读取txt文件并转换为df；
def read_txt_to_df(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # strip():去除首尾巴的换行和空格符
            # split("_!_"):按括号内字符分割，变成一个列表
            parts = line.strip().split("_!_")
            if len(parts) == 5:
                rows.append(parts)

    # columns是指给每一列取个列名
    df = pd.DataFrame(rows, columns=[
        "news_id", "category_code", "category_name", "title", "keywords"
    ])
    return df

# 2.读取train_df,dev_df,test_df;
train_df = read_txt_to_df(r"./RawData/train_3k.txt")
dev_df = read_txt_to_df(r"./RawData/dev_1k.txt")
test_df = read_txt_to_df(r"./RawData/test_1k.txt")

# print(train_df.head())
# print(dev_df.head())
# print(test_df.head())

# 3.建立标签映射;
label_names = sorted(train_df["category_name"].unique())
# 字典形式
label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for label, idx in label2id.items()}

# print("标签映射：")
# print(label2id)
# print("类别数量：", len(label2id))

# 4. 提取 texts 和 labels
train_texts = train_df["title"].tolist()
# train_texts = train_df["title"] 后是pandas的series，带有行号。
# 必须要转换为普通列表模型才能接收。
train_labels = train_df["category_name"].map(label2id).tolist()

dev_texts = dev_df["title"].tolist()
dev_labels = dev_df["category_name"].map(label2id).tolist()

test_texts = test_df["title"].tolist()
test_labels = test_df["category_name"].map(label2id).tolist()

# 输出查看
# print("label2id:", label2id)
# print("train_texts前3条:", train_texts[:3])
# print("train_labels前3条:", train_labels[:3])

# 5.加载 tokenizer
# 把 bert-base-chinese 这个模型配套的 tokenizer 加载出来，并命名为 tokenizer。
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 6.写Dataset类：返回单条样本的处理结果
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # 输出数据集的size
    def __len__(self):
        return len(self.texts)

    # 按照编号获取数据的标准格式
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 每次处理一条数据
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 单条数据处理后的固定输出格式
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length=32)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_dataset = NewsDataset(dev_texts, dev_labels, tokenizer, max_length=32)
dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length=32)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 7.加载模型，准备优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=15
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 8.定义评估函数
def evaluate(model, dataloader, device):
    model.eval()  # 切换到评估模式，关闭dropout

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # 不计算梯度（更快更省内存）
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits  # (batch_size, num_labels)

            # 取最大值的类别作为预测结果
            # 按行返回最大值的下标，dim=1表示按照行
            preds = torch.argmax(logits, dim=1)

            # 统计正确数量
            # .sum()：等于1的地方全部加起来
            total_correct += (preds == labels).sum().item()
            # 统计张量第0维的长度
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(dataloader)

    return accuracy, avg_loss

# 9.模型训练循环
best_dev_acc = 0.0
for epoch in range(50):
    # 切换到训练模式，Dropout生效。
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 平均训练batchloss
    avg_train_loss = total_loss / len(train_dataloader)
    # 一个Epoch结束后在验证集上验证一次
    dev_acc,dev_loss = evaluate(model, dev_dataloader, device)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
    print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), "best_model.pt")
        print("最佳模型已保存！")

# 10.测试
model.load_state_dict(torch.load("best_model.pt"))
test_loss, test_acc = evaluate(model, test_dataloader, device)
