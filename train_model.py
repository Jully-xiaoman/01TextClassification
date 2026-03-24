import torch
import json
from transformers import BertForSequenceClassification
from data_module import create_datasets_and_loaders

with open("args.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# 1.数据接口
train_dataloader, dev_dataloader, test_dataloader, label2id, id2label = create_datasets_and_loaders(config)

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
