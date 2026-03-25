import torch
import json
from model import create_model
from data_module import create_datasets_and_loaders
from trainer import train, test

def main():
    # 1.读取配置
    with open("args.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2.数据
    train_dataloader, dev_dataloader, test_dataloader, _, _ = create_datasets_and_loaders(config)

    # 3.设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.模型
    model = create_model(config)
    model.to(device)

    # 5.优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # 6.训练
    train(model, train_dataloader, dev_dataloader, optimizer, device, config)

    # 7.测试
    test(model, test_dataloader, device, config)


if __name__ == "__main__":
    main()