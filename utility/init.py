import os
from utility.image_process import prepare_data
from torch.utils.data import DataLoader, TensorDataset


def init_train(config, model):
    X_train, y_train = prepare_data(config)

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    Model = model.Model().cuda()
    return train_loader, Model


def init_test(config, model):
    os.makedirs(config.output_path, exist_ok=True)

    generator = model.Model().cuda()
    model_dir = config.saved_models_path
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("LightUNet_") and f.endswith(".pth")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    return generator, model_files
