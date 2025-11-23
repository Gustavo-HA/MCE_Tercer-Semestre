import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import logging
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Model imports
from cnns.lenet5 import LeNet5
from cnns.alexnet import AlexNet
from cnns.vgg import VGG16
from cnns.inception import GoogLeNet
from cnns.movilenet import MobileNet_1_0
from cnns.movilenetv2 import MobileNetV2_1_0
from cnns.restnet import ResNet18


# ---------- Data utilities ----------

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
MNIST_MEAN = [0.1307]
MNIST_STD = [0.3081]


def build_transforms(dataset: str, model_name: str, train: bool) -> transforms.Compose:
    is_lenet = model_name.lower() == "lenet5"

    if dataset.lower() == "cifar10":
        # Base transforms for CIFAR-10
        aug = []
        if train:
            aug += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        # Channel handling for LeNet5 (expects 1-channel)
        if is_lenet:
            aug += [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        else:
            aug += [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        return transforms.Compose(aug)

    elif dataset.lower() == "mnist":
        # MNIST is 1x28x28 → pad to 32x32
        aug = [transforms.Pad(2)]
        # Channel handling for 3-channel models
        if is_lenet:
            aug += [
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        else:
            aug += [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN * 3, MNIST_STD * 3),
            ]
        return transforms.Compose(aug)

    else:
        raise ValueError(f"Dataset no soportado: {dataset}")


def build_dataloaders(dataset: str, model_name: str, data_dir: Path, batch_size: int,
                      num_workers: int = 2, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = dataset.lower()

    if dataset == "cifar10":
        full_train = datasets.CIFAR10(root=str(data_dir), train=True,
                                      transform=build_transforms("cifar10", model_name, train=True),
                                      download=False)
        test = datasets.CIFAR10(root=str(data_dir), train=False,
                                 transform=build_transforms("cifar10", model_name, train=False),
                                 download=False)
        train_len, val_len = 45000, 5000
    elif dataset == "mnist":
        full_train = datasets.MNIST(root=str(data_dir), train=True,
                                    transform=build_transforms("mnist", model_name, train=True),
                                    download=False)
        test = datasets.MNIST(root=str(data_dir), train=False,
                              transform=build_transforms("mnist", model_name, train=False),
                              download=False)
        train_len, val_len = 50000, 10000
    else:
        raise ValueError(f"Dataset no soportado: {dataset}")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ---------- Model factory ----------

def create_model(model_name: str, num_classes: int = 10):
    name = model_name.lower()
    if name == "lenet5":
        return LeNet5(num_classes=num_classes)
    if name == "alexnet":
        return AlexNet(num_classes=num_classes, dropout=0.5)
    if name == "vgg16":
        return VGG16(num_classes=num_classes, dropout=0.5)
    if name in ("inception", "googlenet"):
        return GoogLeNet(num_classes=num_classes, aux_logits=False)
    if name in ("mobilenet", "mobilenet_v1", "mobilenet1"):
        return MobileNet_1_0(num_classes=num_classes)
    if name in ("mobilenetv2", "mobilenet_v2"):
        return MobileNetV2_1_0(num_classes=num_classes)
    if name in ("resnet", "resnet18"):
        return ResNet18(num_classes=num_classes)
    raise ValueError(f"Modelo desconocido: {model_name}")


# ---------- Training utilities ----------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_sum += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_model(
    dataset: str,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    opt_name: str = "adam",
    device: torch.device | None = None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(dataset, model_name, data_dir, batch_size, seed=seed)

    # Model
    model = create_model(model_name, num_classes=10).to(device)

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    if opt_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs//2,1), gamma=0.1)

    best_val_acc = 0.0
    best_path = output_dir / f"{dataset}_{model_name}_best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow logging
    mlflow.log_params({
        "dataset": dataset,
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": opt_name,
        "device": str(device),
        "seed": seed,
    })

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr,
        }, step=epoch)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, best_path)

    # Final test evaluation with best checkpoint if available
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

    # Log artifact
    if best_path.exists():
        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

    return {"best_val_acc": best_val_acc, "test_acc": test_acc, "ckpt": str(best_path) if best_path.exists() else None}


def run_all(models: List[str], datasets_list: List[str], args):
    data_dir = Path("data")
    output_dir = Path("outputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    mlflow.set_tracking_uri("http://localhost:1825")
    logging.getLogger(__name__).info("MLflow tracking URI en http://localhost:1825")
    
    for ds in datasets_list:
        # One MLflow experiment per dataset
        mlflow.set_experiment(f"visionT2-{ds.upper()}")
        logging.getLogger(__name__).info(f"Iniciando experimento para el conjunto de datos: {ds}")
        results[ds] = {}
        for model_name in models:
            logging.getLogger(__name__).info(f"Entrenando modelo {model_name} en {ds}")
            with mlflow.start_run(run_name=f"{model_name}"):
                res = train_one_model(
                    dataset=ds,
                    model_name=model_name,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    opt_name=args.optimizer,
                    device=device,
                    seed=args.seed,
                )
                results[ds][model_name] = res
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar múltiples CNNs en CIFAR-10 y MNIST")
    parser.add_argument("--datasets", type=str, default="both", choices=["cifar10", "mnist", "both"],
                        help="Qué conjunto(s) de datos entrenar")
    parser.add_argument("--epochs", type=int, default=30, help="úmero de épocas de entrenamiento por modelo")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño del lote para el entrenamiento")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizador")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--models", type=str, nargs="*",
                        default=["lenet5", "alexnet", "vgg16", "inception", "mobilenet", "mobilenetv2", "resnet18"],
                        help="Subconjunto de modelos a entrenar")
    return parser.parse_args()


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("visiont2")
    args = parse_args()
    datasets_list = ["cifar10", "mnist"] if args.datasets == "both" else [args.datasets]
    results = run_all(args.models, datasets_list, args)

    # Print a compact summary
    logger.info("\nResumen de entrenamiento:")
    for ds, res_by_model in results.items():
        logger.info(f"\nDataset: {ds}")
        for model_name, stats in res_by_model.items():
            logger.info(f"  {model_name:12s} | best_val_acc: {stats['best_val_acc']:.4f} | test_acc: {stats['test_acc']:.4f}")


if __name__ == "__main__":
    main()
