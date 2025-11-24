import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from train import build_dataloaders, create_model

def load_model(model_path, dataset, model_name, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(model_name, num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def get_predictions(model, loader, device):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    return all_targets, all_preds

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, 
                fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True') 
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def save_classification_report(y_true, y_pred, classes, output_path):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(output_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs"
    figures_dir = base_dir / "documento/figures"
    reports_dir = base_dir / "documento/class_reports"
    data_dir = base_dir / "data"

    # Asegurar que los directorios existan
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Clases de CIFAR-10
    cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Clases de MNIST
    mnist_classes = [str(i) for i in range(10)]

    model_files = list(outputs_dir.glob("*_best.pt"))
    
    if not model_files:
        print("No model files found in outputs/")
        return

    # Almacenar conteo de parámetros
    param_counts = []

    for model_file in model_files:
        filename = model_file.name
        # Formato esperado: {dataset}_{model_name}_best.pt
        parts = filename.split('_')
        if len(parts) < 3:
            print(f"Skipping {filename}: unexpected format")
            continue
            
        dataset = parts[0]
        # model_name podría contener guiones bajos (e.g. mobilenet_v2), pero aquí los nombres de archivo parecen ser como cifar10_mobilenetv2_best.pt
        # Asumamos que la última parte es 'best.pt' y la primera es dataset. El medio es model_name.
        model_name = "_".join(parts[1:-1])
        
        print(f"Processing {dataset} - {model_name}...")
        
        classes = cifar10_classes if dataset == "cifar10" else mnist_classes
        
        try:
            # Cargar datos
            _, _, test_loader = build_dataloaders(dataset, model_name, data_dir, batch_size=128)
            
            # Cargar modelo
            model = load_model(model_file, dataset, model_name, device)
            
            # Contar parámetros
            num_params = count_parameters(model)
            param_counts.append({
                'dataset': dataset,
                'model': model_name,
                'params': num_params
            })
            print(f"Model {model_name} has {num_params:,} trainable parameters")

            # Obtener predicciones
            y_true, y_pred = get_predictions(model, test_loader, device)
            
            # Guardar matriz de confusión
            cm_path = figures_dir / f"cm_{dataset}_{model_name}.pdf"
            plot_confusion_matrix(y_true, y_pred, classes, cm_path)
            print(f"Saved confusion matrix to {cm_path}")
            
            # Guardar reporte de clasificación
            report_path = reports_dir / f"report_{dataset}_{model_name}.csv"
            save_classification_report(y_true, y_pred, classes, report_path)
            print(f"Saved classification report to {report_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    # Guardar conteo de parámetros en CSV
    if param_counts:
        df_params = pd.DataFrame(param_counts)
        params_path = reports_dir / "model_parameters.csv"
        df_params.to_csv(params_path, index=False)
        print(f"Saved model parameters to {params_path}")

if __name__ == "__main__":
    main()
