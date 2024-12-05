import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
import optuna.visualization

def objective(train_dataset, test_dataset, trial, model_fn, num_classes, device):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 5, 200, step=5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model_fn(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

# 모델 학습 함수
def train_model(model, train_loader, test_loader, num_epochs, lr, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model, model_name)
    print(f"Model saved as {model_name}")


# 모델 평가 함수
def evaluate_model(model, test_loader, device):
    model.eval()  # 평가 모드 설정
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    return true_labels, pred_labels

def optimize_model(objective_function, model_creator, num_classes, image_path, device, n_trials=20):
    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    # Optimize the model parameters
    study.optimize(
        lambda trial: objective_function(trial, model_creator, num_classes, device),
        n_trials=n_trials
    )

    model_name = str(model_creator)

    # Visualization file paths
    history_path = image_path + f"{model_name}_optimization_history.png"
    importance_path = image_path + f"{model_name}_hyperparameter_importance.png"

    # Save optimization history
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image(history_path)  # Save as PNG
    print(f"Optimization history saved to {history_path}")

    # Save parameter importances
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image(importance_path)  # Save as PNG
    print(f"Hyperparameter importance saved to {importance_path}")