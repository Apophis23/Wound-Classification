import argparse
from utils.dataset_util import *
from utils.train import *
from utils.helper import *
from utils.make_model import *
from utils.visualization import *
from sklearn.model_selection import train_test_split

def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Train models with different configurations.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for the model.")
    parser.add_argument("--dataset_path", type=str, default="./datasets/",
                        help="Path to the dataset directory.")
    parser.add_argument("--model_path", type=str, default="./model/",
                        help="Path to save model.")
    parser.add_argument("--image_path", type=str, default="./image/",
                        help="Path to save visualizations.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="test_ratio")
    parser.add_argument("--trial", type=int, default=20,
                        help="optimization number")

    # 명령행 인수 파싱
    args = parser.parse_args()

    # Argument 활용
    num_classes = args.num_classes
    dataset_path = args.dataset_path
    test_ratio = args.train_ratio
    model_path = args.model_path
    image_path = args.image_path
    trial = args.trial

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    df = create_dataset_dataframe(dataset_path)

    print(f"Dataset loaded with {len(df)} samples.")

    original_classes = df['Class']
    df['Class'] = pd.Categorical(df)
    df['Class'] = pd.Categorical(df['Class']).codes
    class_mapping = dict(enumerate(pd.Categorical(original_classes).categories))
    print("클래스 매핑:", class_mapping)
    train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['Class'], random_state=42)

    vit_model = create_vit_model(10)

    study = optimize_model(objective_function=objective, model_creator=vit_model, num_classes=num_classes, n_trials=trial, device=device)

    best_params = study.best_params
    optimal_lr = best_params["learning_rate"]
    optimal_batch_size = best_params["batch_size"]
    optimal_epochs = best_params["num_epochs"]

if __name__ == "__main__":
    main()