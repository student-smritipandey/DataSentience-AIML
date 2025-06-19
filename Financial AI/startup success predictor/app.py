import argparse
from src.train_model import train_and_save_model
from src.predict import predict_new_data


def main():
    parser = argparse.ArgumentParser(description="Startup Success Analyzer")
    parser.add_argument("action", choices=["train", "predict"], help="Action to perform")
    parser.add_argument("file", help="Path to CSV file")
    args = parser.parse_args()

    if args.action == "train":
        print("Training model with data from:", args.file)
        train_and_save_model(args.file)
    elif args.action == "predict":
        print("Predicting outcomes for data in:", args.file)
        predictions = predict_new_data(args.file)
        print("Predictions:", predictions)


if __name__ == "__main__":
    main()