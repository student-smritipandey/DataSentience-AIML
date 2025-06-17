from predict import predict_next_price

def main():
    print("🪙 Ethereum Price Predictor (CLI Version)\n")

    try:
        open_price = float(input("Enter Open Price: "))
        high_price = float(input("Enter High Price: "))
        low_price = float(input("Enter Low Price: "))
        volume = float(input("Enter Volume: "))
    except ValueError:
        print("⚠️ Please enter valid numeric values.")
        return

    result = predict_next_price(open_price, high_price, low_price, volume)
    print(f"\n📈 Predicted Close Price: ${result:.2f}")

if __name__ == "__main__":
    main()
