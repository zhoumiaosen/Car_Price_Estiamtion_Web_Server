## Overview
This project aims to deploy a used car price prediction model on a web platform, allowing users to instantly determine the estimated price of a used car. The AI model is built upon the resources and code available in this repository: [Car_Price_Estimation](https://github.com/zhoumiaosen/Car_Price_Estiamtion).

## Project Structure
```plaintext
car_price_prediction/
│
│
├── models/
│   ├── car_price_model.pth         	  # Saved model state dictionary
│   ├── scaler_X.pkl                	  # Saved X scaler (MinMaxScaler for features)
│   ├── scaler_y.pkl                	  # Saved y scaler (MinMaxScaler for target)
│   ├── Model_encoder.pkl         	  # Saved encoder (encoder for car model)
│   ├── Brand_encoder.pkl            	  # Saved encoder (encoder for car brand)
│   └── Status.pkl              	  # Saved encoder (encoder for car status)
│
├── src/
│   ├── app.py                            # web server
│   ├── train.py                          # Script for training the model
│   ├── predict.py                        # Script for making predictions
│   └── model.py                          # Model definition (PyTorch neural network)
│
├── notebooks/
│   └── Used_Car_Price_Estimation.ipynb   # Jupyter notebook for data exploration and preprocessing
│
└── README.md                             # Project overview and instructions
```

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- joblib (optional, for saving and loading scalers)
- Jupyter Notebook (optional, for data exploration)

You can install the necessary packages using the following command:

```bash
pip install torch scikit-learn pandas joblib
```

## How to Run

### 1. Running Web Sever
Before making a prediction, you'll need to start the web server by running the `app.py` script:

```bash
jupyter notebook notebook/app.ipynb
```

This script will:

- Load the saved model, scalers, and encoders from the models/ directory.
- Start the server, allowing users to input data via the web interface.
- Preprocess the input data to ensure it matches the format of the training data.
- Output the predicted price based on the car features provided.

### 2. Making Predictions

To predict the price, use the following command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "Brand": ["Toyota"],
  "Model": ["Camry"],
  "Year": [2000],
  "Status": ["Used"],
  "Mileage": [20000.0]
}' http://localhost:port/predict

```

This command will:

- Require input data formatted similarly to the training data.
- Send the data to the server for price prediction.
- Output the predicted price based on the provided car features.

### 3. Model Architecture
The model is a simple feedforward neural network with the following layers:

Input Layer: Matches the number of features after preprocessing (5 in this project).
Hidden Layers: Two hidden layers with ReLU activation.
Output Layer: A single neuron with ReLU to predict the price.

### 4. Loading the Model
- **Model:** The trained model's state dictionary is saved as car_price_model.pth in the models/ directory.
- **Scalers:** The MinMaxScaler objects used to normalize features (scaler_X.pkl) and the target variable (scaler_y.pkl) are also saved in the models/ directory.
- **Encoder:** The Encoder for model (Model_encoder.pkl), brand (Brand_encoder.pkl), and status (Status_encoder.pkl) are all saved in the models/ directory.

To load the model scalers, and encoder:

```bash
import torch
import joblib

# Load the model
model_load = torch.load(car_price_model.pth, map_location=device)
model_load.eval()

# Load the scalers
scaler_X = joblib.load(scaler_X.pkl)
scaler_y = joblib.load(scaler_y.pkl)

# Load Encoders
Brand_encoder = joblib.load(Brand_encoder.pkl)
Model_encoder = joblib.load(Model_encoder.pkl)
Status_encoder = joblib.load(Status_encoder.pkl)
```


## Next Steps
- **Model Tuning:**
 Experiment with different neural network architectures and hyperparameters to improve model performance.
- **Feature Engineering:**
 Consider adding more features or transforming existing ones to capture more information from the data.
- **Evaluation:** 
Implement more comprehensive evaluation metrics, like MAE or R^2, to better assess the model's performance.
- **Data Expansion:** 
 Gather and integrate additional datasets to enhance the model’s accuracy and robustness.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request to contribute to the project.

## License
This project is licensed under the MIT License.
