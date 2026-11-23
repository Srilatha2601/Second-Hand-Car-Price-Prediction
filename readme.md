# Second-Hand Car Price Prediction

A machine learning project that predicts the selling price of used cars using vehicle features and a Random Forest Regression model. The project also includes a basic Top 5 car recommendation feature based on price, vehicle age, and kilometers driven.

## Project Overview

Buying a second-hand car can be challenging because prices vary depending on factors such as vehicle age, kilometers driven, fuel type, transmission, and engine capacity.

This project uses historical car data to train a machine learning model that estimates the selling price of a used car based on its features.

It also provides a simple recommendation mechanism to identify five cars with favorable scores based on their price, age, and kilometers driven.

## Objectives

- Predict the selling price of a used car using machine learning.
- Understand the relationship between vehicle features and selling price.
- Perform data preprocessing and feature engineering.
- Evaluate the trained model using the R² score.
- Visualize actual prices against predicted prices.
- Save the trained model for future use.
- Recommend five cars using a scoring formula.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Implements the machine learning workflow |
| Pandas | Loads, cleans, and processes the dataset |
| Matplotlib | Visualizes actual and predicted prices |
| Scikit-learn | Provides data splitting, Random Forest Regression, and model evaluation |
| Pickle | Saves the trained model |
| Git | Tracks changes to the project |
| Google Colab / Jupyter Notebook | Can be used to run Python notebooks if desired |

## Dataset

The project uses a CarDekho car dataset in CSV format.

Each row represents a car, and the columns contain information about the vehicle.

The dataset includes features such as:

- Car name, brand, and model
- Vehicle age
- Kilometers driven
- Fuel type
- Transmission type
- Mileage
- Engine capacity
- Selling price

The selling price is the target variable that the model learns to predict.

**Dataset source:** Kaggle — CarDekho dataset.

The dataset is not included in this repository. Download it from Kaggle and place the CSV file in the project directory.

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`.

### 2. Data Preprocessing

The preprocessing steps include:

- Removing an unwanted index column if it exists.
- Dropping the `car_name`, `brand`, and `model` columns.
- Converting categorical variables into numerical features using one-hot encoding.

The project uses `pd.get_dummies()` with `drop_first=True` for categorical feature encoding.

### 3. Feature and Target Selection

The input features are stored in `X`, while the target selling price is stored in `y`.

The model learns the relationship between the input features and the selling price.

### 4. Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

The training data is used to fit the model, while the testing data is used to evaluate its predictions.

The split uses `random_state=42` for reproducibility.

### 5. Model Training

The project uses **Random Forest Regression**, implemented through Scikit-learn's `RandomForestRegressor`.

Random Forest combines predictions from multiple decision trees to estimate the selling price.

It can learn complex relationships between vehicle features and prices without assuming that the relationship must be linear.

### 6. Model Evaluation

The trained model predicts prices for the test dataset.

The predictions are evaluated using the R² score.

The project also generates a scatter plot comparing actual prices with predicted prices.

### 7. Model Saving

The trained model is saved as `model.pkl` using Pickle.

This allows the trained model to be loaded and reused without training it again.

## Top 5 Car Recommendation

The project includes a basic recommendation feature that calculates a score for each car using:

- Selling price: 50% weight
- Kilometers driven: 30% weight
- Vehicle age: 20% weight

The inverse of each value is used in the scoring formula. Cars with lower prices, fewer kilometers driven, and lower vehicle age generally receive higher scores.

The dataset is sorted by score in descending order, and the five highest-scoring cars are displayed.

This is a rule-based scoring feature, not a separate machine learning model.

## Project Structure

```text
Second-Hand-Car-Price-Prediction/
│
├── car_price_prediction.py
├── car_data.csv
├── README.md
├── requirements.txt
├── .gitignore
└── model.pkl
```

`model.pkl` is generated when the Python program is executed successfully. It does not need to be committed to GitHub because it can be generated again from the training code.

## Installation and Setup

### Prerequisites

- Python 3.9 or later
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Second-Hand-Car-Price-Prediction.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Navigate to the Project Directory

```bash
cd Second-Hand-Car-Price-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download the CarDekho dataset from Kaggle and place the CSV file in the project directory.

Make sure the filename matches the filename used in the Python code.

### 5. Run the Project

```bash
python car_price_prediction.py
```

The program will:

1. Load and preprocess the dataset.
2. Train the Random Forest Regression model.
3. Predict prices for the testing data.
4. Display the R² score.
5. Generate an actual-versus-predicted price graph.
6. Save the trained model as `model.pkl`.
7. Display the Top 5 recommended cars.

## Results

The model is evaluated using the R² score on the held-out test dataset.

The actual-versus-predicted scatter plot provides a visual comparison of model predictions and actual selling prices.

The Top 5 recommendation feature displays cars selected using the scoring formula.

The actual R² score depends on the dataset and model configuration and should be recorded after running the project.

## Limitations

- The predicted price is an estimate, not a guaranteed market price.
- The model depends on the quality and representativeness of the training dataset.
- The current recommendation feature considers only price, vehicle age, and kilometers driven.
- The current code does not include a deployed web application or a real-time market price integration.

## Future Improvements

- Add more vehicle features and improve data preprocessing.
- Compare Random Forest with other regression algorithms.
- Tune model hyperparameters to improve performance.
- Add additional evaluation metrics such as MAE and RMSE.
- Develop a web interface for users to enter car details and receive predictions.
- Improve the recommendation feature using more vehicle condition and reliability information.

## Author

**Srilatha**

B.Tech — Computer Science and Engineering

GitHub: [Srilatha2601](https://github.com/Srilatha2601)
