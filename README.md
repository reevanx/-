# -
智能养殖利用边缘计算和机器学习技术,实时监测畜牧场的环境参数和动物健康状况,优化饲养管理,提高养殖效率和动物福利。
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulate data collection from sensors
def collect_data(n_samples=1000):
    # Simulated environmental data
    temperature = np.random.normal(25, 5, n_samples)  # Average temperature in Celsius
    humidity = np.random.normal(50, 10, n_samples)  # Average humidity in percentage
    
    # Simulated animal health data
    heart_rate = np.random.normal(60, 10, n_samples)  # Average heart rate in bpm
    
    # Simulated health status (0: healthy, 1: needs attention)
    health_status = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    data = pd.DataFrame({
        'Temperature': temperature,
        'Humidity': humidity,
        'HeartRate': heart_rate,
        'HealthStatus': health_status
    })
    return data

# Process and split the dataset
def prepare_data(data):
    X = data[['Temperature', 'Humidity', 'HeartRate']]
    y = data['HealthStatus']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple machine learning model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict health status and recommend actions
def recommend_actions(model, X_test):
    predictions = model.predict(X_test)
    recommendations = []
    for pred in predictions:
        if pred == 0:
            recommendations.append('Maintain current conditions.')
        else:
            recommendations.append('Check animal health and environment.')
    return recommendations

# Main function to orchestrate the data flow
def main():
    data = collect_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    recommendations = recommend_actions(model, X_test)
    
    # Evaluation (for demonstration purposes)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'Model Accuracy: {accuracy:.2f}')
    print('Sample Recommendations:', recommendations[:5])

if __name__ == '__main__':
    main()
