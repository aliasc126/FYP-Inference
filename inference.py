from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
model = load_model('../FYP/testing.h5')
scaler = joblib.load('../FYP/scaler1.pkl')

def predict(cpu, memory, Network):

    new_data = np.array([[cpu, memory, Network]])  # Replace '...' with your actual feature values
    new_data_scaled = scaler.transform(new_data)
    new_data_reshaped = new_data_scaled.reshape(1, 1, new_data_scaled.shape[1])
    predictions = model.predict(new_data_reshaped)
    print (predictions)
    if predictions[0][0]<0.3:
        return 0
    else:
        return 1

def main():
    cpu =56.2
    memory = 112836608.0
    Network = 229940
    
    # read the malicious.csv file
    df = pd.read_csv('../MLOPS-Pipeline/youtube data.csv', low_memory=False)
    # drop the column malicious
    df = df.drop('malicious', axis=1)
    # remove the column pid from both dataframes
    df = df.drop('pid', axis=1)
    
    # take row 0 from the dataframe
    
    row = df.iloc[100]
    print(row)
    cpu = row['Cpu']
    memory = row['memory']
    Network = row['Network']
    
    prediction = predict(cpu, memory, Network)
    print (prediction)
    
if __name__ == "__main__":
    main()