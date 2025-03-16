import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import pickle
import streamlit as st

data = pd.read_csv("D:\Flight_Price_Prediction\Clean_Dataset.csv")

df = pd.read_csv("D:\Flight_Price_Prediction\Clean_Dataset.csv")

df.drop(['Unnamed: 0','flight','duration'],axis=1, inplace=True)

categorical_col = df.select_dtypes(include='object').columns

encoder = {}

for col in categorical_col:
    encoder[col] = LabelEncoder()
    df[col] = encoder[col].fit_transform(df[col])

# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(),annot=True,vmin=-1, vmax=1)
# plt.show()

x  = df.drop('price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.20, random_state=1)

model = LinearRegression()

model.fit(x_train,y_train)

score = model.score(x_test,y_test)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

# r2 = r2_score(y_test, y_pred)




# print(f"Score: {score}")


# print(f"mae: {mae}")

# print(f"mse: {mse}")

# print(f"rmse: {rmse}")

# print(f"r2: {r2}")




with open("Flight_price_prediction.pkl", 'wb') as f:
    pickle.dump(model,f)


with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder,f)


airline = st.selectbox("Airline", pd.unique(data['airline']))
source_city = st.selectbox("Source city", pd.unique(data['source_city']))
departure_time = st.selectbox("Departure Time ", pd.unique(data['departure_time']))
stops = st.selectbox("Number of Stops", pd.unique(data['stops']))
arrival_time = st.selectbox("Arrival Time", pd.unique(data['arrival_time']))
destination_city = st.selectbox("Destination City", pd.unique(data['destination_city']))
Travel_class = st.selectbox("Class", pd.unique(data['class']))
days_left = st.select_slider("Days Left", pd.unique(data['days_left']))


user_data = pd.DataFrame({ 
        'airline':[airline],
        'source_city': [source_city],
        'departure_time' : [departure_time],
        'stops' : [stops],
        'arrival_time': [arrival_time],
        'destination_city' : [destination_city],
        'class' : [Travel_class],
        'days_left' : [days_left]
    })


with open('D:\Flight_Price_Prediction\Flight_price_prediction.pkl', 'rb') as f:
    reloaded_model = pickle.load(f)

with open('D:\Flight_Price_Prediction\encoder.pkl', 'rb') as f:
    reloaded_encoder = pickle.load(f)

for col in user_data.columns:
    if col in reloaded_encoder:
        user_data[col] = reloaded_encoder[col].transform(user_data[col])

if st.button('Pleae show me the travel price'):
    prediction = reloaded_model.predict(user_data)


    st.write(f"Predicted price is Rs.{prediction[0]:.2f}/-")
