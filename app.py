import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

with open('model.pkl','rb') as file:
	bmi_model = pickle.load(file)

bmi_category = ['Extremely Underweight','Underweight','Healthy Weight','Overweight','Obese','Extremely Obese']

def head():
	st.title('BMI Predictor')
	st.markdown('### Welcome to **BMI Predictor App**') 
	st.markdown('#### We will help you predict your Body Mass Index using a machine learning model based on your gender, height, and weight')


def body():
	st.markdown('''Body mass index or BMI is a statistical index using a person's weight and height to provide an estimate of body fat in males and females of any age. It is calculated by taking a person's weight, in kilograms, divided by their height, in meters squared, or BMI = weight (in kg)/ height^2 (in m^2). The number generated from this equation is then the individual's BMI number.

- Severely underweight - BMI less than 16.5kg/m^2
- Underweight - BMI under 18.5 kg/m^2
- Normal weight - BMI greater than or equal to 18.5 to 24.9 kg/m^2
- Overweight – BMI greater than or equal to 25 to 29.9 kg/m^2
- Obesity – BMI greater than or equal to 30 to 39.9 kg/m^2
- Massive obesity – BMI greater than or equal to 40 kg/m^2 

source https://www.ncbi.nlm.nih.gov/books/NBK541070/''')
	gender = st.selectbox(label = 'Gender (Male/Female)',options = ['Female','Male'])
	height = st.number_input(label = 'Height (cm)')
	weight = st.number_input(label = 'Weight (kg)')

	user_features = pd.DataFrame([[gender,height,weight]],columns=['Gender','Height','Weight'])

	bmi_predict = st.button(label = 'Predict BMI')

	if bmi_predict:
		user_bmi = bmi_model.predict(user_features)[0]
		st.markdown(f'## ***{bmi_category[user_bmi]}***')
		user_bmi_proba = bmi_model.predict_proba(user_features)
		df = pd.DataFrame(user_bmi_proba[0],index = bmi_category, columns = ['BMI'])
		st.markdown('### **Prediction Confidence (value as probability)**')
		st.bar_chart(df)


	else:
		pass


def footer():
		pass


def main():
	head()
	body()
	footer()



if __name__ == '__main__':
	main()