import streamlit as st
import joblib
import pandas as pd
import random

# Load the trained logistic regression model
model1 = joblib.load('logistic_regression_model.pkl')
model2 = joblib.load('attrition_model.pkl')

# Streamlit app
def main():
    st.title('Employee Attrition Prediction')

    # Add input components for each feature
    age = st.text_input('Age')
    business_travel = st.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    daily_rate = st.text_input('Daily Rate')
    department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    distance_from_home = st.text_input('Distance From Home')
    education = st.selectbox('Education', ['1', '2', '3', '4', '5'])
    
    # Additional input components
    environment_satisfaction = st.selectbox('Environment Satisfaction', ['1', '2', '3', '4', '5'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    hourly_rate = st.text_input('Hourly Rate')
    job_involvement = st.selectbox('Job Involvement', ['1', '2', '3', '4', '5'])
    job_level = st.selectbox('Job Level', ['1', '2', '3', '4', '5'])
    job_role = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    job_satisfaction = st.selectbox('Job Satisfaction', ['1', '2', '3', '4', '5'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    monthly_income = st.text_input('Monthly Income')
    num_companies_worked = st.text_input('Num Companies Worked')
    over_time = st.selectbox('Over Time', ['Yes', 'No'])
    percent_salary_hike = st.text_input('Percent Salary Hike')
    performance_rating = st.selectbox('Performance Rating', ['1', '2', '3', '4', '5'])
    total_working_years = st.text_input('Total Working Years')
    training_times_last_year = st.text_input('Training Times Last Year')
    work_life_balance = st.selectbox('Work Life Balance', ['1', '2', '3', '4', '5'])
    years_at_company = st.text_input('Years At Company')
    years_in_current_role = st.text_input('Years In Current Role')
    years_since_last_promotion = st.text_input('Years Since Last Promotion')
    years_with_curr_manager = st.text_input('Years With Curr Manager')

    session_state = st.session_state.setdefault('predictions', {'prediction1': None, 'prediction2': None})

    # Make prediction when a button is clicked
    predict_button1 = st.button('Predict1')

    if predict_button1:
        prediction1 = predict(age, business_travel, daily_rate, department, distance_from_home, education,
                             environment_satisfaction, gender, hourly_rate, job_involvement, job_level,
                             job_role, job_satisfaction, marital_status, monthly_income,
                             num_companies_worked, over_time, percent_salary_hike, performance_rating, total_working_years, training_times_last_year, work_life_balance,
                             years_at_company, years_in_current_role, years_since_last_promotion,
                             years_with_curr_manager, model1)

        session_state['prediction1'] = prediction1
    if session_state['prediction1'] is not None:
        st.subheader('Prediction 1')
        st.write(session_state['prediction1'])

    st.header("Chatbot")
    df=pd.read_csv('Employee Attrition.csv')
    #df = pd.read_csv('C:/Users/vyshn/OneDrive - vit.ac.in/Documents/MTech Integrated/3rd_year/Sem-6/SPM/Project/Employee Attrition.csv')
    scenario = st.text_area('Enter the scenario:')
    input_data = {}

    predict_button2 = st.button('Predict2')

    if predict_button2:
        conditions = [condition.strip() for condition in scenario.split('and')]
        for condition in conditions:
            key_value = [item.strip() for item in condition.split('=')]
            key = key_value[0]
            value = key_value[1]
            input_data[key] = value
        
        filled_input_data = generate_random_data(df, input_data)
        
        # Make prediction using Model 2
        prediction2 = predict_attrition(model2, pd.DataFrame(filled_input_data, index=[0]))
        
        session_state['prediction2'] = prediction2

        if session_state['prediction2'] is not None:
            st.subheader('Prediction 2')
        if session_state['prediction2'][0] == 1:
            st.write('Based on the scenario, it is predicted that the employee is likely to leave the organization.')
        else:
            st.write('Based on the scenario, it is predicted that the employee is likely to stay in the organization.')

# Define the function for making predictions
def predict(age, business_travel, daily_rate, department, distance_from_home, education,
            environment_satisfaction, gender, hourly_rate, job_involvement, job_level,
            job_role, job_satisfaction, marital_status, monthly_income,
            num_companies_worked, over_time, percent_salary_hike, performance_rating, total_working_years, training_times_last_year, work_life_balance,
            years_at_company, years_in_current_role, years_since_last_promotion,
            years_with_curr_manager, model):    

    business_travel_encoded = 0 if business_travel == 'Non-Travel' else (1 if business_travel == 'Travel_Rarely' else 2)
    department_encoded = 0 if department == 'Sales' else (1 if department == 'Research & Development' else 2)
    
    # Manual encoding for job role
    job_role_mapping = {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2,
                        'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5,
                        'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8}
    job_role_encoded = job_role_mapping[job_role]

    gender_encoded = 1 if gender == 'Male' else 0
    marital_status_encoded = 0 if marital_status == 'Single' else (1 if marital_status == 'Married' else 2)
    over_time_encoded = 1 if over_time == 'Yes' else 0

    # Combine input features into a single list or array
    features = [int(age), business_travel_encoded, int(daily_rate), department_encoded, int(distance_from_home), int(education),
                int(environment_satisfaction), gender_encoded, int(hourly_rate), int(job_involvement), int(job_level),
                job_role_encoded, int(job_satisfaction), marital_status_encoded, int(monthly_income), 
                int(num_companies_worked), over_time_encoded, int(percent_salary_hike), int(performance_rating), int(total_working_years),
                int(training_times_last_year), int(work_life_balance), int(years_at_company),
                int(years_in_current_role), int(years_since_last_promotion), int(years_with_curr_manager)] 

    # Make prediction using the model
    prediction1 = model.predict([features])
    
    # Map prediction to employee status
    status = "Employee is leaving Organisation" if prediction1[0] == 1 else "Employee is staying in Organisation"

    # Return the prediction
    return status

def predict_attrition(model, input_data):
    return model.predict(input_data)

def generate_random_data(df, input_data):
    filled_data = {}
    for column in df.columns:
        if column not in input_data:
            if df[column].dtype == 'object':
                filled_data[column] = random.choice(df[column])
            else:
                filled_data[column] = random.uniform(df[column].min(), df[column].max())
        else:
            filled_data[column] = input_data[column]
    return filled_data

if __name__ == '__main__':
    main()
