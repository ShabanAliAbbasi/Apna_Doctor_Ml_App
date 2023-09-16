import streamlit as st
import numpy as np
from tensorflow import keras
import pickle
from streamlit_option_menu import option_menu
import json


# import cv2
# import tensorflow as tf
# from PIL import Image

# Call set_page_config as the first Streamlit command
st.set_page_config(
    page_title="Apna Doctor",
    page_icon="hospital",
    layout="wide",
)



multiple_disease_model = pickle.load(open(
    'C:\\Users\\Shoban\\Desktop\\Virtual_Doctor_Project\\Data_about_deseases\\Heart_data\\Save models\\multiple_disease_prediction.sav',
    'rb'))


# diabetes_model = pickle.load(open('C:\Users\Shoban\Desktop\Virtual_Doctor_Project\Data_about_deseases\Heart_data\diabetes.sav', 'rb'))
diabetes_model = pickle.load(open(
    'C:\\Users\\Shoban\\Desktop\\Virtual_Doctor_Project\\Data_about_deseases\\Heart_data\\Save models\\diabetes.sav',
    'rb'))

# heart_disease_model = pickle.load(open('C:\Users\Shoban\Desktop\Virtual_Doctor_Project\Data_about_deseases\Heart_data\heart.sav','rb'))
heart_model = pickle.load(open(
    'C:\\Users\\Shoban\\Desktop\\Virtual_Doctor_Project\\Data_about_deseases\\Heart_data\\\\Save models\\heart2.sav',
    'rb'))
# parkinsons_model = pickle.load(open('C:\Users\Shoban\Desktop\Virtual_Doctor_Project\Data_about_deseases\Heart_data\parkinson.sav', 'rb'))

parkinson_model = pickle.load(open(
    'C:\\Users\\Shoban\\Desktop\\Virtual_Doctor_Project\\Data_about_deseases\\Heart_data\\\\Save models\\parkinson.sav',
    'rb'))

covid_model = pickle.load(open(
    'C:\\Users\\Shoban\\Desktop\\Virtual_Doctor_Project\\Data_about_deseases\\Heart_data\\\\Save models\\covid.sav',
    'rb'))

# Load the trained pneumonia detection model
pneumonia_model = keras.models.load_model("C:\\Users\\Shoban\\Desktop\\testing\\pneumonia_detection_model.h5")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('APNA DOCTOR MULTIPLE DISEASE PREDICTION SYSTEM',
                           ['Home',
                            'General Check Up',
                            'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            "Corona Virus Prediction",
                            "Pheumonia Detection",
                            "Any Feed back",
                            'About me'],
                           icons=["house-heart" ,"prescription2",'activity', 'heart', 'person' , "virus","lungs" ,"pencil-square",'person'],
                           default_index=0)

if selected == "Home":
    # import streamlit as st
    # from PIL import Image
    # import io
    #
    # import json
    # import streamlit as st
    from streamlit_lottie import st_lottie

    st.title('APNA DOCTOR USING MACHING LEARNING AND DEEP LEARNING')

    path = r"C:\Users\Shoban\Downloads\animation_lmhf84b8 (1).json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              # height=,
              width=550,
              speed=1,
              loop=True,
              quality='high',
              key='Home'
              )


    st.subheader("The purpose of this application is to help doctors in diagnosing, if a patient has or is going to have a Any disease and wants to get some details about his health that will help prevent the disease in the future.")



    from streamlit_lottie import st_lottie

    path = r"C:\Users\Shoban\Downloads\animation_lmi6tm1s.json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=400,
              width=400,
              speed=1,
              loop=True,
              quality='high',
              key='about project'
              )

    st.header("""Welcome to Apna Doctor - Predictive Healthcare at Your Fingertips

Imagine having the power to predict potential health issues before they become major concerns. At Apna Doctor, we're ushering in a new era of healthcare where knowledge is prevention, and data is your shield against illness. Our innovative platform combines the latest advancements in machine learning and medical science to offer you personalized healthcare insights like never before.

Discover Your Health Tomorrow, Today

✨ General Disease Predictions: Our cutting-edge algorithms analyze your health data to identify early signs of general illnesses. Whether it's a common cold or a seasonal flu, we provide you with early warnings to stay one step ahead of discomfort.

🩺 Diabetes Prognosis: For those managing diabetes or at risk of developing it, Apna Doctor offers predictive analytics to monitor blood sugar trends and prevent unexpected complications.

❤️ Heart Health Monitoring: We take heart health seriously. Our platform tracks vital indicators and offers risk assessments for cardiovascular conditions, empowering you to make informed lifestyle choices.

🧠 Parkinson's Detection: Early detection of Parkinson's disease can significantly improve patient outcomes. Apna Doctor's predictive tools are designed to identify potential Parkinson's symptoms, facilitating early intervention and support.

🦠 COVID-19 Risk Assessment: In the age of pandemics, knowing your COVID-19 risk is crucial. Our models assess your exposure and health data to provide tailored guidance and precautions.

🫁 Pneumonia Disease Prediction: In an era where healthcare vigilance is paramount, understanding your risk for pneumonia is essential. Our predictive models evaluate your health profile and relevant data to offer personalized insights and precautionary measures.
Your Health, Your Data, Your Future

Apna Doctor is more than just a healthcare platform; it's your health companion for life. We respect your privacy and ensure that your data is safeguarded at all times, so you can trust us with your most valuable asset: your well-being.

Take Control of Your Health Journey

With Apna Doctor, you're in control. Our easy-to-use platform puts health insights, recommendations, and support right at your fingertips. Start today, and embark on a proactive journey toward a healthier, more vibrant life.

Ready to unlock the power of predictive healthcare? [Get Started Now] - Your healthier future awaits with Apna Doctor!






""")

if selected == 'General Check Up':
    st.title('GENERAL CHECK-UP')


    from streamlit_lottie import st_lottie


    path = r"C:\Users\Shoban\Downloads\animation_lmi5d5di.json"
    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=00,
              width=600,
              speed=1,
              loop=True,
              quality='high',
              key='General check up'
              )

    st.markdown(
        '_This_ _purpose_ _of_ _this_  _application_ _is_ _to_ _help_ _doctors_ _diagnose_ , _if_ _a_ _patient_ _has_ _a_ _Any_ _Disease_ _or_ _may_ _have_ _it_ _after_ _sometime_ _using_ _few_ _details_ _about_ _their_ _health_')

    st.markdown('Please **Enter _the_ _below_ details** to know the results -')


    options_1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
                 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue']

    symptoms_1 = st.multiselect('Select options:', options_1)


    options_2 = [ 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
                  'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss']

    symptoms_2 = st.multiselect('Select options:', options_2)

    options_3 = ['restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
                 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration']

    symptoms_3 = st.multiselect('Select options:', options_3)

    options_4 = [ 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
                  'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain']

    symptoms_4 = st.multiselect('Select options:', options_4)

    options_5 = ['diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
                 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision']
    symptoms_5 = st.multiselect('Select options:', options_5)

    options_6 =['phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
                'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements']
    symptoms_6 = st.multiselect('Select options:', options_6)

    options_7 = [ 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
                  'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels']

    symptoms_7 = st.multiselect('Select options:', options_7)

    options_8 = ['puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
                 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain']
    symptoms_8 = st.multiselect('Select options:', options_8)

    options_9 = [ 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
                  'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort']

    symptoms_9 = st.multiselect('Select options:', options_9)

    options_10 = [ 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
                   'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body']


    symptoms_10 = st.multiselect('Select options:', options_10)

    options_11 = [ 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
                   'rusty_sputum', 'lack_of_concentration']
    symptoms_11 = st.multiselect('Select options:', options_11)

    options_12 = [ 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
                   'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf']
    symptoms_12 = st.multiselect('Select options:', options_12)

    options_13 = [ 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
                   'small_dents_in_nails','inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
    symptoms_13 = st.multiselect('Select options:', options_13)

    all_symptoms=symptoms_1+symptoms_2+symptoms_3+symptoms_4+symptoms_5+symptoms_6+symptoms_7+symptoms_8+symptoms_9+symptoms_10+symptoms_11+symptoms_12+symptoms_13
    # st.write(all_symptoms)
    col=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting',
     'vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness',
     'lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache',
     'yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever',
     'yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision',
     'phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region',
     'bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
     'enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
     'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
     'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching',
     'toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation',
     'dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration',
     'visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
     'fluid_overload.1','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
     'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

    result = []


    def func(col, all_symptoms):
        for item in col:
            if item in all_symptoms:
                result.append(1)
            else:
                result.append(0)


    func(col, all_symptoms)
    

    if st.button('Test Result'):
        if len(all_symptoms)<3:
            st.warning("Plz Enter atleast 3 symptoms")
        else:

            input_data=(result)

            # change the input data to a numpy array
            input_data_as_numpy_array = np.asarray(input_data)

            # reshape the numpy array as we are predicting for only on instance
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            multiple_disease_prediction = multiple_disease_model.predict(input_data_reshaped)
            disease=multiple_disease_prediction


            # The Patient either have {disease[0]}Disease or are likely to have it. Please visit the doctor as soon as possible.")
            st.success(f"The Patient either have \' {disease[0]} Disease \' or are likely to have it. Please visit the doctor as soon as possible.")



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('DIABETES PREDICTION')


    from streamlit_lottie import st_lottie

    path = r"C:\Users\Shoban\Downloads\animation_lmi7n9s7.json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=400,
              width=600,
              speed=1,
              loop=True,
              quality='high',
              key='Diabetes'
              )

    st.markdown(
        '_This_ _purpose_ _of_ _this_  _application_ _is_ _to_ _help_ _doctors_ _diagnose_ , _if_ _a_ _patient_ _has_ _a_ _Diabetes_ _Disease_ _or_ _may_ _have_ _it_ _after_ _sometime_ _using_ _few_ _details_ _about_ _their_ _health_')

    st.markdown('Please **Enter _the_ _below_ details** to know the results -')


    Pregnancies= st.slider('Number of Pregnancies', 0, 17, 1)
        


    Glucose = st.slider('Glucose Level' , 1,200,1)


    BloodPressure = st.slider('Blood Pressure value', 0,180,5)

    SkinThickness = st.slider('Skin Thickness value' , 0,100 , 1)


    Insulin = st.slider('Insulin Level' , 0,900, 10)

    BMI = st.slider('BMI value' , 0,70,1)

    min_value = 0.0
    max_value = 3.0
    default_value = 0.1
    DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function value", min_value, max_value, default_value, step=0.1)

    # DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')


    Age = st.text_input(label='Age of The Patient')
    converted_value = None

    # Convert the input to an integer if it's not empty
    if Age:
        try:
            converted_value = int(Age)
        except ValueError:
            st.warning("Invalid input. Please enter a valid integer.")

    # code for Prediction

    # creating a button for Prediction
    input_data = ([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])


    if st.button('Diabetes Test Result'):
        for values in input_data:
            if values[7] =="":
                st.warning("Plz Enter Age ")
            else:
                # change the input data to a numpy array
                input_data_as_numpy_array = np.asarray(input_data)

                # reshape the numpy array as we are predicting for only on instance
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

                diab_prediction = diabetes_model.predict(input_data_reshaped)

                if (diab_prediction[0] == 1):
                    st.success("The Patient either have \' Diabetes  \' or are likely to have it. Please visit the doctor as soon as possible.")

                    # st.success('The person is diabetic')
                    import json
                    import streamlit as st
                    from streamlit_lottie import st_lottie

                    # path = r"C:\Users\Shoban\Downloads\animation_lmhf84b8 (1).json"
                    path = r"C:\Users\Shoban\Downloads\animation_lmi7vboo.json"

                    with open(path, "r") as file:
                        url = json.load(file)

                    st_lottie(url,
                              reverse=True,
                              height=00,
                              width=200,
                              speed=1,
                              loop=True,
                              quality='high',
                              key='Sad'
                              )




                else:
                    st.success('Hurray! You are diabetes FREE.')
                    import json
                    import streamlit as st
                    from streamlit_lottie import st_lottie

                    # path = r"C:\Users\Shoban\Downloads\animation_lmhf84b8 (1).json"
                    path = r"C:\Users\Shoban\Downloads\animation_lmi82wn3.json"

                    with open(path, "r") as file:
                        url = json.load(file)

                    st_lottie(url,
                              reverse=True,
                              height=00,
                              width=200,
                              speed=1,
                              loop=True,
                              quality='high',
                              key='Happy'
                              )



# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # Define selected_options as an empty list before using it
    st.title('HEART DISEASE PREDICTION')

    from streamlit_lottie import st_lottie

    path = r"C:\Users\Shoban\Downloads\animation_lmhf3r1p (1).json"
    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=400,
              width=600,
              speed=1,
              loop=True,
              quality='high',
              key='Heart'
              )


    st.markdown('_This_ _purpose_ _of_ _this_  _application_ _is_ _to_ _help_ _doctors_ _diagnose_ , _if_ _a_ _patient_ _has_ _a_ _Heart_ _Disease_ _or_ _may_ _have_ _it_ _after_ _sometime_ _using_ _few_ _details_ _about_ _their_ _health_')

    st.markdown('Please **Enter _the_ _below_ details** to know the results -')
    # getting the input data from the user


    age = st.text_input(label='Age')


    converted_value = None

    # Convert the input to an integer if it's not empty
    if age:
        try:
            converted_value = int(age)
        except ValueError:
            st.warning("Invalid input. Please enter a valid integer.")

    # Display the converted integer value or a message if no input provided
    # if converted_value is not None:
    #     st.write("Converted Integer:", converted_value)
    # elif age == '':
    #     st.write("Please enter an integer.")
    # st.write(type(age))
    gender_ls = ['Male', 'Female']
    sex = st.selectbox('Gender', gender_ls)


    cp_ls = ['Typical Angina', 'Atypical Angina', 'Non-anginal pain', 'Asymptomatic']
    cp = st.selectbox('Chest pain Type', cp_ls)

    trestbps = st.slider('Resting Blood Pressure', 0, 220, 120)
    
    chol = st.slider('Serum Cholestoral in mg/dl', 0, 600, 150)

    fbs_ls = ['fasting blood sugar > 120 mg/dl', 'fasting blood sugar < 120 mg/dl']
    fbs = st.selectbox('Fasting Blood Sugar (>126 mg/dL signals diabetes)', fbs_ls)

    restecg_ls = ['Nothing to Note', 'ST-T Wave abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting ECG(Electrocardiographic)', restecg_ls)

    thalach = st.slider('Maximum Heart Rate Achieved (Thalach)', 0, 250, 100)

    exang_ls = ['Yes', 'No']
    exang = st.selectbox('Exercise Induced Angina', exang_ls)

    oldpeak = st.text_input(label='Oldpeak: ST Depression induced by exercise relative to rest (0-6)')
    if oldpeak:
        try:
            converted_value = int(oldpeak)
        except ValueError:
            st.warning("Invalid input. Please enter a valid integer.")


    slope_ls = ['Unslopping: Better heart rate with exercise', 'Flatsloping: Minimal change',
                'Downslopings: Signs of unhealthy heart']
    slope = st.selectbox('Slope of Peak exercise ST Segment', slope_ls)

    ca_ls = [1, 2, 3, 4]
    ca = st.selectbox('Number of Major vessels colored by flourosopy', ca_ls)

    thal_ls = ['No blood flow in some area', 'Normal blood', 'Reversible direct']
    thal = st.selectbox('Thalium Stress result', thal_ls)


    def SEX(sex):
        if sex == 'Male':
            return 1
        if sex == "Female":
            return 0


    def CP(cp):


        if cp == 'Typical Angina':
            return 1
        if cp == 'Atypical Angina':
            return 2
        if cp == 'Non-anginal pain':
            return 3
        if cp == 'Asymptomatic':
            return 4

    def FBS(fbs):
        if fbs == 'fasting blood sugar > 120 mg/dl':
            return 1
        else:
            return 0

    def RESTECG(restecg):

        if restecg =="Nothing to Note":
            return 0
        if restecg == 'ST-T Wave abnormality':
            return 1
        if restecg =="Left Ventricular Hypertrophy":
            return 2
    def EXANG(exang):
        if exang == "Yes":
            return 1
        else:
            return 0


    def SLOPE(slope):
        if slope =='Unslopping: Better heart rate with exercise':
            return 1
        if slope == 'Flatsloping: Minimal change':
            return 2
        if slope == 'Downslopings: Signs of unhealthy heart':
            return 3
    def CA(ca):
        if ca == 1:
            return 1
        if ca == 2:
            return 2
        if ca == 3:
            return 3
        if ca == 4:
            return 4

    def THAl(thal):
        if thal == 'No blood flow in some area':
            return 1
        if thal == 'Normal blood':
            return 2
        if thal == 'Reversible direct':
            return 3



    input_data = (age, SEX(sex), CP(cp), trestbps, chol, FBS(fbs), RESTECG(restecg), thalach, EXANG(exang), oldpeak,
                  SLOPE(slope), CA(ca), THAl(thal))



    heart_diagnosis = ''

    try:
        age = int(age)
    except:
        st.write("")
    try:
        oldpeak = int(oldpeak)
    except:
        st.write("")

    if st.button('Heart Disease Test Result'):

        if (age == '') | (type(age) == str):
            if age == '':
                st.warning("Age is empty plz add age")
            elif type(age) == str:
                st.warning("Invalid input. Please enter age in a  valid integer.")

        elif (oldpeak == '') | (type(oldpeak) == str):
            if oldpeak == '':
                st.warning("oldpeak is empty plz add a num between 0 to 6")
            elif type(oldpeak) == str:
                st.warning("Invalid input. Please enter oldpeak in a  valid integer.")
        else:
            input_data_as_numpy_array = np.asarray(input_data)

                        # reshape the numpy array as we are predicting for only on instance
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            prediction = heart_model.predict(input_data_reshaped)


            if (prediction[0] == 1):
                st.success(
                    "The Patient either have \' Heart Disease \' or are likely to have it. Please visit the doctor as soon as possible.")


                from streamlit_lottie import st_lottie

                path = r"C:\Users\Shoban\Downloads\animation_lmi7vboo.json"

                with open(path, "r") as file:
                    url = json.load(file)

                st_lottie(url,
                          reverse=True,
                          height=00,
                          width=200,
                          speed=1,
                          loop=True,
                          quality='high',
                          key='Sad'
                          )

            else:

                st.success('Hurray! The person does not have any heart disease')

                from streamlit_lottie import st_lottie

                path = r"C:\Users\Shoban\Downloads\animation_lmi82wn3.json"

                with open(path, "r") as file:
                    url = json.load(file)

                st_lottie(url,
                          reverse=True,
                          height=00,
                          width=200,
                          speed=1,
                          loop=True,
                          quality='high',
                          key='Happy'
                          )

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("PARKINSON'S DISEASE PREDICTION")


    from streamlit_lottie import st_lottie

    path = r"C:\Users\Shoban\Downloads\animation_lmi6p7es.json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=00,
              width=400,
              speed=1,
              loop=True,
              quality='high',
              key='Parkinson'
              )

    st.markdown(
        '_This_ _purpose_ _of_ _this_  _application_ _is_ _to_ _help_ _doctors_ _diagnose_ , _if_ _a_ _patient_ _has_ _a_ _Parkinson\'s_ _Disease_ _or_ _may_ _have_ _it_ _after_ _sometime_ _using_ _few_ _details_ _about_ _their_ _health_')

    st.markdown('Please **Enter _the_ _below_ details** to know the results -')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col1:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col2:
        RAP = st.text_input('MDVP:RAP')

    with col3:
        PPQ = st.text_input('MDVP:PPQ')

    with col4:
        DDP = st.text_input('Jitter:DDP')

    with col1:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col2:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col3:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col4:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col1:
        APQ = st.text_input('MDVP:APQ')

    with col2:
        DDA = st.text_input('Shimmer:DDA')

    with col3:
        NHR = st.text_input('NHR')

    with col4:
        HNR = st.text_input('HNR')

    with col1:
        RPDE = st.text_input('RPDE')

    with col2:
        DFA = st.text_input('DFA')

    with col3:
        spread1 = st.text_input('spread1')

    with col4:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # Initialize variables and store them in a dictionary
    if st.button("Parkinson's Test Result"):
        input_data = {
            'fo': fo,
            'fhi': fhi,
            'flo': flo,
            'Jitter_percent': Jitter_percent,
            'Jitter_Abs': Jitter_Abs,
            'RAP': RAP,
            'PPQ': PPQ,
            'DDP': DDP,
            'Shimmer': Shimmer,
            'Shimmer_dB': Shimmer_dB,
            'APQ3': APQ3,
            'APQ5': APQ5,
            'APQ': APQ,
            'DDA': DDA,
            'NHR': NHR,
            'HNR': HNR,
            'RPDE': RPDE,
            'DFA': DFA,
            'spread1': spread1,
            'spread2': spread2,
            'D2': D2,
            'PPE': PPE
        }

        # Check for missing or empty values in user input
        missing_inputs = []
        for variable_name, variable_value in input_data.items():
            if variable_value == "":
                missing_inputs.append(variable_name)

        if missing_inputs:
            st.warning(f"The following fields are missing or empty: {', '.join(missing_inputs)}")
        else:
            # Convert user inputs to floats before using them in the model
            input_data = {variable_name: float(variable_value) for variable_name, variable_value in input_data.items()}

            # Make the prediction using your machine learning model
            parkinsons_prediction = parkinson_model.predict([list(input_data.values())])

            if parkinsons_prediction[0] == 1:
                st.success("The Patient  either have \' Parkinson\'s Disease \' or are likely to have it. Please visit the doctor as soon as possible.")
                import json
                import streamlit as st
                from streamlit_lottie import st_lottie

                # path = r"C:\Users\Shoban\Downloads\animation_lmhf84b8 (1).json"
                path = r"C:\Users\Shoban\Downloads\animation_lmi7vboo.json"

                with open(path, "r") as file:
                    url = json.load(file)

                st_lottie(url,
                          reverse=True,
                          height=00,
                          width=200,
                          speed=1,
                          loop=True,
                          quality='high',
                          key='Sad'
                          )

            else:
                st.success("Hurray! The person does not have Parkinson's disease")


                from streamlit_lottie import st_lottie

                path = r"C:\Users\Shoban\Downloads\animation_lmi82wn3.json"

                with open(path, "r") as file:
                    url = json.load(file)

                st_lottie(url,
                          reverse=True,
                          height=00,
                          width=200,
                          speed=1,
                          loop=True,
                          quality='high',
                          key='Happy'
                          )


if selected == "Corona Virus Prediction":
    st.title('CORONA VIRUS PREDICTION ')


    from streamlit_lottie import st_lottie
    path = r"C:\Users\Shoban\Downloads\animation_lmhyv0rl.json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=400,
              width=600,
              speed=1,
              loop=True,
              quality='high',
              key='Covid19'
              )

    st.markdown(
        '_This_ _purpose_ _of_ _this_  _application_ _is_ _to_ _help_ _doctors_ _diagnose_ , _if_ _a_ _patient_ _has_ _a_ _Corona_ _Virus_ _Disease_ _or_ _may_ _have_ _it_ _after_ _sometime_ _using_ _few_ _details_ _about_ _their_ _health_')

    st.markdown('Please **Enter _the_ _below_ details** to know the results -')

    import numpy as np

    # Define options for select boxes
    gender_options = ['Male', 'Female']
    patient_type_options = ['Not hospitalized', 'Hospitalized']
    intubed_options = ['The patient used ventilator', 'The patient did not use ventilator']
    pneumonia_options = ['Patient already has air sac inflammation', 'Patient does not have air sac inflammation']
    pregnancy_options = ['The patient is pregnant', 'The patient is not pregnant']
    diabetes_options = ['The patient has diabetes', 'The patient has no diabetes']
    copd_options = ['The patient has COPD', 'The patient has not COPD']
    asthma_options = ['The patient has asthma', 'The patient has no asthma']
    inmsupr_options = ['The patient is immunosuppressed', 'The patient is not immunosuppressed']
    hypertension_options = ['The patient has hypertension', 'The patient does not have hypertension']
    other_disease_options = ['The patient has other diseases', 'The patient has no other diseases']
    cardiovascular_options = ['The patient has heart or blood vessel diseases',
                              'The patient has no heart or blood vessel diseases']
    obesity_options = ['The patient is obese', 'The patient is not obese']
    renal_chronic_options = ['The patient has chronic renal disease', 'The patient does not have chronic renal disease']
    tobacco_options = ['The patient is a tobacco user', 'The patient is not a tobacco user']
    contact_other_covid_options = ['The patient has contacted another COVID-19 patient',
                                   'The patient has not contacted another COVID-19 patient']
    icu_options = ['The patient had been admitted to an Intensive Care Unit (ICU)',
                   'The patient had not been admitted to an Intensive Care Unit (ICU)']

    # Create select boxes for user input
    sex = st.selectbox('Gender', gender_options)
    patient_type = st.selectbox('Patient Type', patient_type_options)
    intubed = st.selectbox('Intubed', intubed_options)
    pneumonia = st.selectbox('Pneumonia', pneumonia_options)
    age = st.number_input('Age', min_value=0, max_value=150)
    pregnancy = st.selectbox('Pregnancy', pregnancy_options)
    diabetes = st.selectbox('Diabetes', diabetes_options)
    copd = st.selectbox('COPD', copd_options)
    asthma = st.selectbox('Asthma', asthma_options)
    inmsupr = st.selectbox('Immunosuppressed', inmsupr_options)
    hypertension = st.selectbox('Hypertension', hypertension_options)
    other_disease = st.selectbox('Other Diseases', other_disease_options)
    cardiovascular = st.selectbox('Cardiovascular', cardiovascular_options)
    obesity = st.selectbox('Obesity', obesity_options)
    renal_chronic = st.selectbox('Chronic Renal Disease', renal_chronic_options)
    tobacco = st.selectbox('Tobacco User', tobacco_options)
    contact_other_covid = st.selectbox('Contact with COVID-19 Patient', contact_other_covid_options)
    icu = st.selectbox('ICU Admission', icu_options)


    # Function to convert user inputs to numerical values
    def convert_input_to_numeric(value, options):
        return options.index(value) + 1


    # Convert user inputs to numerical values
    sex_numeric = convert_input_to_numeric(sex, gender_options)
    patient_type_numeric = convert_input_to_numeric(patient_type, patient_type_options)
    intubed_numeric = convert_input_to_numeric(intubed, intubed_options)
    pneumonia_numeric = convert_input_to_numeric(pneumonia, pneumonia_options)
    pregnancy_numeric = convert_input_to_numeric(pregnancy, pregnancy_options)
    diabetes_numeric = convert_input_to_numeric(diabetes, diabetes_options)
    copd_numeric = convert_input_to_numeric(copd, copd_options)
    asthma_numeric = convert_input_to_numeric(asthma, asthma_options)
    inmsupr_numeric = convert_input_to_numeric(inmsupr, inmsupr_options)
    hypertension_numeric = convert_input_to_numeric(hypertension, hypertension_options)
    other_disease_numeric = convert_input_to_numeric(other_disease, other_disease_options)
    cardiovascular_numeric = convert_input_to_numeric(cardiovascular, cardiovascular_options)
    obesity_numeric = convert_input_to_numeric(obesity, obesity_options)
    renal_chronic_numeric = convert_input_to_numeric(renal_chronic, renal_chronic_options)
    tobacco_numeric = convert_input_to_numeric(tobacco, tobacco_options)
    contact_other_covid_numeric = convert_input_to_numeric(contact_other_covid, contact_other_covid_options)
    icu_numeric = convert_input_to_numeric(icu, icu_options)



    # Create a feature vector from user inputs
    feature_vector = [sex_numeric, patient_type_numeric, intubed_numeric, pneumonia_numeric, age, pregnancy_numeric,
                      diabetes_numeric, copd_numeric, asthma_numeric, inmsupr_numeric, hypertension_numeric,
                      other_disease_numeric, cardiovascular_numeric, obesity_numeric, renal_chronic_numeric,
                      tobacco_numeric, contact_other_covid_numeric, icu_numeric]

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(feature_vector)

            # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    covid_diagnosis = ''
    if st.button("Covid Test Result"):
        prediction = covid_model.predict(input_data_reshaped)
            # print(prediction)


        if (prediction[0] == 1):
            st.success('The Patient either have Covid +ve or are likely to have it. Please visit the doctor as soon as possible.')

            from streamlit_lottie import st_lottie

            path = r"C:\Users\Shoban\Downloads\animation_lmi7vboo.json"

            with open(path, "r") as file:
                url = json.load(file)

            st_lottie(url,
                      reverse=True,
                      height=00,
                      width=200,
                      speed=1,
                      loop=True,
                      quality='high',
                      key='Sad'
                      )

        else:
            st.success('Hurray! Patient have covid -ve')

            from streamlit_lottie import st_lottie

            path = r"C:\Users\Shoban\Downloads\animation_lmi82wn3.json"

            with open(path, "r") as file:
                url = json.load(file)

            st_lottie(url,
                      reverse=True,
                      height=00,
                      width=200,
                      speed=1,
                      loop=True,
                      quality='high',
                      key='Happy'
                      )

if selected == "Pheumonia Detection":


    # Streamlit UI
    st.title("PNEUMONIA DETECTION ")

    from streamlit_lottie import st_lottie

    path = r"C:\Users\Shoban\Downloads\animation_lmi6b7lk.json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=00,
              width=400,
              speed=1,
              loop=True,
              quality='high',
              key='Pheunomonia'
              )

    st.markdown(
        "_The_ _purpose_ _of_ _this_ _application_ _is_ _to_ _help_ _doctors_ _diagnose_ _if_ _a_ _patient_ _has_ _Pneumonia_ _or_ _may_ _have_ _it_ _after_ _some_ _time_ _using_ _X-Ray_ _detection_.")

    st.markdown('Please **Upload a chest X-ray image** to know the results -')
    # Define a confidence threshold for pneumonia predictions
    pneumonia_threshold = 0.5  # Adjust this threshold as needed



    # Upload image
    uploaded_image = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to a format suitable for prediction
        image = np.array(image)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Load and preprocess the image for pneumonia detection
        img_pneumonia = cv2.resize(img, (150, 150))
        img_pneumonia = img_pneumonia / 255.0
        img_pneumonia = np.expand_dims(img_pneumonia, axis=0)  # Add the batch dimension

        # Make a prediction using the pneumonia detection model
        prediction_pneumonia = pneumonia_model.predict(img_pneumonia)

        # Interpret the pneumonia prediction
        pneumonia_confidence = float(prediction_pneumonia[0])

        if pneumonia_confidence >= pneumonia_threshold:
            st.success('The Patient either have PNEUMONIA or are likely to have it. Please visit the doctor as soon as possible.')
            # st.write("Prediction: PNEUMONIA")
            st.write(f"Confidence: {pneumonia_confidence:.2f}")


            from streamlit_lottie import st_lottie

            # path = r"C:\Users\Shoban\Downloads\animation_lmhf84b8 (1).json"
            path = r"C:\Users\Shoban\Downloads\animation_lmi7vboo.json"

            with open(path, "r") as file:
                url = json.load(file)

            st_lottie(url,
                      reverse=True,
                      height=00,
                      width=200,
                      speed=1,
                      loop=True,
                      quality='high',
                      key='Sad'
                      )

        else:
            st.success("Hurray! X-ray is  NORMAL")

            from streamlit_lottie import st_lottie

            # path = r"C:\Users\Shoban\Downloads\animation_lmhf84b8 (1).json"
            path = r"C:\Users\Shoban\Downloads\animation_lmi82wn3.json"

            with open(path, "r") as file:
                url = json.load(file)

            st_lottie(url,
                      reverse=True,
                      height=00,
                      width=200,
                      speed=1,
                      loop=True,
                      quality='high',
                      key='Happy'
                      )
if selected == "Any Feed back":

    import datetime

    from streamlit_lottie import st_lottie

    path = r"C:\Users\Shoban\Downloads\animation_lmk56iph.json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=400,
              width=600,
              speed=1,
              loop=True,
              quality='high',
              key='feedback'
              )


    # Create a feedback form
    st.subheader("Feedback Form")
    name = st.text_input("Enter your name:")
    feedback_text = st.text_area("Enter your feedback here:")
    submit_button = st.button("Submit Feedback")

    # Define the feedback file path (modify this path as needed)
    feedback_file_path = "feedback.txt"

    # Handle user input and save feedback to a text file
    if submit_button:
        # Capture user feedback and timestamp
        user_feedback = feedback_text
        timestamp = datetime.datetime.now()
        try:
            # Save the feedback and timestamp to a text file
            with open(r"C:\Users\Shoban\Desktop\ApnaDoctor\FeedBackFromUser.txt.txt", "a") as feedback_file:
                feedback_file.write(f"\nUser Name: {name}\n")
                feedback_file.write(f"Timestamp: {timestamp}\n")
                feedback_file.write(f"Feedback: {user_feedback}\n")
                feedback_file.write("\n")  # Add a separator between feedback entries
                feedback_file.write("-"*50)
            # Acknowledge and confirm feedback submission
            st.success("Thank you for your feedback! We appreciate it.")
        except:
            st.success("Thank you for your feedback! We appreciate it.")
if selected == 'About me':

    from streamlit_lottie import st_lottie

    path = r"C:\Users\Shoban\Downloads\animation_lmi6ye9k.json"

    with open(path, "r") as file:
        url = json.load(file)

    st_lottie(url,
              reverse=True,
              height=400,
              width=800,
              speed=1,
              loop=True,
              quality='high',
              key='About me'
              )


    # About Me
    st.header("About Me")
    st.write("I am a passionate Data Scientist. I love to learn and build cool projects.")

    # Education
    st.header("Education")
    st.subheader("Currently studying in 3rd year of  Bachelor's degree in Information Technology ")
    st.write("In MBBS Campus Dadu , University Of Sindh")
    st.write("CGPA in Last year 3.61")


    # Skills
    st.header("Skills")
    st.write("Programming Languages: Python, C++")
    st.write("Data Analysis: Pandas, NumPy" , "Matplotlib","Seaborn")
    st.write("Machine Learning: Scikit-learn, TensorFlow")
    st.write("Web scrapping: BeautifulSoup, Selenium")

    # Contact Information
    st.header("Contact Information")
    st.write("Gmail: shobyaleegmail.com")
    st.write("P:Num: +923093349289")
    st.write("Github :https://www.github.com/ShabanAliAbbasi")
    st.write("Fiver: https://www.fiverr.com/users/shaban_ali52")
    st.write("LinkedIn: https://www.linkedin.com/in/shaban-ali-abbasi-744b9a287")
    st.write("Facebook: https://www.facebook.com/shaban.abbasi.1029")
    st.write("Instagram: https://www.instagram.com/shaban.abbasii")


    # Footer
    st.write("Thank you for visiting my portfolio!")
