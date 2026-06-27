import streamlit as st
from src.inference import get_prediction

# Initialise session state variable
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

def app_sidebar():
    st.sidebar.header('Flower measurements')
    sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)
    def get_input_features():
        return {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width,
        }
    sdb_col1, sdb_col2 = st.sidebar.columns(2)
    with sdb_col1:
        predict_button = st.sidebar.button("Classify", key="predict")
    with sdb_col2:
        reset_button = st.sidebar.button("Reset", key="clear")
    if predict_button:
        st.session_state['input_features'] = get_input_features()
    if reset_button:
        st.session_state['input_features'] = {}
    return None

def app_body():
    title = '<p style="font-family:arial, sans-serif; color:Black; font-size: 40px;"><b>Iris Species Classifier</b></p>'
    st.markdown(title, unsafe_allow_html=True)
    default_msg = '**Predicted species:** {}'
    if st.session_state['input_features']:
        assessment = get_prediction(**st.session_state['input_features'])
        st.success(default_msg.format(assessment.title()))
    return None

def main():
    app_sidebar()
    app_body()
    return None

if __name__ == "__main__":
    main()
