import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
apptitle = 'DSSI Toy App'

st.set_page_config(page_title=apptitle, layout='wide')

st.title('My First Streamlit Application')
st.write('Reference: https://docs.streamlit.io/en/stable/api.html#display-data')
st.balloons() 

# Load Iris dataset
st.subheader('**Iris Data**')
db = datasets.load_iris()

df = pd.DataFrame(db.data, columns=db.feature_names)
df['species'] = pd.Categorical.from_codes(db.target, db.target_names)

col1, col2 = st.columns([2,1])
with col1:
    # Display dataframe as an interactive table
    st.dataframe(df, use_container_width=True)
with col2:
    # Plot petal-length distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    if 1==1: # Evaluate True to show plot
        df['petal length (cm)'].hist(bins = 10, ax=ax)
        fig.suptitle("Petal Length Distribution")
        st.pyplot(fig)
