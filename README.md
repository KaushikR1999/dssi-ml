## Data Product Development and Deployment with Streamlit
### Requirements
1. Setup Github account
2. Install Python 3.11+
3. Create virtual environment
```
python -m venv venv
```
4. Activate virtual environment
* Windows
```
venv\Scripts\activate
```
* Linux/MacOS
```
source venv/bin/activate
```
5. Install required packages:
```
pip install -r requirements.txt
```
### Prelude: Try Streamlit
1. Create toy application with Streamlit.
2. Push repository to GitHub.
3. Deploy on Streamlit community cloud.  

Sample application code: [toy-app.py](toy-app.py)
### Step 1: Train and Save Model
1. Load the bundled Iris dataset directly from scikit-learn.
2. Use the training and model registry scripts to automate model training and persistence.
3. Run the training module to load scikit-learn's Iris dataset and train an Iris species classifier:
```
python -m src.training
```
Sample training script: [training.py](src/training.py)  
Sample model registry script: [model_registry.py](src/model_registry.py)
### Step 2: Create App and Load Model
1. Develop an inference script to serve predictions.
2. Create an Iris classifier application with Streamlit that predicts a species from flower measurements.  

Sample inference script: [inference.py](src/inference.py)  
Sample application code: [app.py](app.py)
### Step 3: Test App Locally
Run and test the application locally:
```
streamlit run app.py
```
### Step 4: Deploy App Online
1. Commit repository to GitHub.
2. Deploy on Streamlit community cloud.
