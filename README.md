# Loan predictor
## Streamlit Demo

This project demonstrates how to present machine learning solution as a web application using [Streamlit](https://www.streamlit.io/) framework. The data used in this repo is the [Loan approval dataset](https://www.kaggle.com/datasets/anishdevedward/loan-approval-dataset) from Kaggle.

Try app [here](https://titanic.streamlit.app/)!

## Files

- `app.py`: streamlit app file
- `model.py`: script for generating models and making predictions
- `loan_aproval.csv`: data file for training and testing
- `model` directory: saved model files
- `requirements.txt`: package requirements files
- `smart-loan-predictor.ipynb`: jupyter notebook for model exploration and development

## Run Demo Locally 

### Shell

For directly run streamlit locally in the repo root folder as follows:

>*<font color="red">Warning:</font> 
<br>The requirements.txt file was created for Python 3.12.3. If you have a different version of Python or encounter errors during installation, install the required modules manually.*

#### Linux:
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
For some systems, you may need to use 'python' instead of 'python3'.

#### Windows:
```shell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 to view the app.


## Streamlit Cloud Deployment
 
1. Put your app on GitHub (like this repo)
Make sure it's in a public folder and that you have a `requirements.txt` file.
 
2. Sign into Streamlit Cloud
Sign into share.streamlit.io with your GitHub email address, you need to have access to Streamlit Cloud service.
 
3. Deploy and share!  
Click "New app", then fill in your repo, branch, and file path, choose a Python version (3.12.3 for this demo) and click "Deploy", then you should be able to see your app.


