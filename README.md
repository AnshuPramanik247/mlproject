# Student Score Predictor

## Deploy to AWS Elastic Beanstalk

Create a Python 3.11 Elastic Beanstalk environment and deploy the repository root.
The Beanstalk WSGI entry point is configured as `application:application` in
`.ebextensions/python.config`. Keep `artifacts/models.pkl` and
`artifacts/preprocessor.pkl` in the deployment source bundle; predictions need
both files.

The versions of NumPy, pandas, scikit-learn, dill, and Flask in
`requirements.txt` are pinned so the saved model can be loaded consistently.
These versions work with the project's Python 3.8 environment as well as Python
3.11. Install the project dependencies before running locally:

```powershell
.\venv\python.exe -m pip install -r requirements.txt
.\venv\python.exe application.py
```