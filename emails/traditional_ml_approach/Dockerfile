FROM python:3.10.12-slim-buster
COPY . /app/
WORKDIR /app 
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "deploy_streamlit.py"]

