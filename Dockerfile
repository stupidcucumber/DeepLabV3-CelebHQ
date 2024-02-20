FROM python:3.10-bullseye

WORKDIR /app
COPY requirements.txt ./

RUN pip3 install --no-cache -r requirements.txt

COPY . ./

ENTRYPOINT ["python", "train.py", "--data" , "celeb.csv", "--epochs", "20", "--batch-size", "16", "--mapping", "configs/mapping.json"]