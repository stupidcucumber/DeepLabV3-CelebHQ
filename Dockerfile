FROM python:3.10-bullseye

WORKDIR /app
COPY requirements.txt ./

RUN pip3 install --no-cache -r requirements.txt

COPY . ./

ENTRYPOINT ["python", "train.py", "--data" , "datasets/CelebAMask-HQ-1/celeb.csv", "--epochs", "20", "--batch-size", "16", "--mapping", "datasets/CelebAMask-HQ-1/mapping.json"]