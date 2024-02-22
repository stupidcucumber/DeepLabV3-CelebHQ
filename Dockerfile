FROM python:3.10-bullseye

WORKDIR /app
COPY requirements.txt ./

RUN pip3 install --no-cache -r requirements.txt

COPY . ./

ENTRYPOINT ["python", "train.py", "--data" , "datasets/CelebAMask-HQ/celeb.csv", "--mapping", "datasets/CelebAMask-HQ/mapping.json", "--device", "cuda:0"]