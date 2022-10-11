FROM python:3.9

COPY ./requirements.txt /usr/requirements.txt

WORKDIR /usr

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./src /usr/src
COPY ./models /usr/models


CMD [ "uvicorn", "--app-dir", "src/", "main:app" ]

