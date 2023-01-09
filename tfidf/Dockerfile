FROM python:3.8

WORKDIR /app

COPY . .

RUN apt-get -y update && \
    apt-get -y upgrade

RUN pip install -U pip

RUN pip install numpy==1.21.1

CMD ["python", "assignment.py", "tests.py"] 