FROM python:3.7-stretch

RUN apt-get update
RUN apt-get -y install libgdcm-tools
RUN pip install boto3
RUN pip install sqlalchemy
RUN pip install temp
RUN pip install pandas
RUN pip install psycopg2

ADD metadata_images.py /
ADD credentials /root/.aws/
ADD credentials_db.json /

CMD ["python", "./metadata_images.py"]
