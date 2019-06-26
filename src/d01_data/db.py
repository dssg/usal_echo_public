import pandas as pd
import sqlalchemy
import yaml


def get_sqlalchemy_engine():
    with open("../conf/local/db.yaml") as f:
        conf = yaml.safe_load(f)

    connection_string = "postgresql://{}:{}@{}/{}".format(
        conf["user"], conf["pw"], conf["host"], conf["DB"]
    )
    conn = sqlalchemy.create_engine(connection_string)

    return conn


def read_table(name):
    query = f"select * from {name}"
    conn = get_sqlalchemy_engine()
    return pd.read_sql(query, conn)
