import pandas as pd
import sqlalchemy


def get_sqlalchemy_engine():
    # TODO: figure out convention for configurations in this project structure
    with open("../conf/local/db.json") as f:
        conf = eval(f.read())

    connection_string = "postgresql://{}:{}@{}/{}".format(
        conf["user"], conf["pw"], conf["host"], conf["DB"]
    )
    conn = sqlalchemy.create_engine(connection_string)

    return conn


def read_table(name):
    query = f"select * from {name}"
    conn = get_sqlalchemy_engine()
    return pd.read_sql(query, conn)
