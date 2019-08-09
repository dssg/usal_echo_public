"""
Script for running full pipeline
"""
import luigi
import os

from luigi.contrib.postgres import PostgresTarget
from luigi import LocalTarget

from d00_utils.db_utils import dbReadWriteData
from d01_data.ingestion_dcm import ingest_dcm
from d01_data.ingestion_xtdb import ingest_xtdb
from d02_intermediate.clean_xtdb import clean_tables
from d02_intermediate.clean_dcm import clean_dcm_meta
from d02_intermediate.filter_instances import filter_all
from d02_intermediate.download_dcm import s3_download_decomp_dcm


def get_postgres_credentials():
    """
    Access postgres credentials from .json in user root directory
    Outputs host postgres server address, db user/password, db name
    
    """
    io = dbReadWriteData()
    creds = io.credentials
    return creds["host"], creds["user"], creds["psswd"], creds["database"]


class S3Bucket(luigi.ExternalTask):  # ensure connection to S3 bucket
    def output(self):
        return luigi.S3Target("s3://cibercv/")
        # Reference: https://stackoverflow.com/questions/33332058/luigi-pipeline-beginning-in-s3


class IngestDCM(luigi.Task):
    """
    Pipeline step of ingesting dicom metadata from .dcm files in database
    Populates postgres db schema raw
    
    """
#    def requires(self):
#        return S3Bucket()

    def output(self):
        host, user, psswd, database = get_postgres_credentials()
        table = "raw.metadata"
        update_id = "ingest"
        return PostgresTarget(host, database, user, psswd, table, update_id)

    def run(self):
        ingest_dcm()


class CleanDCM(luigi.Task):
    """
    Pipeline step of cleaning dicom metadata, i.e. filtering only necessary tags
    Populates postgres db schema clean
    
    """

    def requires(self):
        return IngestDCM()

    def output(self):
        host, user, password, database = get_postgres_credentials()
        table = "clean.meta_lite"
        update_id = "clean"
        return PostgresTarget(host, database, user, password, table, update_id)
        # https://luigi.readthedocs.io/en/stable/api/luigi.contrib.postgres.html
        # e.g. https://vsupalov.com/luigi-query-postgresql/

    def run(self):
        return clean_dcm_meta()


class IngestXTDB(luigi.Task):
    """Ingests Xcelera data from cibercv csv files. 
    
    Populates postgres db schema raw.
    Assumes that all tables have been ingested if raw.A_measgraphic exists.
    This assumption is because ingest_xtdb() populates all tables at once
    """

#    def requires(self):
#        return S3Bucket()

    def output(self):  # output tables
        host, user, password, database = get_postgres_credentials()
        table = "raw.A_measgraphic"
        update_id = "ingest"
        return PostgresTarget(host, database, user, password, table, update_id)

    def run(self):
        return ingest_xtdb()


class CleanXTDB(luigi.Task):
    """Cleans Xcelera data and populates postgres db schema clean.
    
    Assumes that all tables have been cleaned if clean.a_measgraphic exists.
    This assumption is because clean_tables() cleans all tables at once.
    
    """

    def requires(self):
        return IngestXTDB()

    def output(self):
        host, user, password, database = get_postgres_credentials()
        table = "clean.a_measgraphic"
        update_id = "clean"
        return PostgresTarget(host, database, user, password, table, update_id)

    def run(self):
        return clean_tables()


class FilterInstances(luigi.Task):
    def requires(self):
        return [CleanXTDB(), CleanDCM()]

    def output(self):
        host, user, password, database = get_postgres_credentials()
        table = "views.instances_w_labels"
        update_id = "filter"
        return PostgresTarget(host, database, user, password, table, update_id)

    def run(self):
        filter_all()


class DownloadDecompressDCM(luigi.Task):

    downsampleratio = luigi.FloatParameter(default=0.1)
    traintestratio = luigi.FloatParameter(default=0.5)
    tablename = luigi.Parameter(default="instances_w_labels")
    train = luigi.BoolParameter(default=False)
    datadir = luigi.Parameter(default=os.path.expanduser("/home/ubuntu/data/"))

    def requires(self):
        return FilterInstances()

    def output(self):
        if self.train is True:
            dir_name = "train_split{}_downsampleby{}".format(
                int(100 * self.train_test_ratio), int(1 / self.downsample_ratio)
            )
        else:
            dir_name = "test_split{}_downsampleby{}".format(
                int(100 - 100 * self.train_test_ratio), int(1 / self.downsample_ratio)
            )
        return luigi.LocalFileSystem.isdir(
            os.path.join(datadir, "01_raw", dir_name, "raw")
        )

    def run(self):
        return s3_download_decomp_dcm(
            self.downsampleratio, self.traintestratio, self.tablename, self.train
        )


class DCMtoJPG(luigi.Task):

    datadir = luigi.Parameter(default=os.path.expanduser("/home/ubuntu/data/"))
    
    def requires(self):
        return DownloadDecompressDCM()
    
    def output(self):
        return luigi.LocalFileSystem.isdir(
            os.path.join(datadir, "02_intermediate", self.img_dir)
        )    
    
    def run(self):
        img_dir = os.path.join(datadir, "02_intermediate", self.input())
        dcmdir_to_jpgs_for_classification(os.path.join(datadir, "01_raw", self.input), img_dir)


class PredictView(luigi.Task):
    def requires(self):
        return

    def output(self):
        return

    def run(self):
        return


class Pipeline(luigi.WrapperTask):
    """
    Dummy task which triggers entire dependency chain for project
    """

    def requires(self):
        yield DCMtoJPG()
        #yield CleanXTDB()
        

