import luigi
from luigi.s3 import S3Target
from luigi.postgres import PostgresTarget # luigi.contrib.postgres?
from luigi import LocalTarget

from test_script import square 
from ..d01_data.ingestion_dcm import ingest_dcm
from ..d01_data.ingestion_xt_db import ingest_xt_db
from ..d02_intermediate.dicom_meta_lite import get_meta_lite


def get_postgres_credentials():
	''' 
	Access postgres credentials from .json in user root directory
	Outputs host postgres server address, db user/password, db name
	'''
	filename = os.path.expanduser('~') + '/.psql_credentials.json'
	with open(filename) as f:
		data = json.load(f)
		return data["host"], data["user"], data["psswd"], data["database"]
		# host = data["host"]
		# user = data["user"]
		# password = data["psswd"]
		# database = data["database"]

# Okay to define this globally?
host, user, password, database = get_postgres_credentials()

class S3Bucket(luigi.ExternalTask): # ensure connection to S3 bucket

	def output(self):
		return luigi.S3Target('s3://cibercv/')
	# e.g. https://stackoverflow.com/questions/33332058/luigi-pipeline-beginning-in-s3


class IngestDicom(luigi.Task):

	def requires(self):
		return S3Bucket()

	def output(self):
		return S3Target('s3://usal-pipeline/ingestion/dicom/metadata.csv')

	def run(self):
		ingest_dcm()


class CreateMetadataTable(luigi.Task):

	def requires(self):
		return IngestDicom()
		
	def output(self):
		# host, user, password, database = get_postgres_credentials()
		#table, update_id
		return PostgresTarget(host, database, user, password, table, update_id, port=None)

		# https://luigi.readthedocs.io/en/stable/api/luigi.contrib.postgres.html
		# e.g. https://vsupalov.com/luigi-query-postgresql/

	def run(self):
		get_meta_lite()


class IngestXcelera(luigi.Task):

	def requires(self): # input tables
		host, user, password, database = get_postgres_credentials()
		# table, update_id -- multiple tables
		return PostgresTarget(host, database, user, password, table, update_id, port=None)

	def output(self): # output tables
		# table, update_id
		return PostgresTarget(host, database, user, password, table, update_id, port=None)
		# return S3Target('s3://usal-pipeline/ingestion/xcelera/study_characteristics.csv')

	def run(self):
		return ingest_xt_db()


class CleanXcelera(luigi.Task):

	def requires (self):
		return IngestXcelera()

	def output(self): 
		# table, update id
		return PostgresTarget(host, database, user, password, table, update_id, port=None)

	def run(self):
		return clean() # get actual function name




# Changes to develop_pipeline.png
# ingestion_s3 --> ingestion_dcm
# ingestion_xcelera arrow out directly to PostgresSQL
# add filtering.py
# transforming.py likely within Zhang et al.
# .py scripts --> function calls?






class PrintNumbers(luigi.Task):
    n = luigi.IntParameter()
 
    def requires(self):
        return []
 
    def output(self):
        return luigi.LocalTarget("numbers_up_to_{}.txt".format(self.n))
 
    def run(self):
        with self.output().open('w') as f:
            for i in range(1, self.n+1):
                f.write("{}\n".format(i))
 
class SquaredNumbers(luigi.Task):
    n = luigi.IntParameter()
 
    def requires(self):
        return [PrintNumbers(n=self.n)]
 
    def output(self):
        return luigi.LocalTarget("squares_up_to_{}.txt".format(self.n))
 
    def run(self):
        with self.input()[0].open() as fin, self.output().open('w') as fout:
            for line in fin:
                n = int(line.strip())
                out = n*n
                print(out)
                fout.write("{}:{}\n".format(n, out))

