import luigi
from luigi.s3 import S3Target
from luigi.postgres import PostgresTarget 

from ..d00_utils.db_utils import *
from ..d01_data.ingestion_dcm import ingest_dcm
from ..d01_data.ingestion_xtdb import ingest_xtdb
from ..d02_intermediate.clean_dcm import clean_dcm
from ..d02_intermediate.clean_xtdb import clean_tables

# Questions
# credentials - make a super class according to GitHub
# Then sub-classes, e.g. IngestDCM(), will be a sub-class of parents QueryInfo and Luigi
#       how to do this properly?
# How to get credentials from dbReadWriteData as opposed to redoing my own way
# Multiple inheritance example: https://www.pythonforbeginners.com/super/working-python-super-function

'''
def get_postgres_credentials():
    	'''
#	Access postgres credentials from .json in user root directory
#	Outputs host postgres server address, db user/password, db name
	'''
	filename = os.path.expanduser('~') + '/.psql_credentials.json'
	with open(filename) as f:
		data = json.load(f)
		return data["host"], data["user"], data["psswd"], data["database"]
'''

class QueryInfo():
    def __init__(self, table, update_id):
        self.table = table
        self.update_id = update_id

        # Option A - isn't this redundant, declaring variables already declared in parent?
        io = dbReadWriteData()
        creds = io.credentials
        self.host = cred["host"] #, etc.

        # Option B - json load
        filename = os.path.expanduser('~') + '/.psql_credentials.json'
        with open(filename) as f:
            data = json.load(f)
            self.host = data["host"]
            self.user = data["user"]
            self.psswd = data["psswd"]
            self.database = data["database"]

    def output(self):
        '''
        Returns a PostgresTarget representing the executed query
        '''
        return PostgresTarget(host=self.host, database=self.database, user=self.user \
                            password=self.psswd, table=self.table, update_id=self.update_id)

                                

    

class S3Bucket(luigi.ExternalTask): # ensure connection to S3 bucket

	def output(self):
		return luigi.S3Target('s3://cibercv/')
	# e.g. https://stackoverflow.com/questions/33332058/luigi-pipeline-beginning-in-s3


class IngestDCM(luigi.Task):

	def requires(self):
		return S3Bucket()

	def output(self):
            host, user, psswd, database = get_postgres_credentials()
	    table = 'raw.metadata'
            update_id = 'ingest'
            return PostgresTarget(host, database, user, password, table, update_id)
	
        def run(self):
		ingest_dcm()


class CleanDCM(luigi.Task):

	def requires(self):
    	    return IngestDCM()
		
	def output(self):
	    host, user, password, database = get_postgres_credentials()
            table = 'clean.meta_lite'
            update_id = 'clean'
            return PostgresTarget(host, database, user, password, table, update_id)
    	    # https://luigi.readthedocs.io/en/stable/api/luigi.contrib.postgres.html
            # e.g. https://vsupalov.com/luigi-query-postgresql/


class IngestXTDB(luigi.Task):
    '''
    Assumes that all tables exist if a_measgraphic exists.
    This assumption is based upon ingest_xtdb() populates all tables at once from csv's in cibercv db
    '''

	def requires(self):
            return S3Bucket()

	def output(self): # output tables
	    # table, update_id
            table = 'raw.A_measgraphic'
            update_id = 'ingest'
	    return PostgresTarget(host, database, user, password, table, update_id)

	def run(self):
	    return ingest_xt_db()


class CleanXTDB(luigi.Task):

	def requires (self):
	    return IngestXTDB()

	def output(self): 
	    host, user, password, database = get_postgres_credentials()
            table = 'clean.a_measgraphic'
            update_id = 'clean'
	    return PostgresTarget(host, database, user, password, table, update_id)

	def run(self):
	    return clean_tables()






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

