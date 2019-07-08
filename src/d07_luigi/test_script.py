def square(n):
	return n*n

# load credentials
import json
import os

filename = os.path.expanduser('~') + '/.psql_credentials.json'
with open(filename) as f:
	data = json.load(f)
	print(data["host"])