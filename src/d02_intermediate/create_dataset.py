import sys
sys.path.append('instance_filters/')

import filter_machines

'''
This is a high-level script which runs all filtering of instances needed to create the dataset.
Summaries of each filtering steps are included below. Further details can be found in the corresponding script
'''

filter_machines.filter_by_machine_type()
