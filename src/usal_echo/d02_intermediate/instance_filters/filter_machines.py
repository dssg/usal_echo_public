from usal_echo.d00_utils.db_utils import dbReadWriteClean, dbReadWriteViews
from usal_echo.d00_utils.log_utils import *

"""
This script contains filter_by_machine_type(), which creates
the tables views.machines_all_bmi and views.machines_new_bmi,
which contain the following columns: studyidk, equipment, bmi

machines_all_bmi contains machine type + bmi for every study in db
machines_new_bmi contains machine type + bmi for studies taken on
    "new" machines, i.e. ECOEPIQ2, EPIQ7-1, ECOIE33, AFFINITI_1, AFFINITI_2
"""


def filter_by_machine_type():

    logger = setup_logging(__name__, "filter_machines.py")

    io_clean = dbReadWriteClean()
    io_views = dbReadWriteViews()

    # get relevant table, drop unnecessary columns
    summ_df = io_clean.get_table("dm_spain_view_study_summary")
    cols_to_drop = [
        "age",
        "gender",
        "patientweight",
        "patientheight",
        "findingcode",
        "findingcodetext",
        "conclusion",
        "reasonforstudy",
        "studylocation",
        "modality",
    ]
    machine_df = summ_df.drop(labels=cols_to_drop, axis=1)
    io_views.save_to_db(machine_df, "machines_all_bmi")

    # filter out old machines
    machines_to_keep = ["ECOEPIQ2", "EPIQ7-1", "ECOIE33", "AFFINITI_1", "AFFINITI_2"]
    machine_new_df = machine_df[machine_df["equipment"].isin(machines_to_keep)]
    io_views.save_to_db(machine_new_df, "machines_new_bmi")
    logger.info("New tables created: views.machines_all_bmi and views.machines_new_bmi")
