import pandas as pd

from d00_utils.db_utils import (
    dbReadWriteClean,
    dbReadWriteViews,
    dbReadWriteMeasurement,
)


def get_recommendation(row):
    return (
        "normal"
        if row["measurement_value"] >= 60
        else "abnormal"
        if row["measurement_value"] < 40
        else "greyzone"
    )


def retrieve_meas():
    """Write ground truth volumes, ejection fractions, and recommendations."""
    io_clean = dbReadWriteClean()
    io_views = dbReadWriteViews()
    io_measurement = dbReadWriteMeasurement()

    # For measurement names and units on the study level.
    measurement_abstract_rpt_df = io_clean.get_table("measurement_abstract_rpt")
    measurement_abstract_rpt_df = measurement_abstract_rpt_df.drop(["value"], axis=1)

    # For measurement values on the instance/indexinmglist/meassequence level.
    a_measgraphref_df = io_clean.get_table("a_measgraphref")
    a_measgraphref_df = a_measgraphref_df.drop(
        ["srinstanceidk", "imagesopinstanceuid", "measurementuid"], axis=1
    )

    # For instances with A2C/A4C views.
    instances_w_labels_df = io_views.get_table("instances_w_labels")
    instances_w_a2c_a4c_labels_df = instances_w_labels_df[
        (instances_w_labels_df["view"] != "plax")
    ]
    instances_w_a2c_a4c_labels_df = instances_w_a2c_a4c_labels_df[
        ["studyidk", "instanceidk", "filename"]
    ]

    # All measurement values for A2C/A4C instances with measurement names and units.
    merge_df = measurement_abstract_rpt_df.merge(
        a_measgraphref_df, on=["studyidk", "measabstractnumber"]
    )
    merge_df = merge_df.merge(
        instances_w_a2c_a4c_labels_df, on=["studyidk", "instanceidk"]
    )

    # To calculate ejection fractions, need gold-standard end systole/diastole volumes (MDD-ps4, non-negative).
    filter_df = merge_df[merge_df["name"].isin(["VTD(MDD-ps4)", "VTS(MDD-ps4)"])]
    filter_df = filter_df[filter_df["value"] > 0]

    # Rename and reorder columns for measurement schema.
    rename_df = filter_df[
        [
            "studyidk",
            "instanceidk",
            "filename",
            "name",
            "unitname",
            "value",
            "indexinmglist",
        ]
    ]
    rename_df = rename_df.rename(
        columns={
            "studyidk": "study_id",
            "instanceidk": "instance_id",
            "filename": "file_name",
            "name": "measurement_name",
            "unitname": "measurement_unit",
            "value": "measurement_value",
        }
    )

    # Get median measurement values over meassequence/indexinmglist.
    agg_dict = {
        "measurement_unit": pd.Series.unique,
        "measurement_value": pd.Series.median,
    }
    volume_df = (
        rename_df.groupby(
            [
                "study_id",
                "instance_id",
                "file_name",
                "measurement_name",
                "indexinmglist",
            ]
        )
        .agg(agg_dict)
        .reset_index()
    )
    volume_df = (
        volume_df.groupby(["study_id", "instance_id", "file_name", "measurement_name"])
        .agg(agg_dict)
        .reset_index()
    )

    # Get diastole and systole volumes that are in the same instances.
    diastole_df = volume_df[volume_df["measurement_name"].str.contains("VTD")]
    systole_df = volume_df[volume_df["measurement_name"].str.contains("VTS")]

    diastole_df = diastole_df.drop(["measurement_name", "measurement_unit"], axis=1)
    systole_df = systole_df.drop(["measurement_name", "measurement_unit"], axis=1)

    diastole_df = diastole_df[
        diastole_df["instance_id"].isin(systole_df["instance_id"].unique())
    ]
    systole_df = systole_df[
        systole_df["instance_id"].isin(diastole_df["instance_id"].unique())
    ]

    # Calculate ejection fractions where diastole volume is no less than systole volume.
    ef_df = diastole_df.merge(
        systole_df, on=["study_id", "instance_id"], suffixes=["_diastole", "_systole"]
    )
    ef_df = ef_df[
        ef_df["measurement_value_diastole"] >= ef_df["measurement_value_systole"]
    ]

    ef_df["file_name"] = ef_df["file_name_diastole"]
    ef_df["measurement_name"] = "FE(MDD-ps4)"
    ef_df["measurement_unit"] = "%"
    ef_df["measurement_value"] = (
        (ef_df["measurement_value_diastole"] - ef_df["measurement_value_systole"])
        / ef_df["measurement_value_diastole"]
        * 100
    )

    ef_df = ef_df.drop(
        [
            "file_name_diastole",
            "measurement_value_diastole",
            "file_name_systole",
            "measurement_value_systole",
        ],
        axis=1,
    )

    # Get recommendations based on ejection fraction values.
    recommendation_df = ef_df.copy()
    recommendation_df["measurement_name"] = "recommendation"
    recommendation_df["measurement_unit"] = ""
    recommendation_df["measurement_value"] = recommendation_df.apply(
        get_recommendation, axis=1
    )

    # Write volumes, ejection fractions, and recommendations.
    ground_truth_df = volume_df.append(ef_df).append(recommendation_df)
    ground_truth_df["file_name"] = (
        "a_"
        + ground_truth_df["study_id"].astype(str)
        + "_"
        + ground_truth_df["file_name"]
    )

    # Add serial id.
    old_ground_truth_df = io_measurement.get_table("ground_truths")
    start = len(old_ground_truth_df)
    ground_truth_id = pd.Series(start + ground_truth_df.index)
    ground_truth_df.insert(0, "ground_truth_id", ground_truth_id)
    all_ground_truth_df = old_ground_truth_df.append(ground_truth_df)
    io_measurement.save_to_db(all_ground_truth_df, "ground_truths")
