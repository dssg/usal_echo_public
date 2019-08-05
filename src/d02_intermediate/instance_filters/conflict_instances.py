from d00_utils.db_utils import dbReadWriteViews


def find_table_conflicts():
    """ Creates table with instances that contain conflicts
	    i.e. these instances should not be used """

    io_views = dbReadWriteViews()

    all_df = io_views.get_table("frames_sorted_by_views_temp")
    df5 = all_df.drop(all_df[(all_df["view"] == "")].index)

    # Get db which contains all off-limits instances for other classes
    # e.g. is 'is_multiview'==True
    conflict_inst_df = df5.drop(df5[(df5["is_multiview"] == False)].index)
    agg_functions_conflict = {"studyidk": "first"}
    conflict_inst_df = conflict_inst_df.groupby(["instanceidk"]).agg(
        agg_functions_conflict
    )
    conflict_inst_df.reset_index(inplace=True)

    io_views.save_to_db(conflict_inst_df, "instances_w_conflicts")
    print("New table created: views.instances_w_conflicts")

