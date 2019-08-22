--BASIC INFRA REQUIREMENTS
DROP SCHEMA IF EXISTS raw CASCADE;

CREATE SCHEMA raw;

 --METADATA FROM DICOM IMAGES info
DROP TABLE IF EXISTS raw.metadata;

CREATE TABLE raw.metadata(
    dirname varchar,
    filename varchar,
    tag1 varchar,
    tag2 varchar,
    value varchar
 );

--CLASSIFICATION SCHEMA
--tables for persisting the outputs of the measurement process
DROP SCHEMA IF EXISTS classification CASCADE;

CREATE SCHEMA classification;

--table predictions:
drop table if exists classification.predictions;

create table classification.predictions(
    study_id integer,
    file_name varchar,
    model_name varchar,
    date_run varchar,
    img_dir varchar,
    view23_pred varchar,
    view4_dev varchar,
    view4_seg varchar,
    prediction_id serial,
    primary key(prediction_id)
);


--table evaluations
drop table if exists classification.evaluations;

create table classification.evaluations(
    model_name varchar,
    img_dir varchar,
    date_run varchar,
    study_filter varchar,
    view varchar,
    tn float,
    fp float,
    fn float,
    tp float,
    evaluation_id serial,
    primary key(evaluation_id)
);

--table ground_truths
drop table if exists classification.ground_truths;

create table classification.ground_truths(
    ground_truth_id serial,
    study_id integer,
    instance_id integer,
    file_name varchar,
    view_name varchar,
    primary key(ground_truth_id)
);

--table probabilities_frames (10 per instance)
DROP TABLE IF EXISTS classification.probabilities_frames;

CREATE TABLE classification.probabilities_frames (
    study_id text,
    file_name text,
    model_name text,
    date_run timestamp without time zone,
    img_dir text,
    output_plax_far double precision,
    output_plax_plax double precision,
    output_plax_laz double precision,
    output_psax_az double precision,
    output_psax_mv double precision,
    output_psax_pap double precision,
    output_a2c_lvocc_s double precision,
    output_a2c_laocc double precision,
    output_a2c double precision,
    output_a3c_lvocc_s double precision,
    output_a3c_laocc double precision,
    output_a3c double precision,
    output_a4c_lvocc_s double precision,
    output_a4c_laocc double precision,
    output_a4c double precision,
    output_a5c double precision,
    output_other double precision,
    output_rvinf double precision,
    output_psax_avz double precision,
    output_suprasternal double precision,
    output_subcostal double precision,
    output_plax_lac double precision,
    output_psax_apex double precision,
    probabilities_frame_id serial NOT NULL
);

--table probabilities_instances: average the probabilities of the frames
DROP TABLE IF EXISTS classification.probabilities_instances;

CREATE TABLE classification.probabilities_instances (
    study_id bigint,
    file_name text,
    model_name text,
    date_run text,
    img_dir text,
    output_plax_far double precision,
    output_plax_plax double precision,
    output_plax_laz double precision,
    output_psax_az double precision,
    output_psax_mv double precision,
    output_psax_pap double precision,
    output_a2c_lvocc_s double precision,
    output_a2c_laocc double precision,
    output_a2c double precision,
    output_a3c_lvocc_s double precision,
    output_a3c_laocc double precision,
    output_a3c double precision,
    output_a4c_lvocc_s double precision,
    output_a4c_laocc double precision,
    output_a4c double precision,
    output_a5c double precision,
    output_other double precision,
    output_rvinf double precision,
    output_psax_avz double precision,
    output_suprasternal double precision,
    output_subcostal double precision,
    output_plax_lac double precision,
    output_psax_apex double precision,
    frame_count bigint,
    probabilities_instance_id serial NOT NULL
);


--SEGMENTATION SCHEMA
----tables for persisting the outputs of segmentation model

DROP SCHEMA IF EXISTS segmentation CASCADE;

CREATE SCHEMA segmentation;

--predictions: store the predictions for each study
drop table if exists segmentation.predictions;

create table segmentation.predictions(
   prediction_id serial,
   study_id integer,
   instance_id integer,
   file_name varchar,
   num_frames integer,
   model_name varchar,
   date_run timestamp with time zone,
   output_np_lv bytea,
   output_np_la bytea,
   output_np_lvo bytea,
   output_image_seg varchar,
   output_image_orig varchar,
   output_image_overlay varchar,
   primary key(prediction_id)
);

--evaluation:
drop table if exists segmentation.evaluations;

create table segmentation.evaluations(
    evaluation_id serial,
    study_id integer,
    instance_id integer,
    file_name varchar,
    frame integer,
    model_name varchar,
    chamber varchar,
    score_type varchar,
    score_value float,
    gt_view_name varchar,
    pred_view_name varchar,
    primary key(evaluation_id)
);

--ground truth
drop table if exists segmentation.ground_truths;

create table segmentation.ground_truths(
    ground_truth_id serial,
    study_id integer,
    instance_id integer,
    file_name varchar,
    frame integer,
    chamber varchar,
    view_name varchar,
    numpy_array bytea,
    primary key(ground_truth_id)
);


--MEASUREMENTS SCHEMA
--tables for persisting the outputs of the measurement process
DROP SCHEMA IF EXISTS measurement CASCADE;

CREATE SCHEMA measurement;

--table calculations:
drop table if exists measurement.calculations;

CREATE TABLE measurement.calculations (
    calculation_id text,
    date_run text,
    file_name text,
    instance_id bigint,
    measurement_name text,
    measurement_unit text,
    measurement_value text,
    study_id bigint
);


--table evaluations:
drop table if exists measurement.evaluations;

CREATE TABLE measurement.evaluations (
    calculation_id text,
    date_run text,
    evaluation_id text,
    file_name text,
    instance_id bigint,
    measurement_name text,
    measurement_unit text,
    measurement_value text,
    score_type text,
    score_value text,
    study_id bigint
);



--table ground_truths:
drop table if exists measurement.ground_truths;

CREATE TABLE measurement.ground_truths (
    ground_truth_id bigint,
    study_id bigint,
    instance_id bigint,
    file_name text,
    measurement_name text,
    measurement_unit text,
    measurement_value text
);

