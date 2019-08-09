--CLASSIFICATION SCHEMA
--tables for persisting the outputs of the measurement process
DROP SCHEMA IF EXISTS classification CASCADE;

CREATE SCHEMA classification;

--table predictions:
drop table if exists classification.predictions;

create table classification.predictions(
    prediction_id serial,
    study_id integer,
    instance_id integer,
    file_name varchar,
    model_name varchar,
    date_run timestamp with time zone,
    primary key(prediction_id)
);


--table evaluations
drop table if exists classification.evaluations;

create table classification.evaluations(
    evaluation_id serial,
    study_id integer,
    instance_id integer,
    file_name varchar,
    score_type varchar,
    score_value float,
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


--SEGMENTATION SCHEMA
----tables for persisting the outputs of segmentation model

DROP SCHEMA IF EXISTS segmentation  CASCADE;

CREATE SCHEMA segmentation;

--predictions: store the predictions for each study
drop table if exists segmentation.predictions;

create table segmentation.predictions(
   prediction_id serial,
   study_id integer,
   instance_id integer,
   file_name varchar,
   frame integer,
   view_name varchar,
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
    frame integer,
    chamber varchar,
    score_type varchar,
    score_value float,
    primary key(evaluation_id)
);

--ground truth
drop table if exists segmentation.ground_truths;

create table segmentation.ground_truths(
    ground_truth_id serial,
    study_id integer,
    instance_id integer,
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

create table measurement.calculations(
    calculation_id serial,
    instance_id integer,
    study_id integer,
    measurement_name varchar,
    measurement_unit varchar,
    measurement_value float,
    date_run timestamp with time zone,
    primary key(calculation_id)
);

--table evaluations:
drop table if exists measurement.evaluations;

create table measurement.evaluations(
    evaluation_id serial,
    instance_id integer,
    study_id integer,
    score_type varchar,
    score_value float,
    primary key(evaluation_id)
);

--table ground_truths:
drop table if exists measurement.ground_truths;

create table measurement.ground_truths(
    ground_truth_id serial,
    instance_id varchar,
    study_id varchar,
    measurement_name varchar,
    measurement_unit varchar,
    measurement_value float,
    primary key(ground_truth_id)
);
