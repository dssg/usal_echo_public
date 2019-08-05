--SEGMENTATION SCHEMA
----tables for persisting the outputs of segmentation model

DROP SCHEMA IF EXISTS segmentation  CASCADE;

CREATE SCHEMA segmentation;

--predictions: store the predictions for each study
drop table segmentation.predictions if exists;

create table segmentation.predictions(
   prediction_id serial,
   instance_id integer,
   frame integer,
   chamber varchar,
   study_id integer,
   view_name varchar,
   output_np bytea [],
   output_image uuid [],
   date_run timestamp with time zone,
   primary key(prediction_id)
);

--evaluation:
drop table segmentation.evaluations id exists;

create table segmentation.evaluations(
    evaluation_id serial,
    instance_id integer,
    frame integer,
    chamber varchar,
    study_id integer,
    score_type varchar,
    score_value float,
    primary key(evaluation_id)
);

--ground truth
drop table segmentation.ground_truths exists;

create table segmentation.ground_truths(
    ground_truth_id serial,
    instance_id integer,
    frame integer,
    chamber varchar,
    study_id integer,
    view_name varchar,
    numpy_array bytea[],
    primary key(ground_truth_id)
);