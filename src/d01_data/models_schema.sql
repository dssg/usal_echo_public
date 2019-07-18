DROP SCHEMA IF EXISTS modeling;

CREATE SCHEMA modeling;


--table model_groups: specific configuration of hyperparameters, features used,
--type of algorithm
drop table if exists modeling.model_groups;

create table modeling.model_groups(
  model_group_id serial,
  model_type varchar,
  model_parameters uuid,
  features_list text [],
  primary key(model_group_id)
);

--table models: contains the fit train data used on the model_group
drop table if exists modeling.models;

create table modeling.models(
  model_id serial,
  model_group_id integer references modeling.model_groups(model_group_id),
  run_time timestamp with time zone,
  model_hash uuid,
  train_matrix_uuid uuid,
  train_start_time timestamp with time zone,
  train_end_time timestamp with time zone,
  primary key(model_id)
);

--table predictions: store the predictions made for each observation
--on the testing set
drop table if exists modeling.predictions;

create table modeling.predictions(
  model_id integer references modeling.model_groups(model_group_id),
  entity id varchar,
  as_of_date timestamp with time zone,
  score double,
  label_value varchar,
  primary key(model_id, entity_id)
);

--table predition_output_files: specifically for the prediction of the
--segmentation task we will need to store 2 numpy arrays with the x,y
--coordinate of the submask and 4 different images related to these submasks
drop table if exists modeling.prediction_output_files;

create table modeling.prediction_output_files(
  model_id integer references modeling.model_groups(model_group_id),
  entity_id varchar,
  chamber_label varchar,
  output_np_array bytea[],
  output_image uuid [],
  primary key(model_id, entity_id)
);
