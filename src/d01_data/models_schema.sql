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
  entity_id varchar,
  as_of_date timestamp with time zone,
  score double,
  label_value varchar,
  primary key(model_id, entity_id)
);

--table prediction_output_files: specifically for the prediction of the
--segmentation task we will need to store 2 numpy arrays with the x,y
--coordinate of the submask and 3 different images related to these submasks:
--original image, prediction and overlay of the two.
drop table if exists modeling.prediction_output_files;

create table modeling.prediction_output_files(
  model_id integer references modeling.model_groups(model_group_id),
  entity_id varchar,
  output_id serial,
  chamber_label varchar,
  output_np_array bytea[],
  output_image uuid [],
  primary key(output_id, model_id, entity_id)
);


--table evaluations: stores the evalutation metric used for this algorithm
drop table if exists modeling.evaluations;

create table modeling.evaluations(
  model_id integer references modeling.model_groups(model_group_id),
  metric varchar,
  value double,
  comment varchar,
  evalutaion_start_time timestamp with time zone,
  evaluation_end_time timestamp with time zone,
  primary key(model_id, metric)
);

--table feature_importances: stores the feature importance from a model (if available)
drop table if exists modeling.feature_importances;

create table modeling.feature_importances(
  model_id integer references modeling.model_groups(model_group_id),
  feature varchar,
  feature_importance integer,
  rank_abs integer,
  rank_pct double,
  primary key(model_id,feature)
);
