Steps for re-training the classification model:
1) In `data/m1_usal/load_data.py`, set the variables `directory_train` and `directory_test` corresponding to your training and validation data.
2) In `vgg_13/load_model.py`, the attribute `self.opt` is currently set to retrain the entire network. Should you wish to only perform transfer learning on the last three fully-connected layers, follow the corresponding comments directly below the `self.opt` attribute.
3) Hyperparameters have been tuned for this model, however they can be changed in `config.json`. Descriptions of each parameter are below.
4) From the `echo_deeplearning/` directory, run the command ```python run.py vgg_13```

You can easily configure and run other models within this same directory structure, as outlined by Zhang's instructions below.


Hyperparameters
    All parameters are stored in config.json files

    data (string) - data folder name (data preprocessing is located in this folder)
    feature_dim (int) - number of channels in the input images
    label_dim (int) - number of unique segmentation labels
    mean (float) - mean pixel of images
    dropout (float) - dropout probability for dropout layers
    weight_decay (float) - weight of weight decay in calculating the loss
    learning_rate (float) - learning rate for ADAM optimizer
    crop_max (int) - maximum crop range from each side of image
    blackout_max (int) - maximum radius for blackout circles (disease detection models don't have blackout)
    image_size (int) - dimension of image (assumed square image)
    epochs (int) - number of epochs to run
    epoch_save_interval (int) - number of epochs run before saving model and visualizing validation results
    batch_size (int) - batch size for stochastic gradient descent
    summary_interval (int) - number of steps in training before saving variables for tensorboard
    loss_smoothing (int) - number of loss values averaged for display


Configure and run with new model
    a4c/ and hcm/ folders included as examples. 
    Model requires data to be preprocessed. 
        - Create your own data folder (under 'data/[data_folder]/)
        - Make changes to 'data/a4c/load_data.py' or 'data/hcm/load_data.py', follow format, and save under 'data/[data_folder]/load_data.py'

Models require hyperparameters
        - Create your own experiment folder (under 'models/unet_12/[experiment]/')
        - Modify hyperparameters and save under 'models/unet_12/[experiment]/config.json'
            Make sure the 'data' parameter is updated to the [data_folder] you created

Run src/run.py from the echo_deeplearning/ folder 
        - Read src/run.py for arguments options (only required arugment is the model and experiment)
        - Format is 'python src/run.py models/[model]/[experiment]'
            ex: 'python src/run.py models/unet_12/a4c'
            ex 2: 'python src/run.py models/unet_12/a4c --train True --gpu 0 --val_split 0 --debug False --retrain False' 
    - Note if terminal begins to overflow with progress text, stretch the terminal window so all information fits on one line

Visualize progress, loss, training accuracy, and validation accuracy
        - Tensorboard variables will be stored under 'models/[model]/[experiment]/train/[val_split]'
        - Run 'tensorboard --logdir models/[model]/[experiment]/train/'
        - View at 'http://0.0.0.0:6006'
        ex: 'tensorboard --logdir models/unet_12/a4c/train/'
