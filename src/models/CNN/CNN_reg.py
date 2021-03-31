from typing import Iterator, List, Union, Tuple
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow import keras, config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import History
from tensorflow.keras.metrics import RootMeanSquaredError

import tensorflow_addons as tfa

gpus = config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        config.experimental.set_virtual_device_configuration(gpus[0], [config.experimental.VirtualDeviceConfiguration(memory_limit=4048)])
    except RuntimeError as e:
        print(e)


def build_dataframe(root_folder, df):
   # Get list of images paths
    img_paths = []
    for path, _, files in os.walk(root_folder):
        for name in files:
            img_paths.append(os.path.join(path, name))
    # add the correct path for the image locations.
    df["image_location"] = img_paths
    df = df.filter(["image_location", "ghi_clipped_x"])
    return df  


def split_data(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Accepts a Pandas DataFrame and splits it into training, testing and validation data. Returns DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """
    # split the data with a validation size of 20%
    tmp_train, val = train_test_split(df, test_size=0.3, random_state=1)  
    # split the train data with an overall test size of 10%
    train, test = train_test_split(tmp_train, test_size=0.125, random_state=1)  

    print("shape train: ", train.shape)  
    print("shape val: ", val.shape)  
    print("shape test: ", test.shape)  

    return train, val, test 


def get_mean_baseline(train: pd.DataFrame, val: pd.DataFrame) -> float:
    """Calculates the mean MAE and MAPE baselines by taking the mean values of the training data as prediction for the
    validation target feature.

    Parameters
    ----------
    train : pd.DataFrame
        Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Pandas DataFrame containing your validation data.

    Returns
    -------
    float
        MAPE value.
    """
    y_hat = train["ghi_clipped_x"].mean()
    val.loc[:, "y_hat"] = y_hat
    mae = MeanAbsoluteError()
    mape = MeanAbsolutePercentageError()
    rmse = RootMeanSquaredError()
    mae = mae(val["ghi_clipped_x"], val["y_hat"]).numpy()
    mape = mape(val["ghi_clipped_x"], val["y_hat"]).numpy()
    rmse = rmse(val["ghi_clipped_x"], val["y_hat"]).numpy()

    print("mean baseline MAPE: ", mape)
    print("mean baseline RMSE: ", rmse)
    print("mean baseline MAE: ", mae)
    return mape


def create_generators(
    df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, viz_augmentations: False):
    """Accepts four Pandas DataFrames: all your data, the training, validation and test DataFrames. Creates and returns
    keras ImageDataGenerators. Within this function you can also visualize the augmentations of the ImageDataGenerators.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.
    train : pd.DataFrame
        Your Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Your Pandas DataFrame containing your validation data.
    test : pd.DataFrame
        Your Pandas DataFrame containing your testing data.

    Returns
    -------
    Tuple[Iterator, Iterator, Iterator]
        keras ImageDataGenerators used for training, validating and testing of your models.
    """
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=[0.9, 1.1],
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )  # create an ImageDataGenerator with multiple image augmentations
    validation_generator = ImageDataGenerator(rescale=1.0 / 255)  
    # except for rescaling, no augmentations are needed for validation and testing generators
    test_generator = ImageDataGenerator(rescale=1.0 / 255)
    # visualize image augmentations
    if viz_augmentations:
        visualize_augmentations(train_generator, df)

    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        x_col="image_location",  # this is where your image data is stored
        y_col="ghi_clipped_x",  # this is your target feature
        class_mode="raw",  # use "raw" for regressions
        target_size=(240, 240),
        batch_size=64, # increase or decrease to fit your GPU
    )

    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=val,
        x_col="image_location",
        y_col="ghi_clipped_x",
        class_mode="raw",
        target_size=(240, 240),
        batch_size=64,
    )
    test_generator = test_generator.flow_from_dataframe(
        dataframe=test,
        x_col="image_location",
        y_col="ghi_clipped_x",
        class_mode="raw",
        target_size=(240, 240),
        batch_size=64,
    )
    return train_generator, validation_generator, test_generator


def visualize_augmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    """Visualizes the keras augmentations with matplotlib in 3x3 grid. This function is part of create_generators() and
    can be accessed from there.

    Parameters
    ----------
    data_generator : Iterator
        The keras data generator of your training data.
    df : pd.DataFrame
        The Pandas DataFrame containing your training data.
    """
    # creating a small dataframe with one image
    series = df.iloc[2]
    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(
        dataframe=df_augmentation_visualization,
        x_col="image_location",
        y_col="ghi_clipped_x",
        class_mode="raw",
        target_size=(240, 240),  # size of the image
        batch_size=1,  # use only one image for visualization
    )

    fig, axarr = plt.subplots(3,3)
    batchs = [next(iterator_visualizations) for k in range(9)]
    imgs = [batch[0][0, :, :, :] for batch in batchs]
    for ax,im in zip(axarr.ravel(), imgs):
        ax.imshow(im)
    fig.savefig('data_aug.png')


def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """
    logdir = (
        "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_percentage_error",
        min_delta=1,  # model should improve by at least 1%
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        "data/models/" + model_name,
        monitor="val_mean_absolute_percentage_error",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
        save_freq="epoch",  # save every epoch
    )
    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]


def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
) -> History:
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.
    model_function : Model
        Keras model function like small_cnn()  or adapt_efficient_net().
    lr : float
        Learning rate.
    train_generator : Iterator
        keras ImageDataGenerators for the training data.
    validation_generator : Iterator
        keras ImageDataGenerators for the validation data.
    test_generator : Iterator
        keras ImageDataGenerators for the test data.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history. For an example
        see plot_results().
    """

    callbacks = get_callbacks(model_name)
    model = model_function
    model.summary()
    plot_model(model, to_file=model_name + ".jpg", show_shapes=True)

    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger

    model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError(), RootMeanSquaredError()]
    )
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=6, # adjust this according to the number of CPU cores of your machine
    )

    test_eval = model.evaluate(
        test_generator,
        callbacks=callbacks,
    )
    return history, test_eval


def small_cnn() -> Sequential:
    """A very small custom convolutional neural network with image input dimensions of 240x240x3.

    Returns
    -------
    Sequential
        The keras Sequential model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(240, 240, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    return model


def plot_results(model_history_small_cnn: History, mean_baseline: float):
    """This function uses seaborn with matplotlib to plot the trainig and validation losses of both input models in an
    sns.relplot(). The mean baseline is plotted as a horizontal red dotted line.

    Parameters
    ----------
    model_history_small_cnn : History
        keras History object of the model.fit() method.
    mean_baseline : float
        Result of the get_mean_baseline() function.
    """

    # create a dictionary for each model history and loss type
    dict1 = {
        "MAPE": model_history_small_cnn.history["mean_absolute_percentage_error"],
        "type": "training",
        "model": "small_cnn",
    }
    dict2 = {
        "MAPE": model_history_small_cnn.history["val_mean_absolute_percentage_error"],
        "type": "validation",
        "model": "small_cnn",
    }

    # convert the dicts to pd.Series and concat them to a pd.DataFrame in the long format
    s1 = pd.DataFrame(dict1)
    s2 = pd.DataFrame(dict2)
    df = pd.concat([s1, s2], axis=0).reset_index()
    grid = sns.relplot(data=df, x=df["index"], y="MAPE", hue="model", col="type", kind="line", legend=False)
    grid.set(ylim=(0, 100))  # set the y-axis limit
    for ax in grid.axes.flat:
        ax.axhline(
            y=mean_baseline, color="lightcoral", linestyle="dashed"
        )  # add a mean baseline horizontal bar to each plot
        ax.set(xlabel="Epoch")
    labels = ["small_cnn", "mean_baseline"]  # custom labels for the plot

    plt.legend(labels=labels)
    plt.savefig("training_validation.png")
    plt.show()


def run(small_sample=False):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """

    root = "../../../Solais_Data/mobotix1_prepro_240/"
    df_tmp = pd.read_csv("../../../processed_data/CNN_target.csv")
    df = build_dataframe(root, df_tmp)

    if small_sample == True:
        # set small_sampe to True if you want to check if your code works without long waiting
        df = df.iloc[0:1000] 
    
    train, val, test = split_data(df)  # split your data
    mean_baseline = get_mean_baseline(train, val)

    # train_generator, validation_generator, test_generator = create_generators(
    #     df=df, train=train, val=val, test=test, viz_augmentations=True
    # )

    # small_cnn_history, test_eval = run_model(
    #     model_name="small_cnn",
    #     model_function=small_cnn(),
    #     lr=0.001,
    #     train_generator=train_generator,
    #     validation_generator=validation_generator,
    #     test_generator=test_generator,
    # )
    # plot_results(small_cnn_history, mean_baseline)
    # print(f"eval on test dataset : {test_eval}")
    # #hist = pd.DataFrame(small_cnn_history.history)
    # #hist.to_csv("history.csv")

if __name__ == "__main__":
    run(small_sample=False)