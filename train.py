import shutil
import subprocess
from moviepy.editor import VideoFileClip
import pathlib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential, Model, load_model, save_model
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def move_csv(path_from, path_to):
    for item in os.listdir(path_from):
        if item.endswith('.csv'):
            shutil.move(pathlib.Path(path_from, item), pathlib.Path(path_to, item))


def cut_train_videos(clipping_length, folder_path_from, folder_path_to, fps):
    os.chdir(folder_path_from)
    list_of_videos = []
    for el in os.listdir(folder_path_from):
        if el.endswith('mp4'):
            list_of_videos.append(el)

    for video in list_of_videos:
        clip = VideoFileClip(video)
        duration_sec = clip.duration
        clipping_cycles = int(duration_sec // clipping_length)
        if clipping_cycles > 0:
            for i in range(clipping_cycles):
                start = 0 + i * clipping_length
                end = (i + 1) * clipping_length
                n_clip = clip.subclip(start, end)
                n_clip.write_videofile(f"{folder_path_to}/{start}_{end}_{video}", fps=fps)


def get_au_train(video_folder, save_path, path_to_OpenFace):
    working_dir = pathlib.Path(path_to_OpenFace)
    os.chdir(working_dir)

    video_list = os.listdir(video_folder)
    for video in video_list:
        video_dir = os.path.join(video_folder, video)
        print(video_dir)
        subprocess.run(['FeatureExtraction.exe', '-f', video_dir])
        print('all good')

    path_from = pathlib.Path(path_to_OpenFace, 'processed')
    os.makedirs(save_path, exist_ok=True)
    move_csv(path_from, save_path)


def concat_train_csv(csv_folder_path):
    dataset = []
    for file in os.listdir(csv_folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(pathlib.Path(csv_folder_path, file))
            AU_arr = list(np.concatenate((df[df.columns[-35:-18]].values, df[df.columns[11:13]].values), axis=1))
            dataset.append(AU_arr)
            np.save('dataset_lie', dataset)

    return dataset


def prepare_train_set(clipping_length, folder_path_truth_video_from, folder_path_truth_cut_to, folder_path_deceptive_video_from,
                      folder_path_deceptive_cut_to, fps, save_csv_path_true, save_csv_path_false, path_to_OpenFace):

    cut_train_videos(clipping_length=clipping_length, folder_path_from=folder_path_truth_video_from,
                     folder_path_to=folder_path_truth_cut_to, fps=fps)
    cut_train_videos(clipping_length=clipping_length, folder_path_from=folder_path_deceptive_video_from,
                     folder_path_to=folder_path_deceptive_cut_to, fps=fps)

    get_au_train(video_folder=folder_path_truth_cut_to, save_path=save_csv_path_true, path_to_OpenFace=path_to_OpenFace)
    get_au_train(video_folder=folder_path_deceptive_cut_to, save_path=save_csv_path_false,
                 path_to_OpenFace=path_to_OpenFace)

    dataset_true = concat_train_csv(csv_folder_path=save_csv_path_true)
    dataset_lie = concat_train_csv(csv_folder_path=save_csv_path_false)

    X = np.concatenate((dataset_lie, dataset_true))
    y_0 = np.array([0 for el in dataset_lie])
    y_1 = np.array([1 for el in dataset_true])
    y = np.concatenate((y_0, y_1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, stratify=y)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, model_save_path):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(4, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics='Accuracy')

    r_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )
    # ch_pt_callback = tf.keras.callbacks.ModelCheckpoint(
    #     '/content/my_model',
    #     monitor="val_loss",
    #     verbose=0,
    #     save_best_only=True,
    #     save_weights_only=False,
    #     mode="auto",
    #     save_freq="epoch",
    # )

    model.fit(X_train, y_train, batch_size=32, epochs=60, validation_data=(X_test, y_test),
              callbacks=[r_lr_callback])

    model.tf.keras.models.save(model_save_path)

    return model


# truthful videos
folder_path_truth_video_from = pathlib.Path(
    r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\RealLifeDeceptionDetection.2016\Real'
    r'-life_Deception_Detection_2016\Clips\Truthful')
folder_path_truth_cut_to = pathlib.Path(
    r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\RealLifeDeceptionDetection.2016\Real'
    r'-life_Deception_Detection_2016\Clips\Truthful\cut_videos_truthful')

# deceptive videos
folder_path_deceptive_video_from = pathlib.Path(
    r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\RealLifeDeceptionDetection.2016\Real'
    r'-life_Deception_Detection_2016\Clips\Deceptive')
folder_path_deceptive_cut_to = pathlib.Path(
    r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\RealLifeDeceptionDetection.2016\Real'
    r'-life_Deception_Detection_2016\Clips\Deceptive\cut_videos_deceptive')

# OpenFace folder
path_to_OpenFace = pathlib.Path(r'C:/Users/ale-d/OneDrive/Рабочий стол/LieFolder/OpenFace_2.2.0_win_x64/')

# path to csv files processed by OpenFace
save_csv_path_true = pathlib.Path(r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\OpenFace_2.2.0_win_x64\processed\truthful_csv')
save_csv_path_false = pathlib.Path(r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\OpenFace_2.2.0_win_x64\processed\deceptive_csv')


X_train, X_test, y_train, y_test = prepare_train_set(clipping_length=10, folder_path_truth_video_from=folder_path_truth_video_from, folder_path_truth_cut_to=folder_path_truth_cut_to,
                                                      folder_path_deceptive_video_from=folder_path_deceptive_video_from, folder_path_deceptive_cut_to=folder_path_deceptive_cut_to,
                                                      fps=30, save_csv_path_true=save_csv_path_true, save_csv_path_false=save_csv_path_false, path_to_OpenFace=path_to_OpenFace)


train_model(X_train, X_test, y_train, y_test, r'C:\users\model.h5')



