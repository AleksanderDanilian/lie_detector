import shutil
import subprocess

from moviepy.editor import VideoFileClip
import pathlib
import os
import math
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model


def cut_videos(clipping_length, file_path, file_folder_save_path, fps=30):
    """
    Функция нарезает видно на отрезки равные clipping_length и пересохраняет их
    с указанным fps.
    :param clipping_length: длина нарезки видео в секундах
    :param file_path: путь к видео
    :param file_folder_save_path: путь к папке, куда будем сохранять
    :param fps: frames per second - пересохраняет видно с такими показателями
    :return:
    """
    os.makedirs(pathlib.Path(file_folder_save_path, 'Cut'), exist_ok=True)
    os.chdir(file_path.parent)
    video = file_path.name
    clip = VideoFileClip(video)
    duration_sec = clip.duration
    clipping_cycles = math.floor(duration_sec / clipping_length)

    try:
        existing_folders = os.listdir(str(file_folder_save_path) + '/Cut')  # все цифровые папки
        existing_folders = int(sorted(existing_folders)[-1])  # берем последнюю
    except Exception as e:
        print(e, 'Нет папок в Cut, создаем первую')
        existing_folders = 0
        pass

    os.makedirs(pathlib.Path(file_folder_save_path, 'Cut', f'{existing_folders + 1}'))

    if clipping_cycles == 0:  # от 0 до 10 сек видео
        start = 0
        end = int(clip.duration)
        n_clip = clip.subclip(start, end)
        n_clip.write_videofile(f"{file_folder_save_path}/Cut/{existing_folders + 1}/{start}_{end}_{video}", fps=fps)

    elif clipping_cycles > 0:
        for i in range(clipping_cycles):
            try:
                start = 0 + i * clipping_length
                end = (i + 1) * clipping_length
                n_clip = clip.subclip(start, end)
                n_clip.write_videofile(f"{file_folder_save_path}/Cut/{existing_folders + 1}/{start}_{end}_{video}",
                                       fps=fps)

            except OSError:
                try:
                    start = (i - 1) * clipping_length  # откатываемся с последнего цикла
                    end = int(clip.duration) - 1
                    n_clip = clip.subclip(start, end)
                    n_clip.write_videofile(f"{file_folder_save_path}/Cut/{existing_folders + 1}/{start}_{end}_{video}",
                                           fps=30)
                except OSError as e:
                    print(e)
                    pass

    path_with_cut_videos = f"{file_folder_save_path}/Cut/{existing_folders + 1}"

    return path_with_cut_videos


def get_au(path_with_cut_videos, path_to_Open_Face_folder):
    """
    Функция получения метрик с программы OpenFace.
    :param path_with_cut_videos: путь к нарезанным видео.
    :param path_to_Open_Face_folder: путь к папке с программой OpenFace.
    :return:
    .csv файл с метриками. Сохраняется в той же папке, что и нарезанные видео.
    """
    os.chdir(path_to_Open_Face_folder)

    video_list = os.listdir(path_with_cut_videos)
    for video in video_list:
        if video.endswith('.mp4'):
            video_dir = str(pathlib.Path(path_with_cut_videos, video))
            path_to_feature_extraction = str(pathlib.Path(path_to_Open_Face_folder, 'FeatureExtraction.exe'))
            subprocess.run([path_to_feature_extraction, '-f', video_dir])
            shutil.move(pathlib.Path(path_to_Open_Face_folder, 'processed', f'{video[:-3]}csv'),
                        pathlib.Path(path_with_cut_videos, f'{video[:-3]}csv'))
        else:
            pass


def concat_dataframes(path_with_cut_videos, clipping_length=10, fps=30):
    """
    Функция сложения датафреймов и подготовки массива для подачи в нейронную сеть.
    :param path_with_cut_videos: путь к папке с нарезанными видео
    :param fps: frames per second - пересохраняет видно с такими показателями
    :param clipping_length: длина нарезки видео в секундах
    :return:
    numpy массив c данными (выход программы OpenFace)
    """
    au_arr = None
    for i, file in enumerate(os.listdir(path_with_cut_videos)):
        if file.endswith('.csv'):
            if au_arr is None:
                df = pd.read_csv(pathlib.Path(path_with_cut_videos, file))
                au_arr = list(np.concatenate((df[df.columns[-35:-18]].values, df[df.columns[11:13]].values), axis=1))
                print('we are here')
            else:
                df = pd.read_csv(pathlib.Path(path_with_cut_videos, file))
                temp_au_arr = list(
                    np.concatenate((df[df.columns[-35:-18]].values, df[df.columns[11:13]].values), axis=1))
                au_arr = np.concatenate((au_arr, temp_au_arr))
                print('Nope, here')

    au_arr = np.array(au_arr)
    leng = au_arr.shape[0]  # смотрим, сколько всего фреймов(строчек) получилось в нашем файле
    to_append_in_the_end = au_arr[0: (clipping_length * fps) - leng % (clipping_length * fps)]
    au_arr = np.concatenate((au_arr, to_append_in_the_end))

    return au_arr


def get_predict(model, au_arr, fps, clipping_length):
    """
    Функция предикта и подсчета среднего значения по каждому отрезку видео.
    :param model: модель нейронной сети
    :param au_arr: numpy массив c данными (выход программы OpenFace)
    :param fps: frames per second - пересохраняет видно с такими показателями
    :param clipping_length: длина нарезки видео в секундах
    :return:
    Выдает значение lie/True (берет предикт по каждому отрезку видео, считает среднее, и выдает ответ).
    """
    arr_batch_length = fps * clipping_length  # какими кусками подаем данные в модель (на каких кусках обучали, на таких и подаем)

    nr_batches = int(au_arr.shape[0] / arr_batch_length)
    pred_results = []
    for i in range(nr_batches):
        pred = model.predict(np.expand_dims(au_arr[i * arr_batch_length: (i + 1) * arr_batch_length, :], axis=0))[0][0]
        pred_results.append(pred)

    answer = round(sum(pred_results) / len(pred_results), 0)

    if answer == 0:
        return 'lie'
    else:
        return 'true'


def main(file_path, file_folder_save_path, path_to_Open_Face_folder, model_path, clipping_length=10, fps=30):
    """
    Основная функция для инференса. Входной видео файл разбивается на фрагменты длинной clipping_length, пересохраняются
    с указанным fps. Если длина файла не кратна clipping length, то csv файл с AU метриками, с которым работает сама
    обученная модель, удлинняется за счет копирования строк в кол-ве, равном недостающему кол-ву строк для получения
    массивов длинной clipping_length*fps.
    :param file_path: путь к видео. Видео в формате mp4.
    :param file_folder_save_path: папка для сохранения данных
    :param path_to_Open_Face_folder: папка с распакованным дистрибутивом OpenFace.
    :param model_path: путь к модели. Модель в формате .h5 или в специальном формате tf (станданртый выход model.save())
    :param clipping_length: длина нарезки видео в секундах
    :param fps: frames per second - пересохраняет видно с такими показателями
    :return:
    Выдает значение lie/True (берет предикт по каждому отрезку видео, считает среднее, и выдает ответ).
    """

    path_with_cut_videos = cut_videos(clipping_length=clipping_length, file_path=file_path,
                                      file_folder_save_path=file_folder_save_path, fps=fps)

    get_au(path_with_cut_videos=path_with_cut_videos,
           path_to_Open_Face_folder=path_to_Open_Face_folder)

    au_arr = concat_dataframes(path_with_cut_videos=path_with_cut_videos, clipping_length=clipping_length, fps=fps)

    model = load_model(str(model_path))

    answer = get_predict(model, au_arr, fps, clipping_length)

    return answer


answer = main(file_path=pathlib.Path(
            r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\RealLifeDeceptionDetection.2016\Real'
            r'-life_Deception_Detection_2016\Clips\test_2.mp4'),
            file_folder_save_path=pathlib.Path(
                r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\RealLifeDeceptionDetection.2016\Real'
                r'-life_Deception_Detection_2016\Clips\Test'),
            path_to_Open_Face_folder=pathlib.Path(r'C:\Users\ale-d\OneDrive\Рабочий стол\LieFolder\OpenFace_2.2.0_win_x64'),
            model_path=pathlib.Path(r'C:\test_model.h5'))

print(answer)
