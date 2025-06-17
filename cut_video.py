from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import os
import shutil


def cut_video(video_path, folder_path):

    if os.path.exists(f"./{folder_path}/"):
        shutil.rmtree(f"./{folder_path}/")
    os.mkdir(f"./{folder_path}/")

    dirpath, dirnames, filenames = next(os.walk(f"{video_path}/{folder_path}/"))

    shutil.copyfile(f"{video_path}/{folder_path}/{filenames[0]}", f"./{folder_path}/{filenames[0]}")

    for video_name in dirnames:
        video_path = f"{dirpath}{video_name}/{video_name}.mp4"
        new_video_path = f"./{folder_path}/{video_name}.mp4"
        # cut 4 minutes from the video
        ffmpeg_extract_subclip(video_path, 0, 240, outputfile=new_video_path)


if __name__ == '__main__':
    video_path = '/home/pashnya/Downloads/vk_video_intern'

    train_folder = 'data_train_short'
    test_folder = 'data_test_short'

    cut_video(video_path, train_folder)
    cut_video(video_path, test_folder)


