import cv2
import requests
import numpy as np
from datetime import datetime
import os
import sys
import glob
from pathlib import Path
import logging
import sh
import argparse
from PIL import Image

# create logger with 'spam_application'
logging.getLogger('Raspi_application')
logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.DEBUG)


def delete_data(path):
    """ function deletes data library
    :param path: string"""
    files = glob.glob(path + '/*')
    for f in files:
        logging.debug(f'Deleting: {f}')
        os.remove(f)


def record_video(length_secs, path_to_stream, path_to_data):
    """ function records video
     :param: length_secs: int
             path_to_stream: string
             path_to_data: string
     """

    r = requests.get(path_to_stream, stream=True)
    if r.status_code == 200:
        bytes_loc = bytes()
        time_start = datetime.now()
        logging.debug(f'Start recording at: {time_start}')
        for chunk in r.iter_content(chunk_size=1024):
            bytes_loc += chunk
            a = bytes_loc.find(b'\xff\xd8')  # JPEG start
            b = bytes_loc.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = bytes_loc[a:b+2]  # actual image
                bytes_loc = bytes_loc[b+2:]  # other information
                # decode to colored image ( another option is cv2.IMREAD_GRAYSCALE)
                i = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                datetimeobj = datetime.now()  # get time stamp
                cv2.imwrite(path_to_data + '/img' + str(datetimeobj) + '.jpg', i)
                if cv2.waitKey(1) == 27 or (datetimeobj - time_start).seconds > length_secs:  # if user  hit esc
                    logging.debug('End recording.')
                    break  # exit program
    else:
        print("Received unexpected status code {}".format(r.status_code))


def mask_fun(img_path, erode_func, output_lib, sbool=False, output_ext=".jpg"):
    """:param img_path: string, path to the image.
       :param erode_func: function, masking function.
       :param output_lib: string, path to output directory.
       :param sbool: boolean, if True it will save the masked image, default False.
       :param output_ext: string, type of output image, default jpg.

       :returns np.arrays of g_img, mask_g and img

       """
    # set cropping parameters:
    hlc, hrc = 100, 1100
    vlc, vuc = 0, 670
    # extract img name:
    img_name = os.path.splitext(Path(img_path).name)[0]

    # output name
    g_img_path = output_lib + '/' + 'gr_' + img_name + output_ext  # output path to g_img
    # mask_img_path = output_lib + '/' + 'mask_' + img_name + output_ext  # output path to mask

    img = cv2.imread(img_path)[vlc:vuc, hlc:hrc]  # cv2 read & crop
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to hsv
    mask = cv2.inRange(hsv, (35, 25, 25), (80, 255, 255))  # mask by slicing the green spectrum

    # apply the masking:
    imask = mask > 0
    g_img = np.zeros_like(img, np.uint8)
    mask_g = np.zeros_like(img, np.uint8)

    # make white background
    g_img.fill(255)
    g_img[imask] = img[imask]
    mask_g[~imask] = img[~ imask]
    mask_g[mask_g != 0] = 255

    # erosion
    g_img, mask, img = erode_func(img, mask_g)

    if sbool:  # save to directory
        cv2.imwrite(g_img_path, g_img)
        # cv2.imwrite(mask_img_path, mask_g)

    return g_img, mask, img


def erode_function(img, mask, n=2, ite=3):
    kernel = np.ones((n, n), np.uint8)
    mask_erosion = cv2.erode(mask, kernel, iterations=ite)
    imask_erosion = mask_erosion == 0
    g_img = np.zeros_like(img, np.uint8)
    g_img.fill(255)
    g_img[imask_erosion] = img[imask_erosion]
    return g_img, mask_erosion, img


def convert_images_to_masked(input_dir, output_dir, mask_func, erode_func):

    input_files_names = glob.glob(input_dir + '/*')
    logging.debug(f'Number of frames to mask: {len(input_files_names)}.')
    for file in input_files_names:
        mask_func(file, erode_func, output_lib=output_dir, sbool=True)


def main():

    # set parameters
    path_to_stream = 'http://192.168.11.115:8080/?action=streaming'  # Wi-Fi Broadcast.

    # set working directories, input and output.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data_directory = dir_path + '/data'
    path_to_masked_data_directory = dir_path + '/data-m2'

    # delete content from existing directories.
    #delete_data(path_to_data_directory)  # deletes content of the data library
    delete_data(path_to_masked_data_directory)  # deletes content of the masked data library
    # create a video.
    #record_video(30, path_to_stream, path_to_data_directory)  # records a video
    # apply mask.
    convert_images_to_masked(path_to_data_directory, path_to_masked_data_directory, mask_fun, erode_function)


if __name__ == "__main__":
    main()
