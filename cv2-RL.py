import cv2
import requests
import numpy as np
from datetime import datetime
import os
import glob
from pathlib import Path
from PIL import Image


def delete_data(path):
    """ function deletes data library
    :param path: string"""
    files = glob.glob(path + '/*')
    for f in files:
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
                    break  # exit program
    else:
        print("Received unexpected status code {}".format(r.status_code))


def mask_img(img_path, output_lib, sbool=False, output_ext=".jpg"):
    """:param img_path: string, path to the image.
       :param output_lib: string, path to output directory.
       :param sbool: boolean, if True it will save the masked image, default False.
       :param output_ext: string, type of output image, default jpg.
       :returns np.arrays of green and mask_g
       """

    img_name = os.path.splitext(Path(img_path).name)[0]

    # output names
    green_img_path = output_lib + '/' + 'gr_' + img_name + output_ext  # output green img path

    img = cv2.imread(img_path)[:, 150:1050]  # cv2 read
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to hsv
    mask = cv2.inRange(hsv, (35, 25, 25), (80, 255, 255))  # mask by green spectrum

    # slice out the green
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    mask_g = np.zeros_like(img, np.uint8)

    # make white background
    green.fill(255)

    green[imask] = img[imask]
    mask_g[~imask] = img[~ imask]

    if sbool:  # save to directory if sbool == True
        cv2.imwrite(green_img_path, green)
        print(green_img_path)

    return green, mask_g


def convert_images_to_masked(input_dir, output_dir, mask_function):

    input_files_names = glob.glob(input_dir + '/*')
    print(len(input_files_names))
    for file in input_files_names:
        mask_function(file, output_lib=output_dir, sbool=True)
        
        
#def jpeg-to-jpg(path):
#    img = Image.open(path)
#    rgb_img = img.convert('RGB')
#    rgb_img.save('image.jpg')


def main():

    # set parameters
    ## path_to_stream = 'http://192.168.11.115:8080/?action=streaming'  # Wi-Fi Broadcast.

    # set working directories, input and output.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data_directory = dir_path + '/data'
    path_to_masked_data_directory = dir_path + '/data-m'


    ## delete_data(path_to_data_directory)  # deletes content of the data library
    delete_data(path_to_masked_data_directory)  # deletes content of the masked data library
    ## record_video(30, path_to_stream, path_to_data_directory)

    convert_images_to_masked(path_to_data_directory, path_to_masked_data_directory, mask_img)


if __name__ == "__main__":
    main()
