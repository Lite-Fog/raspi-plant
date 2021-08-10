import cv2
import requests
import numpy as np
from datetime import datetime
import os
import sys
import glob
from pathlib import Path
import logging
import re
import paramiko
import time
import argparse
# from PIL import Image

# connect to docker daemon
import docker


""" Raspi-plant interacting module """

# create logger with 'spam_application'
logging.getLogger('Raspi_application')
logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.DEBUG)


def set_working_directories(wd_path, debug_mode=False):
    """ function receives the working directory path <wd_path>, searches for all occurrences of dataset written in
    the applicable form, and then creates 2 new directories with a successive serial number according to the last one
    found in the search.
    :param wd_path: string.
    :param debug_mode: in debug_mode=True do not create new directories.
    :return 3-tuple of paths for the new directories which was created.
    """
    data_lib_path = wd_path + '/d/'
    list_subfolders_with_paths = [f.path.split('/')[-1] for f in os.scandir(wd_path + '/d/') if f.is_dir()]
    p_data, p_data_m = re.compile('^dataset[0-9]{1,3}'), re.compile('^dataset-m[0-9]{1,3}')  # new datasets' path
    p_output = re.compile('^output[0-9]{1,3}')
    len_1 = len("dataset")
    len_2 = len("dataset-m")
    len_3 = len("output")
    list_data_libs = sorted([int(s[len_1:]) for s in list_subfolders_with_paths if p_data.match(s)])
    list_data_m_libs = sorted([int(s[len_2:]) for s in list_subfolders_with_paths if p_data_m.match(s)])
    list_output_libs = sorted([int(s[len_3:]) for s in list_subfolders_with_paths if p_output.match(s)])

    assert len(list_data_libs) == len(list_data_m_libs), "output libraries directories mismatched:" \
                                                         " please delete manually."

    if not list_data_libs:  # none datasets directories exist yet.

        path_data, path_data_m = data_lib_path + 'dataset0', data_lib_path + 'dataset-m0'
        path_output = data_lib_path + 'output0'

        logging.debug(f'dataset_serial: {0}.')

        if not debug_mode:
            os.mkdir(path_data)
            os.mkdir(path_data_m)
            os.mkdir(path_output)
        return path_data, path_data_m, path_output

    else:
        dataset_serial = str(list_data_libs[-1] + 1)

        logging.debug(f'list_data_libs={list_data_libs}.')
        logging.debug(f'dataset_serial: {dataset_serial}.')

        dataset_name = 'dataset' + dataset_serial
        dataset_m_name = 'dataset-m' + dataset_serial
        output_name = 'output' + dataset_serial
        path_data, path_data_m, path_output = tuple(
            ''.join(i) for i in zip(tuple((data_lib_path, data_lib_path, data_lib_path)), tuple((dataset_name, dataset_m_name, output_name))))
        if not debug_mode:
            os.mkdir(path_data)
            os.mkdir(path_data_m)
            os.mkdir(path_output)
        return path_data, path_data_m, path_output


def SSH_open_camera(client, sharpness, brightness, contrast, fps, res_x, res_y, port=8080):
    """SSH script to start broadcasting on the remote host.
     :params: <mjpg-streamer parameters>"""
    OPENING_CAMERA_CMD = f"mjpg_streamer -i \"input_raspicam.so -br {brightness} -co {contrast} -sh {sharpness} -x {res_x}  -y {res_y}  -fps {fps}\" -o \'output_http.so -p {port}\'"
    logging.debug('opening_camera_CMD: {0}'.format(OPENING_CAMERA_CMD))
    # stdin, stdout, stderr = client.exec_command("./startcam.sh")
    stdin, stdout, stderr = client.exec_command(OPENING_CAMERA_CMD)
    time.sleep(1)  # bug fix: AttributeError
    for i in reversed(range(10)):
        time.sleep(1)
        logging.debug(f"sleeping time... {i}")

    return stdin, stdout, stderr


def SSH_shutdown_camera(client):
    """SSH script to end broadcasting on the remote host"""
    stdin, stdout, stderr = client.exec_command("ps aux")
    time.sleep(1)  # bug fix: AttributeError
    data = stdout.readlines()
    for line in data:
        if line.find('mjpg_streamer') != -1:
            process_to_kill = re.findall('\d+', line)[0]
            stdin, stdout, stderr = client.exec_command(f"kill {process_to_kill}")
            time.sleep(1)  # bug fix: AttributeError
            logging.debug(f'process {process_to_kill} is now gone.')
            return stdin, stdout, stderr


def delete_data(path):
    """ function deletes library
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
                jpg = bytes_loc[a:b + 2]  # actual image
                bytes_loc = bytes_loc[b + 2:]  # other information
                # decode to colored image ( another option is cv2.IMREAD_GRAYSChttps://github.com/ryanfb/docker_visualsfm.gitALE)
                i = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                datetimeobj = datetime.now()  # get time stamp
                cv2.imwrite(path_to_data + '/img' + str(datetimeobj) + '.jpg', i)
                if cv2.waitKey(1) == 27 or (datetimeobj - time_start).seconds > length_secs:  # if user  hit esc
                    logging.debug('End recording.')
                    break  # exit program
    else:
        print("Received unexpected status code {}".format(r.status_code))


def mask_images(img_path, erode_func, output_lib, sbool=False, output_ext=".jpg"):
    """:param img_path: string, path to the image.
       :param erode_func: function, masking function.
       :param output_lib: string, path to output directory.
       :param sbool: boolean, if True it will save the masked image, default Fahttps://github.com/ryanfb/docker_visualsfm.gitlse.
       :param output_ext: string, type of output image, default jpg.

       :returns np.arrays of g_img, mask_g and img

       """
    # set cropping parameters:
    hlc, hrc = 0, 1000
    vlc, vuc = 0, 670
    # extract img name:
    img_name = os.path.splitext(Path(img_path).name)[0]

    # output name
    g_img_path = output_lib + '/' + 'gr_' + img_name + output_ext  # output path to g_img

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

    return g_img, mask, img


def erode_function(img, mask, n=2, ite=1):
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
    """ raspi- plant. """

    DEBUG_MODE = True

    # set broadcasting parameters
    RASPI_USER = "pi"
    RASPI_PI_WIFI = "192.168.11.115"
    RASPI_PI_ETHERNET = "10.150.180.52"
    RASPI_PWD = 'Mancave3090!'
    RASPI_BROADCAST_PORT = 8085
    SSH_PORT = 22
    path_to_stream = f'http://{RASPI_PI_ETHERNET}:{str(RASPI_BROADCAST_PORT)}/?action=streaming'  # Wi-Fi Broadcast.

    parser = argparse.ArgumentParser(description="Raspi-plant")
    parser.add_argument("-on", "--on", help="start streaming", nargs='+', type=int)
    parser.add_argument("-off", "--off", help="end streaming", nargs='+', type=str)
    parser.add_argument("-r", "--record", help="record a video", nargs='+', type=int)

    parser.add_argument("-sa", "--algorithm",
                        help="choose algorithm: False (default) for full search, True for LSH", nargs='+',
                        type=bool, default=False)
    parser.add_argument("-R", "--report", help="create visual report", action="store_true")

    args = parser.parse_args()

    # ssh connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(RASPI_PI_ETHERNET, SSH_PORT, RASPI_USER, RASPI_PWD)

    # set working directories, input and output.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data_directory, path_to_masked_data_directory, path_to_output_dir = set_working_directories(dir_path, debug_mode=DEBUG_MODE)

    # control
    logging.debug(f"Path to the data directory: {path_to_data_directory}")
    logging.debug(f"Path to the masked-data directory: {path_to_masked_data_directory}")
    logging.debug(f"Path to output directory: {path_to_output_dir}")

    # delete content from existing directories.
    # delete_data(path_to_data_directory)  # deletes content of the data library
    # delete_data(path_to_masked_data_directory)  # deletes content of the masked data library

    """ create a video. """
    #SSH_open_camera(ssh, sharpness=50, brightness=50, contrast=60, fps=2, res_x=1080, res_y=720, port=RASPI_BROADCAST_PORT)
    #record_video(28, path_to_stream, path_to_data_directory)  # records a video
    #SSH_shutdown_camera(ssh)

    logging.info(f"SSH transport is: {ssh.get_transport().active}")
    logging.info("closing SSH.. ")
    ssh.close()
    logging.info("SSH transport is now closed")

    """ apply mask. """
    convert_images_to_masked(path_to_data_directory, path_to_masked_data_directory, mask_images, erode_function)

    """ docker VisualSFM"""
    #dclient = docker.from_env()
    #print(dclient.containers.run('ryanfb/visualsfm', 'echo hello world'))


if __name__ == "__main__":
    main()
