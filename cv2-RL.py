""" Raspi-plant interacting module """

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
import docker
import open3d as o3d


# parser
parser = argparse.ArgumentParser(description="Raspi-plant")
parser.add_argument("-rec", "--record", default=28, help="make a video", type=int)
parser.add_argument("-deb", "--debug", default=False, help="end streaming", type=bool)
parser.add_argument("-col", "--colmap", default=False, help="run colmap from docker container", type=bool)
parser.add_argument("-dele", "--delete", default=False, help="delete working directory", type=bool)
parser.add_argument("-mask", "--masking", default=False, help="mask images", type=bool)
parser.add_argument("-out", "--outliers", default=False, help="remove outliers: white pixels and isolated points", type=bool)
parser.add_argument("-p", "--poisson", default=False, help="create poisson mesh and save file", type=bool)


FLAGS = parser.parse_args()

# broadcasting Global
RASPI_USER = "pi"
RASPI_PI_WIFI = "192.168.11.115"
RASPI_PI_ETHERNET = "10.150.180.52"
RASPI_PWD = '1234'
RASPI_BROADCAST_PORT = 8085
SSH_PORT = 22
PATH_TO_STREAM = f'http://{RASPI_PI_ETHERNET}:{str(RASPI_BROADCAST_PORT)}/?action=streaming'  # Wi-Fi Broadcast.

# container's structure.
DATASET_PATH = "/root/data"
DATABASE_PATH = "/root/data/output/database.db"
IMAGESET_PATH = "/root/data/dataset-m"
OUTPUT_PATH = "/root/data/output"
SPARSE_PATH = "/root/data/output/0"
NEW_SPARSE_PATH = "/root/data/output/sparse"
DENSE_PATH = "/root/data/output"
DENSE_PLY_PATH = "/root/data/output/dense.ply"

# create logger with 'spam_application'
logging.getLogger('Raspi- application')
logging.basicConfig(stream=sys.stdout, filemode='a', level=logging.DEBUG)


def set_working_directories(wd_path, debug_mode=False):
    """ function receives the working directory path <wd_path>, searches for datasets' occurrences, and then creates
    two new directories with a successive serial number according to the last one found in the search.
    :param
    wd_path: string.
    :param debug_mode: if debug_mode=True it doesn't create new directories.
    :return 3-tuple of
    paths for the new directories which was created.
    """
    data_lib_path = wd_path + '/d/'
    list_subfolders_with_paths = [f.path.split('/')[-1] for f in os.scandir(wd_path + '/d/') if f.is_dir()]
    p_data, p_data_m = re.compile('^dataset[0-9]{1,3}'), re.compile('^dataset-m[0-9]{1,3}')  # new datasets' path
    # p_output = re.compile('^output[0-9]{1,3}')
    len_1 = len("dataset")
    len_2 = len("dataset-m")
    # len_3 = len("output")
    list_data_libs = sorted([int(s[len_1:]) for s in list_subfolders_with_paths if p_data.match(s)])
    list_data_m_libs = sorted([int(s[len_2:]) for s in list_subfolders_with_paths if p_data_m.match(s)])
    # list_output_libs = sorted([int(s[len_3:]) for s in list_subfolders_with_paths if p_output.match(s)])

    assert len(list_data_libs) == len(list_data_m_libs), "output libraries directories mismatched:" \
                                                         " please delete manually."

    if not list_data_libs:  # no datasets directories exist yet.

        dataset_serial = str(0)
        logging.debug(f'dataset_serial: {dataset_serial}.')

    else:
        if debug_mode:
            dataset_serial = str(list_data_libs[-1])
        else:
            dataset_serial = str(list_data_libs[-1] + 1)

    logging.debug(f'dataset_serial: {dataset_serial}.')
    dataset_name = 'dataset' + dataset_serial
    dataset_m_name = 'dataset-m' + dataset_serial
    output_name = 'output' + dataset_serial
    path_data, path_data_m, path_output = tuple(
        ''.join(i) for i in zip(tuple([data_lib_path] * 3),
                                tuple((dataset_name, dataset_m_name, output_name))))

    if not debug_mode:  # create new directories.
        os.mkdir(path_data)
        os.mkdir(path_data_m)
        os.mkdir(path_output)

    return path_data, path_data_m, path_output, dataset_serial


def SSH_open_camera(client, sharpness, brightness, contrast, fps, res_x, res_y, port=8080):
    """SSH script to start broadcasting on the remote host.
     :params: <mjpg-streamer parametetruers>"""
    OPENING_CAMERA_CMD = f"mjpg_streamer -i \"input_raspicam.so -br {brightness} -co {contrast} -sh {sharpness} -x {res_x}  -y {res_y}  -fps {fps}\" -o \'output_http.so -p {port}\'"
    logging.debug('opening_camera_CMD: {0}'.format(OPENING_CAMERA_CMD))
    stdin, stdout, stderr = client.exec_command(OPENING_CAMERA_CMD)
    time.sleep(1)  # bug fix: AttributeError
    for i in reversed(range(10)):
        time.sleep(1)
        logging.debug(f"sleeping time..: {i}")

    return stdin, stdout, stderr


def SSH_shutdown_camera(client):
    """SSH script to end broadcasting on the remote host"""
    stdin, stdout, stderr = client.exec_command("ps aux")
    time.sleep(1)  # bug fix: AttributeError
    data = stdout.readlines()  # retrieve processes output.
    for line in data:
        if line.find('mjpg_streamer') != -1:
            process_to_kill = re.findall('\d+', line)[0]
            stdin, stdout, stderr = client.exec_command(f"kill {process_to_kill}")  # execute process.
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
                # decode to colored image
                i = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                datetimeobj = datetime.now()  # get time stamp
                cv2.imwrite(path_to_data + '/img' + str(datetimeobj) + '.jpg', i)
                if cv2.waitKey(1) == 27 or (datetimeobj - time_start).seconds > length_secs:  # if user  hit esc
                    logging.debug('End recording.')
                    break  # exit program
    else:
        logging.error("Received unexpected status code {}".format(r.status_code))


def mask_images(img_path, erode_func, output_lib, params, output_ext=".jpg"):
    """:param img_path: string, path to the image.
       :param erode_func: function, masking function.
       :param output_lib: string, path to output directory.
       :param output_ext: string, type of output image, default jpg.
       :param params: dictionary mask- parameters.

       :returns np.arrays of g_img, mask_g and img

       """

    # set cropping and masking parameters:
    hlc, hrc, vlc, vuc = params["hlc"], params["hrc"], params["vlc"], params["vuc"]
    n, n_iter = params["n"], params["n_iter"]

    # extract img name:
    img_name = os.path.splitext(Path(img_path).name)[0]

    # output name
    g_img_path = output_lib + '/' + 'gr_' + img_name + output_ext  # output path to g_img

    img = cv2.imread(img_path)[vlc:vuc, hlc:hrc]  # cv2 read & crop
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to hsv
    mask = cv2.inRange(hsv, (35, 25, 25), (80, 255, 255))  # mask by slicing the green spectrum
    # mask = cv2.inRange(hsv, (0, 42, 0), (179, 255, 255)) # everything except white

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
    g_img, mask, img = erode_func(img, mask_g, n, n_iter)

    # save to directory
    cv2.imwrite(g_img_path, g_img)

    return g_img, mask, img


def erode_function(img, mask, n, ite):
    kernel = np.ones((n, n), np.uint8)
    mask_erosion = cv2.erode(mask, kernel, iterations=ite)
    imask_erosion = mask_erosion == 0
    g_img = np.zeros_like(img, np.uint8)
    g_img.fill(255)
    g_img[imask_erosion] = img[imask_erosion]
    return g_img, mask_erosion, img


def convert_images_to_masked(input_dir, output_dir, mask_func, erode_func, mask_params):
    input_files_names = glob.glob(input_dir + '/*')
    logging.debug(f'Number of frames to mask: {len(input_files_names)}.')
    for file in input_files_names:
        mask_func(file, erode_func, output_dir, mask_params)


def run_colmap(client, cmd_dict, mount_dict, wd):
    """ dockerized COLMAP"""
    for colmap_command, command in cmd_dict.items():
        logging.info(f"executing colmap: {colmap_command}")
        _execute_colmap_command(client, command, mount_dict, wd)


def _execute_colmap_command(cl, cmd, mount_dict, wd, container_name='colmap:test'):
    return cl.containers.run(container_name, cmd, volumes=mount_dict, working_dir=wd, runtime="nvidia", detach=False,
                             auto_remove=True)


def create_outliers_list(pcd):
    # Convert open3d format to numpy array
    pcd_colors = np.asarray(pcd.colors) * 255
    pcd_colors_summed = np.expand_dims(pcd_colors.sum(axis=1), axis=1)
    return np.where(np.any(pcd_colors_summed > 720, axis=1))[0].tolist()


def remove_outliers(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud = cloud.select_by_index(ind, invert=False)
    return inlier_cloud.remove_radius_outlier(nb_points=100, radius=0.1)[0], outlier_cloud


def poisson_reconstruction(pcd):
    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, width=0, scale=1.1, linear_fit=False)[0]


def main():
    # set working directories, input and output.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_data_directory, path_to_masked_data_directory, path_to_output_dir, serial = set_working_directories(
        dir_path, debug_mode=FLAGS.debug)

    input_dense = path_to_output_dir + '/dense.ply'
    output_inlier_cloud = path_to_output_dir + '/dense_inlier.ply'
    output_poisson = path_to_output_dir + '/poisson.ply'

    # logging control
    logging.debug(f"Path to the data directory: {path_to_data_directory}")
    logging.debug(f"Path to the masked-data directory: {path_to_masked_data_directory}")
    logging.debug(f"Path to output directory: {path_to_output_dir}")

    # delete content from existing directories.
    if FLAGS.delete:
        delete_data(path_to_data_directory)  # deletes content of the data library
        delete_data(path_to_masked_data_directory)  # deletes content of the masked data library

    # ssh connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(RASPI_PI_WIFI, SSH_PORT, RASPI_USER, RASPI_PWD)

    """ record a video. """
    if FLAGS.record and FLAGS.record != 0:
        SSH_open_camera(ssh, sharpness=50, brightness=50, contrast=60, fps=2, res_x=1080, res_y=720,
                        port=RASPI_BROADCAST_PORT)
        record_video(28, PATH_TO_STREAM, path_to_data_directory)  # records a video
        SSH_shutdown_camera(ssh)

    logging.info(f"SSH transport is: {ssh.get_transport().active}")
    logging.info("closing SSH connection: ")
    ssh.close()
    logging.info("SSH transport is now closed")

    """ apply mask. """
    if FLAGS.masking:
        mask_params = {"hlc": 150, "hrc": 950, "vlc": 0, "vuc": 730, "n": 8, "n_iter": 20}
        convert_images_to_masked(path_to_data_directory, path_to_masked_data_directory, mask_images, erode_function,
                             mask_params)

    """colmap in docker"""
    if FLAGS.colmap:
        client = docker.from_env()  # connect to docker daemon

        path_to_mount = path_to_data_directory
        path_masked_to_mount = path_to_masked_data_directory
        path_output_to_mount = path_to_output_dir

        destination_mount = IMAGESET_PATH[:-2]
        destination_masked_mount = IMAGESET_PATH
        destination_output_mount = OUTPUT_PATH

        mount_dict = {path_to_mount: {'bind': destination_mount, 'mode': 'rw'},
                      path_masked_to_mount: {'bind': destination_masked_mount, 'mode': 'rw'},
                      path_output_to_mount: {'bind': destination_output_mount, 'mode': 'rw'}}

        CMD1 = f"colmap feature_extractor --database_path {DATABASE_PATH} --image_path {IMAGESET_PATH}"
        CMD2 = f"colmap exhaustive_matcher \
            --database_path {DATABASE_PATH}"
        CMD3 = f"colmap mapper \
            --database_path {DATABASE_PATH} \
            --image_path {IMAGESET_PATH} \
            --output_path {OUTPUT_PATH}"
        CMD4 = f"colmap image_undistorter \
            --image_path {IMAGESET_PATH} \
            --input_path {SPARSE_PATH} \
            --output_path {DENSE_PATH} \
            --output_type COLMAP"
        CMD5 = f"colmap patch_match_stereo \
            --workspace_path {DENSE_PATH} \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true"
        CMD6 = f"colmap stereo_fusion \
            --workspace_path {DENSE_PATH} \
            --workspace_format COLMAP \
            --input_type geometric \
            --output_path {DENSE_PLY_PATH}"
        CMD7 = f"colmap model_converter \
            --input_path {SPARSE_PATH} \
            --output_path {SPARSE_PATH} \
            --output_type TXT"

        colmap_cmds = {"feature_extractor": CMD1, "exhaustive_matcher": CMD2, "mapper": CMD3, "image_undistorter": CMD4,
                       "patch_match_stereo": CMD5, "stereo_fusion": CMD6, "model_converter": CMD7}

        run_colmap(client, colmap_cmds, mount_dict, DATASET_PATH)

    """outliers removal"""
    if FLAGS.outliers:
        pcd = o3d.io.read_point_cloud(input_dense)
        inlier_cloud, outlier_cloud = remove_outliers(pcd, create_outliers_list(pcd))
        logging.info(f"number of points removed by knn:{len(pcd.points)-len(inlier_cloud.points)}")
        #  save to file
        o3d.io.write_point_cloud(output_inlier_cloud, inlier_cloud, write_ascii=True, compressed=False, print_progress=False)

    """poisson meshing"""
    if FLAGS.poisson:
        pcd = o3d.io.read_point_cloud(output_inlier_cloud) if Path(output_inlier_cloud).is_file() \
            else o3d.io.read_point_cloud(input_dense)
        poisson_mesh = poisson_reconstruction(pcd)
        o3d.io.write_triangle_mesh(output_poisson, poisson_mesh, write_ascii=True, compressed=False)


if __name__ == "__main__":
    main()
