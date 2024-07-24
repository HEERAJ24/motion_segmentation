# Transform the coordinates in RGB camera frame to event camera frame
# RGB camera is cam0, Event camera 1 is cam1 and event camera 2 is cam2
import numpy as np
import cv2
import os
import json
from scipy.spatial.transform import Rotation as R
import trimesh

# read .npy file

g = {
    'data_path': '/home/eventcamera/data/dataset/Jul1/test_klt_2/vicon_data/',
    'json_path_camera_sys': '/home/eventcamera/data/dataset/Jul1/test_klt_2/vicon_data/event_cam_sys.json',
    'json_path_object': '/home/eventcamera/data/dataset/Jul1/test_klt_2/vicon_data/object1.json',
    'json_path_event_cam_left': '/home/eventcamera/data/dataset/Jul1/test_klt_2/event_cam_left/e2calib/',
    'json_path_event_cam_right': '/home/eventcamera/data/dataset/Jul1/test_klt_2/event_cam_right/e2calib/',
    'rgb_image_path': '/home/eventcamera/data/dataset/Jul1/test_klt_2/rgb/',
    "obj_path": '/home/eventcamera/data/KLT/obj_000003.ply',
    "obj_name": "test_klt",
    "output_dir": '/home/eventcamera/data/dataset/Jul1/test_klt_2/annotation/'
}

selected_path = g

data_path = selected_path['data_path']
json_path_camera_sys = selected_path['json_path_camera_sys']
json_path_object = selected_path['json_path_object']
json_path_event_cam_left = selected_path['json_path_event_cam_left']
json_path_event_cam_right = selected_path['json_path_event_cam_right']
output_dir = selected_path["output_dir"]

for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        os.unlink(file_path)


# with open(json_path_camera, 'r') as f:
#    data_camera = json.load(f)

rgb_image_path = selected_path['rgb_image_path']
rgb_timestamp = os.listdir(rgb_image_path)
rgb_timestamp.sort()

event_cam_left_timestamp = os.listdir(json_path_event_cam_left)
event_cam_left_timestamp.sort()

event_cam_right_timestamp = os.listdir(json_path_event_cam_right)
event_cam_right_timestamp.sort()
H_cam_optical_2_base = np.eye(4)

with open(json_path_object, 'r') as file:
    object_array = json.load(file)
# extract only timestamp in a numpy array from dictionary loaded_array
timestamp_object = []

for k, v in object_array.items():
    timestamp_object.append(v['timestamp'])
timestamp_object = np.array(timestamp_object)


def find_closest_elements(A, B):
    result = {}

    for a in A:
        closest_b = min(B, key=lambda x: abs(x - a))
        result[a] = closest_b
        B.remove(closest_b)

    return result


def remove_extension_and_convert_to_int(arr):
    # Remove ".png" extension and convert to integers
    modified_arr = [int(file_name[:-4]) for file_name in arr if file_name.endswith('.png')]
    return modified_arr


# Compute vicons coordinates of object corresponding to rgb image. The timestamp of the rgb image is used to find the
# nearest timestamp in vicon data of object
rgb_timestamp = remove_extension_and_convert_to_int(rgb_timestamp)
event_cam_left_timestamp = remove_extension_and_convert_to_int(event_cam_left_timestamp)
event_cam_right_timestamp = remove_extension_and_convert_to_int(event_cam_right_timestamp)
# convert list of strings to list of integers
# Associate timestamps in both event cameras to rgb camera timestamps.
timestamp_object = list(map(int, timestamp_object))
result_dict = find_closest_elements(rgb_timestamp,
                                    timestamp_object)  # Output in format (rgb_timestamp, timestamp_object)
#vicon_coord = []
timestamps_closest_object = list(result_dict.values())
result_dict2 = find_closest_elements(rgb_timestamp, event_cam_left_timestamp)
result_dict3 = find_closest_elements(rgb_timestamp, event_cam_right_timestamp)

keys_to_remove = []

for key, value in result_dict2.items():
    deviation = abs(int(key) - int(value))

    if deviation > 10000000:
        keys_to_remove.append(key)

for key in keys_to_remove:
    del result_dict2[key]

keys_remove = []

for keys, values in result_dict3.items():
    deviations = abs(int(keys) - int(values))
    if deviations > 10000000:
        keys_remove.append(keys)

for keys in keys_remove:
    del result_dict3[keys]

timestamp_closest_ec_left = list(result_dict2.values())
timestamp_closest_ec_right = list(result_dict3.values())
translations_with_timestamps = {
    timestamp: np.array(object_array[str(timestamp)]["translation"])
    for timestamp in timestamps_closest_object}
rotations_with_timestamps = {
    timestamp: np.array(object_array[str(timestamp)]["rotation"])
    for timestamp in timestamps_closest_object
}
'''
# update the x and y coordinates of all translation in the dictionary by -1. Replace this values in object_array
for key, value in translations_with_timestamps.items():
    value[0] = value[0] + 0.0  #- 0.14
    value[1] = value[1] - 0.03 #- 0.1
    value[2] = value[2] - 0.05  #- 0.01
    object_array[str(key)]["translation"] = value
'''
# Transformation matrix obtained from eye in hand calibration
H_cam_vicon_2_cam_optical = np.array([[0.00563068, 0.03006136, 0.9995322, -0.05282819],
                                      [-0.99982796, -0.01749663, 0.00615856, 0.03674293],
                                      [0.01767357, -0.99939491, 0.02995767, 0.00407536],
                                      [0., 0., 0., 1.]])

# RGB camera
#params = [1.81601107e+03, 1.81264445e+03, 1.00383169e+03, 7.16010695e+02]
params = [2001.0250442780605, 2001.2767496004499, 970.1619103491635, 684.6369964551955]
# params = [2592.7798180209766, 2597.1074116646814, 1121.2441077660412, 690.1066893999352]
camera_matrix = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
#distortion_coefficients = np.array([-1.76581808e-01, 1.06210912e-01, -1.55074994e-04,
#                                    5.03366350e-04, -4.07696624e-02])
distortion_coefficients = np.array(
    [-0.16662668463462832, 0.09713587034707222, 0.00044649384097793574, 0.0006466275306382167])

camera_mtx_cam1 = np.array(
    [[726.8015187965628, 0, 275.7447750580928], [0, 725.2525019712525, 212.00552936834734], [0, 0, 1]])
# [[718.9289498879248, 0, 287.4206641081329], [0, 718.8476596505732, 232.6402787336837], [0, 0, 1]])
distortion_coeffs_cam1 = np.array(
    [-0.40846674226069263, 0.2067801786931747, 0.007207741354236219, 0.006261397706848751])
#[-0.3094967913882128, -0.10722657430965295, 0.008512403913427787, 0.000592616793055609])

camera_mtx_cam2 = np.array(
    [[756.1204892251881, 0, 283.43129078329116], [0, 754.9663110831402, 227.0824683283377], [0, 0, 1]])
#[[745.1353300950308, 0,  291.1070763508334], [0, 747.3744176138202, 245.89026445203564], [0, 0, 1]])
distortion_coeffs_cam2 = np.array(
    [-0.4107919894947944, 0.20240048255503923, 0.006230890760180581, 0.00970389087313616])
#[-0.2496983957244161, -0.2978060510925673, 0.009342174824708725, 0.00328860522240014])

#=============================================================================
# Transformation matrix from camera 1 to camera 0 and camera 2 to camera 1. cam0 is rgb camera, cam1 is event camera 1
# =============================================================================
quat_cam2_cam1 = [0.00026235, -0.00376717, -0.00030698, 0.99999282]
quat_cam1_cam0 = [-0.00916228, -0.04119687, 0.0005243, 0.9991089]
R_cam1_cam0 = R.from_quat(quat_cam1_cam0).as_matrix()

# Transformation matrix from camera 1 to camera 0
H_cam1_2_rgb = np.array([[0.9988487792874476, 0.0047032240817731635, 0.04773882905149164, 0.05547200068236508],
                         [-0.00480866672973969, 0.9999862455691012, 0.0020941339237770754, 0.004239383045238393],
                         [-0.04772832324996561, -0.0023212832324058085, 0.9988576569281016, 0.007983029405103982],
                         [0.0, 0.0, 0.0, 1.0]])

H_cam2_cam1 = np.array([[0.9999747401826854, 0.0016840625425892184, -0.006905282755123367, -0.1029018195612104],
                        [-0.0017160652568493136, 0.999987803370837, -0.004631223338073897, 0.0005686198826895407],
                        [0.006897399264200306, 0.004642956270043212, 0.9999654338228244, 0.00848555474834492],
                        [0.0, 0.0, 0.0, 1.0]])

# =============================================================================

transformations = {}
# Read the vicon coordinates of the even camera system. Traverse through the coordinates
with open(json_path_camera_sys, 'r') as f:
    data = json.load(f)
for i, v in data.items():
    if i == str(len(data) - 1):
        continue
    translation = data[str(i)]['translation']
    rotation_quat = data[str(i)]['rotation']

    # get rotation matrix from quaternion
    rotation = R.from_quat(rotation_quat).as_matrix()
    # make homogeneous transformation matrix
    H_base_2_cam_vicon = np.eye(4)
    H_base_2_cam_vicon[:3, :3] = rotation
    H_base_2_cam_vicon[:3, 3] = translation

    # make homogeneous transformation matrix from base to camera optical frame
    H_base_2_cam_optical = np.matmul(H_base_2_cam_vicon, H_cam_vicon_2_cam_optical)

    # invert H_vicon_2_cam_optical to get H_cam_optical_2_vicon
    H_cam_optical_2_base = np.eye(4)
    H_cam_optical_2_base[:3, :3] = np.transpose(H_base_2_cam_optical[:3, :3])
    H_cam_optical_2_base[:3, 3] = -np.matmul(np.transpose(H_base_2_cam_optical[:3, :3]), H_base_2_cam_optical[:3, 3])
    t_x = object_array[str(v['timestamp'])]['translation'][0]
    t_y = object_array[str(v['timestamp'])]['translation'][1]
    t_z = object_array[str(v['timestamp'])]['translation'][2]
    r_x = object_array[str(v['timestamp'])]['rotation'][0]
    r_y = object_array[str(v['timestamp'])]['rotation'][1]
    r_z = object_array[str(v['timestamp'])]['rotation'][2]
    r_w = object_array[str(v['timestamp'])]['rotation'][3]
    rotation = R.from_quat([r_x, r_y, r_z, r_w]).as_matrix()

    '''H_v_2_point = np.array([[1, 0, 0, t_x],
                            [0, 1, 0, t_y],
                            [0, 0, 1, t_z],
                            [0, 0, 0, 1]])'''

    H_v_2_point = np.eye(4)
    H_v_2_point[:3, :3] = rotation
    H_v_2_point[:3, 3] = [t_x, t_y, t_z]

    H_cam_optical_2_point = np.matmul(H_cam_optical_2_base, H_v_2_point)
    t_cam_optical_2_point = H_cam_optical_2_point[:3, 3]
    r_cam_optical_2_point = H_cam_optical_2_point[:3, :3]
    H_base_2_cam_optical = np.matmul(H_base_2_cam_vicon, H_cam_vicon_2_cam_optical)
    # invert H_vicon_2_cam_optical to get H_cam_optical_2_vicon
    H_cam_optical_2_base = np.eye(4)
    H_cam_optical_2_base[:3, :3] = np.transpose(H_base_2_cam_optical[:3, :3])
    H_cam_optical_2_base[:3, 3] = -np.matmul(np.transpose(H_base_2_cam_optical[:3, :3]), H_base_2_cam_optical[:3, 3])
    # Compute translation t_cam_optical_2_base
    t_cam_optical_2_base = H_cam_optical_2_base[:3, 3]
    # t_cam_optical_2_base = np.transpose(H_base_2_cam_optical[:3, :3])
    H_rgb_2_point = H_cam_optical_2_point
    # project point (x,y,z) in cam0 coordinate to cam1 coordinate
    point_cam0 = np.array([
        [1, 0, 0, t_cam_optical_2_base[0]],
        [0, 1, 0, t_cam_optical_2_base[1]],
        [0, 0, 1, t_cam_optical_2_base[2]],
        [0, 0, 0, 1]])

    H_cam1_2_point = np.matmul(H_cam1_2_rgb, H_rgb_2_point)

    t_cam1_2_point = H_cam1_2_point[:3, 3]
    H_cam2_2_point = np.matmul(H_cam2_cam1, H_cam1_2_point)
    t_cam2_2_point = H_cam2_2_point[:3, 3]
    transformations[str(data[str(i)]['timestamp'])] = {'H_cam_optical_2_base': H_cam_optical_2_base.tolist(),
                                                       'H_cam_optical_2_point': H_cam_optical_2_point.tolist(),
                                                       'H_base_2_cam_vicon': H_base_2_cam_vicon.tolist(),
                                                       't_cam_optical_2_point': t_cam_optical_2_point.tolist(),
                                                       't_cam_optical_2_base': t_cam_optical_2_base.tolist(),
                                                       'point_event_cam_left': t_cam1_2_point.tolist(),
                                                       'point_event_cam_right': t_cam2_2_point.tolist(),
                                                       'rotation': rotation.tolist(),
                                                       'timestamp': str(v['timestamp'])
                                                       }

with open('/home/eventcamera/data/transformations/transformations.json', 'w') as json_file:
    json.dump(transformations, json_file, indent=2)
print('saved transformations data')

count = 0
with open('/home/eventcamera/data/transformations/transformations.json', 'r') as file:
    projected_point_rgb_ec1_ec2 = json.load(file)
kr = {}
vr = {}
timestamp_closest_ec_right = sorted(timestamp_closest_ec_right)
timestamp_closest_ec_left = sorted(timestamp_closest_ec_left)
rgb_timestamp = sorted(rgb_timestamp)


for (kr, vr), (k, v) in zip(rotations_with_timestamps.items(), translations_with_timestamps.items()):
    print(kr)
    # for future frames just add to count
    rgb_t = rgb_timestamp[count]
    ec_left = timestamp_closest_ec_left[count]
    ec_right = timestamp_closest_ec_right[count]
    rgb_img_path = selected_path['rgb_image_path'] + str(rgb_t) + ".png"
    event_cam_left = selected_path['json_path_event_cam_left'] + str(ec_left) + ".png"
    event_cam_right = selected_path['json_path_event_cam_right'] + str(ec_right) + ".png"
    H_v_2_point = np.array([
        [1, 0, 0, v[0]],
        [0, 1, 0, v[1]],
        [0, 0, 1, v[2]],
        [0, 0, 0, 1]])

    t_cam_optical_2_point = np.array(projected_point_rgb_ec1_ec2[str(k)]['t_cam_optical_2_point'])
    H_cam_optical_2_point = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_cam_optical_2_point'])
    rotation = H_cam_optical_2_point[:3, :3]

    print(t_cam_optical_2_point)
    points_2d = cv2.projectPoints(t_cam_optical_2_point, np.eye(3), np.zeros(3), camera_matrix, distortion_coefficients)
    points_2d = np.round(points_2d[0]).astype(int)
    img_test = cv2.imread(rgb_img_path)
    img_test = cv2.circle(img_test, tuple(points_2d[0][0]), 20, (255, 0, 0), -1)

    obj_geometry = trimesh.load_mesh(selected_path["obj_path"])

    if not isinstance(obj_geometry, trimesh.Trimesh):
        print("The object is not a Trimesh object. It is a", type(obj_geometry))

    trimesh_object = obj_geometry.convex_hull
    points_3d = np.array(trimesh_object.sample(3000)) / 1000
    vertices = np.array(trimesh_object.vertices) / 1000

    if selected_path["obj_path"] == '/home/eventcamera/data/KLT/obj_000010.ply':
        translation_vector = np.array([-0.04, 0.05, 0.1])
        vertices -= translation_vector
        points_3d -= translation_vector

    if selected_path["obj_path"] == '/home/eventcamera/data/KLT/obj_000004.ply':
        translation_vector = np.array([0, 0, 0])
        vertices -= translation_vector
        points_3d -= translation_vector
        rotation_matrix = R.from_euler('z', 90, degrees=True).as_matrix()
        vertices = np.dot(vertices, rotation_matrix)
        points_3d = np.dot(points_3d, rotation_matrix)

    if selected_path["obj_path"] == '/home/eventcamera/data/KLT/obj_000009.ply':
        rotation_matrix = R.from_euler('z', 90, degrees=True).as_matrix()
        vertices = np.dot(vertices, rotation_matrix)
        points_3d = np.dot(points_3d, rotation_matrix)
        translation_vector = np.array([0.3, 0.3, 0])
        vertices -= translation_vector
        points_3d -= translation_vector

    if selected_path["obj_path"] == '/home/eventcamera/data/KLT/obj_000003.ply':
        #translation_vector = np.array([0.1, -0.1, 0])
        translation_vector = np.array([0.05, -0.05, 0])
        vertices -= translation_vector
        points_3d -= translation_vector

    if selected_path["obj_path"] == '/home/eventcamera/data/KLT/zivid.ply':
        translation_vector = np.array([0, 0, 0])
        #translation_vector = np.array([0, 0.05, 0])
        vertices -= translation_vector
        points_3d -= translation_vector

    klt_3d_transform_points = np.matmul(H_cam_optical_2_point, np.vstack((points_3d.T, np.ones(points_3d.shape[0]))))[:3, :].T
    klt_3d_transform_vertices = np.matmul(H_cam_optical_2_point, np.vstack((vertices.T, np.ones(vertices.shape[0]))))[:3, :].T
    center_3d = np.mean(klt_3d_transform_points, axis=0)
    # compute xmin, xmyx, ymin, ymax, zmin, zmax
    xmin = np.min(klt_3d_transform_points[:, 0])
    xmax = np.max(klt_3d_transform_points[:, 0])
    ymin = np.min(klt_3d_transform_points[:, 1])
    ymax = np.max(klt_3d_transform_points[:, 1])
    zmin = np.min(klt_3d_transform_points[:, 2])
    zmax = np.max(klt_3d_transform_points[:, 2])
    Bbox = np.array([xmin, xmax, ymin, ymax, zmin, zmax])

    # Prepare the data
    rotmat = R.from_matrix(rotation)
    euler_angles = rotmat.as_euler('xyz', degrees=True)
    pose = np.concatenate((center_3d, euler_angles))

    # Convert numpy arrays to lists for JSON serialization
    Bbox_list = Bbox.tolist()
    pose_list = pose.tolist()

    # Combine the data into a dictionary
    data = {
        "Timestamp": k,
        "Bbox": Bbox_list,
        "Pose": pose_list,
        "Object": selected_path["obj_name"]
    }

    # Write the data to a JSON file
    with open(os.path.join(output_dir, "data.json"), 'a') as file:
        json.dump(data, file)
        file.write('\n')

    klt_2d_points, _ = cv2.projectPoints(klt_3d_transform_points, np.eye(3), np.zeros(3), camera_matrix,
                                         distortion_coefficients)
    klt_2d_vertices, _ = cv2.projectPoints(klt_3d_transform_vertices, np.eye(3), np.zeros(3), camera_matrix,
                                         distortion_coefficients)
    center_2d, _ = cv2.projectPoints(np.array([center_3d]), np.eye(3), np.zeros(3), camera_matrix,
                                     distortion_coefficients)
    center_2d = center_2d[0, 0]

    for point in klt_2d_points:
        img_test = cv2.circle(img_test, tuple(point[0].astype(int)), 2, (200, 200, 200), -1)
    for point in klt_2d_vertices:
        img_test = cv2.circle(img_test, tuple(point[0].astype(int)), 8, (0, 0, 255), -1)
    img_test = cv2.circle(img_test, tuple(center_2d.astype(int)), 8, (0, 0, 255), -1)

    t_cam1_2_point = np.array(projected_point_rgb_ec1_ec2[str(k)]['point_event_cam_left'])
    H_cam1_2_point = np.eye(4)
    H_cam1_2_point[:3, :3] = rotation
    H_cam1_2_point[:3, 3] = t_cam1_2_point
    points_2d_cam1 = cv2.projectPoints(np.array([t_cam1_2_point]), np.eye(3), np.zeros(3), camera_mtx_cam1,
                                       distortion_coeffs_cam1)
    points_2d_cam1 = np.round(points_2d_cam1[0]).astype(int)
    print(points_2d_cam1)
    # Display the 2d points on the image
    img_test_cam1 = cv2.imread(event_cam_left)
    img_test1 = cv2.circle(img_test_cam1, tuple(points_2d_cam1[0][0]), 5, (255, 0, 0), -1)

    klt_3d_transform_points = np.matmul(H_cam1_2_point, np.vstack((points_3d.T, np.ones(points_3d.shape[0]))))[
                              :3, :].T
    klt_3d_transform_vertices = np.matmul(H_cam1_2_point, np.vstack((vertices.T, np.ones(vertices.shape[0]))))[
                                :3, :].T

    klt_2d_points, _ = cv2.projectPoints(klt_3d_transform_points, np.eye(3), np.zeros(3), camera_mtx_cam1,
                                         distortion_coeffs_cam1)
    klt_2d_vertices, _ = cv2.projectPoints(klt_3d_transform_vertices, np.eye(3), np.zeros(3), camera_mtx_cam1,
                                           distortion_coeffs_cam1)

    for point in klt_2d_points:
        img_test1 = cv2.circle(img_test1, tuple(point[0].astype(int)), 1, (255, 255, 255), -1)
    for point in klt_2d_vertices:
        img_test1 = cv2.circle(img_test1, tuple(point[0].astype(int)), 3, (0, 0, 255), -1)

    t_cam2_2_point = np.array(projected_point_rgb_ec1_ec2[str(k)]['point_event_cam_right'])
    H_cam2_2_point = np.eye(4)
    H_cam2_2_point[:3, :3] = rotation
    H_cam2_2_point[:3, 3] = t_cam2_2_point

    points_2d_cam2 = cv2.projectPoints(np.array([t_cam2_2_point]), np.eye(3), np.zeros(3), camera_mtx_cam2,
                                       distortion_coeffs_cam2)
    points_2d_cam2 = np.round(points_2d_cam2[0]).astype(int)
    print(points_2d_cam2)
    img_test_cam2 = cv2.imread(event_cam_right)
    # Display the 2d points on the image
    img_test2 = cv2.circle(img_test_cam2, tuple(points_2d_cam2[0][0]), 5, (255, 0, 0), -1)

    klt_3d_transform_points = np.matmul(H_cam2_2_point, np.vstack((points_3d.T, np.ones(points_3d.shape[0]))))[
                              :3, :].T
    klt_3d_transform_vertices = np.matmul(H_cam2_2_point, np.vstack((vertices.T, np.ones(vertices.shape[0]))))[
                                :3, :].T

    klt_2d_points, _ = cv2.projectPoints(klt_3d_transform_points, np.eye(3), np.zeros(3), camera_mtx_cam2,
                                         distortion_coeffs_cam2)
    klt_2d_vertices, _ = cv2.projectPoints(klt_3d_transform_vertices, np.eye(3), np.zeros(3), camera_mtx_cam2,
                                           distortion_coeffs_cam2)

    for point in klt_2d_points:
        img_test2 = cv2.circle(img_test2, tuple(point[0].astype(int)), 1, (255, 255, 255), -1)
    for point in klt_2d_vertices:
        img_test2 = cv2.circle(img_test2, tuple(point[0].astype(int)), 3, (0, 0, 255), -1)

    img_test = cv2.resize(img_test, (568, 426))
    img_test1 = cv2.resize(img_test1, (568, 426))
    img_test2 = cv2.resize(img_test2, (568, 426))

    concatenated_images = np.hstack((img_test, img_test1, img_test2))
    output_path = os.path.join(output_dir, f'image_{count:03d}.jpg')
    cv2.imwrite(output_path, concatenated_images)

    if k == 1712920784856162863:
        cv2.waitKey(0)
    cv2.waitKey(0)
    count += 1
    cv2.destroyAllWindows()
