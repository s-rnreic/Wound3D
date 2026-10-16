# Glomap doesnt provide the function of passing parameters via ini file
# Solution: create a txt or ini file where the parameter are stored and then passed to the glomap call


#format of the command line interface of glomap
#--constraint_type arg (=ONLY_POINTS)  {ONLY_POINTS, ONLY_CAMERAS, 
#                                         POINTS_AND_CAMERAS_BALANCED, 
#                                         POINTS_AND_CAMERAS}
#   --output_format arg (=bin)            {bin, txt}
#   --ba_iteration_num arg (=3)
#   --retriangulation_iteration_num arg (=1)
#   --skip_preprocessing arg (=0)
#   --skip_view_graph_calibration arg (=0)
#   --skip_relative_pose_estimation arg (=0)
#   --skip_rotation_averaging arg (=0)
#   --skip_global_positioning arg (=0)
#   --skip_bundle_adjustment arg (=0)
#   --skip_retriangulation arg (=0)
#   --skip_pruning arg (=1)
#   --ViewGraphCalib.thres_lower_ratio arg (=0.10000000000000001)
#   --ViewGraphCalib.thres_higher_ratio arg (=10)
#   --ViewGraphCalib.thres_two_view_error arg (=2)
#   --RelPoseEstimation.max_epipolar_error arg (=1)
#   --TrackEstablishment.min_num_tracks_per_view arg (=-1)
#   --TrackEstablishment.min_num_view_per_track arg (=3)
#   --TrackEstablishment.max_num_view_per_track arg (=100)
#   --TrackEstablishment.max_num_tracks arg (=10000000)
#   --GlobalPositioning.optimize_positions arg (=1)
#   --GlobalPositioning.optimize_points arg (=1)
#   --GlobalPositioning.optimize_scales arg (=1)
#   --GlobalPositioning.thres_loss_function arg (=0.10000000000000001)
#   --GlobalPositioning.max_num_iterations arg (=100)
#   --BundleAdjustment.optimize_rotations arg (=1)
#   --BundleAdjustment.optimize_translation arg (=1)
#   --BundleAdjustment.optimize_intrinsics arg (=1)
#   --BundleAdjustment.optimize_points arg (=1)
#   --BundleAdjustment.thres_loss_function arg (=1)
#   --BundleAdjustment.max_num_iterations arg (=200)
#   --Triangulation.complete_max_reproj_error arg (=15)
#   --Triangulation.merge_max_reproj_error arg (=15)
#   --Triangulation.min_angle arg (=1)
#   --Triangulation.min_num_matches arg (=15)
#   --Thresholds.max_epipolar_error_E arg (=1)
#   --Thresholds.max_epipolar_error_F arg (=4)
#   --Thresholds.max_epipolar_error_H arg (=4)
#   --Thresholds.min_inlier_num arg (=30)
#   --Thresholds.min_inlier_ratio arg (=0.25)
#   --Thresholds.max_rotation_error arg (=10)
#
#

import os
import configparser
from wound3d.legacy.run_command import run_command

def create_glomap_ini(file_path):
    config = configparser.ConfigParser()
    config.optionxform = str

    config['GLOMAP'] = {
        'constraint_type': 'ONLY_POINTS',
        'output_format': 'bin',
        'ba_iteration_num': '5',
        'retriangulation_iteration_num': '1',
        'skip_preprocessing': '0',
        'skip_view_graph_calibration': '0',
        'skip_relative_pose_estimation': '0',
        'skip_rotation_averaging': '0',
        'skip_global_positioning': '0',
        'skip_bundle_adjustment': '0',
        'skip_retriangulation': '0',
        'skip_pruning': '0',
        'ViewGraphCalib.thres_lower_ratio': '0.10000000000000001',
        'ViewGraphCalib.thres_higher_ratio': '10',
        'ViewGraphCalib.thres_two_view_error': '2',
        'RelPoseEstimation.max_epipolar_error': '1',
        'TrackEstablishment.min_num_tracks_per_view': '-1',
        'TrackEstablishment.min_num_view_per_track': '3',
        'TrackEstablishment.max_num_view_per_track': '1000',
        'TrackEstablishment.max_num_tracks': '50000',
        'GlobalPositioning.optimize_positions': '1',
        'GlobalPositioning.optimize_points': '1',
        'GlobalPositioning.optimize_scales': '1',
        'GlobalPositioning.thres_loss_function': '0.10000000000000001',
        'GlobalPositioning.max_num_iterations': '100',
        'BundleAdjustment.optimize_rotations': '1',
        'BundleAdjustment.optimize_translation': '1',
        'BundleAdjustment.optimize_intrinsics': '1',
        'BundleAdjustment.optimize_points': '1',
        'BundleAdjustment.thres_loss_function': '1',
        'BundleAdjustment.max_num_iterations': '200',
        'Triangulation.complete_max_reproj_error': '1.4',
        'Triangulation.merge_max_reproj_error': '1.4',
        'Triangulation.min_angle': '5',
        'Triangulation.min_num_matches': '15',
        'Thresholds.max_angle_error': '1',
        'Thresholds.max_reprojection_error': '0.01',
        'Thresholds.max_epipolar_error_E': '1',
        'Thresholds.max_epipolar_error_F': '4',
        'Thresholds.max_epipolar_error_H': '4',
        'Thresholds.min_inlier_num': '30',
        'Thresholds.min_inlier_ratio': '0.25',
        'Thresholds.max_rotation_error': '10'
    }

    with open(file_path, 'w') as configfile:
        config.write(configfile)
    
    return file_path

def execute_glomap(ini_file_path, database_path, output_path, image_path):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(ini_file_path)

    command = f"glomap mapper --database_path \"{database_path}\" --output_path \"{output_path}\" --image_path \"{image_path}\""
    
    for key in config['GLOMAP']:
        command += f" --{key} {config['GLOMAP'][key]}"

    run_command(command)
