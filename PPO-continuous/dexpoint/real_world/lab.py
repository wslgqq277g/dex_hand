import numpy as np
from sapien.core import Pose
#origin
# CAM2ROBOT = Pose.from_transformation_matrix(np.array(
#     [[0.60346958, 0.36270068, -0.7101216, 0.962396],
#      [0.7960018, -0.22156729, 0.56328419, -0.35524235],
#      [0.04696384, -0.90518294, -0.42241951, 0.31896536],
#      [0., 0., 0., 1.]]
# ))


# CAM2ROBOT = Pose.from_transformation_matrix(np.array(
#     [[0.10346958, 0.36270068, -0.7101216, 0.562396],
#      [-0.7960018, -0.22156729, 0.56328419, -0.65524235],
#      [0.14696384, -1.30518294, -0.42241951, 0.11896536],
#      [0., 0., 0., 1.]]
# ))
CAM2ROBOT = np.array(
    [[0.10346958, 0.36270068, -0.7101216, 0.562396],
     [-0.7960018, -0.22156729, 0.56328419, -0.65524235],
     [0.14696384, -1.30518294, -0.42241951, 0.11896536],
     [0., 0., 0., 1.]]
)

DESK2ROBOT_Z_AXIS = -0.05

# Relocate
RELOCATE_BOUND = [0.2, 0.8, -0.4, 0.4, DESK2ROBOT_Z_AXIS + 0.005, 0.6]

# TODO:
# old
# ROBOT2BASE = Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]))
# new
# ROBOT2BASE = Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]),
#                     q=np.array([0.9937122,  -0.11196447, 0, 0]),
#                     # q=np.array([1, 0, 0, 0]),
#                     )
ROBOT2BASE = Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]),
                    q=np.array([0.9937122,  -0.11196447, 0, 0]),
                    # q=np.array([1, 0, 0, 0]),
                    )


# Table size
TABLE_XY_SIZE = np.array([0.6, 1.2])
TABLE_ORIGIN = np.array([0, -0.15])

