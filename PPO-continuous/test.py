from scipy.spatial.transform import Rotation

quaternion = Rotation.from_euler('x', [60],degrees=True).as_quat()

print(quaternion)