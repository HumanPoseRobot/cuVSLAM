# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import copy
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_orientation import decide, rays, rotate_calibration


class OrientationTests(unittest.TestCase):
    def records(self, winner, pairs=3):
        return [dict(sample=i, reference=3, rotation=r, inliers=50 if r == winner else 2)
                for i in range(pairs) for r in (0, 180)]

    def test_supplied_calibration_already_rotated_is_kept(self):
        # Firmware metadata may say 180 even when the currently supplied YAML includes it.
        self.assertEqual(decide(self.records(0))['rotation_deg'], 0)

    def test_missing_compensation_is_detected(self):
        self.assertEqual(decide(self.records(180))['rotation_deg'], 180)

    def test_one_frame_or_ambiguous_evidence_does_not_start_vio(self):
        self.assertIsNone(decide(self.records(180, 1))['rotation_deg'])
        records = [dict(sample=i, rotation=r, inliers=30) for i in range(3) for r in (0, 180)]
        self.assertIsNone(decide(records)['rotation_deg'])
        self.assertIsNone(decide([])['rotation_deg'])

    def test_pixel_and_extrinsic_rotation_preserve_geometry_without_mutation(self):
        camera = dict(intrinsics=[.5, .6, 600., 605., 951., 606.], resolution=[1920, 1200],
                      T_cam_imu=np.eye(4).tolist())
        cal = dict(fisheye2=camera); original = copy.deepcopy(cal)
        changed = rotate_calibration(cal, 180)
        uv = np.array([[951., 606.], [800., 500.], [1100., 700.]])
        base = rays(uv, camera)
        mirrored = np.array([1919., 1199.])-uv
        rotated = rays(mirrored, changed['fisheye2'])
        np.testing.assert_allclose(rotated, base@np.diag([-1., -1., 1.]), atol=1e-12)
        np.testing.assert_allclose(np.array(changed['fisheye2']['T_cam_imu'])[:3, :3], np.diag([-1., -1., 1.]))
        self.assertEqual(cal, original)
        self.assertEqual(rotate_calibration(changed, 180), cal)

    def test_invalid_ray_domain_is_not_fabricated(self):
        cam = dict(intrinsics=[.5, .9, 1., 1., 0., 0.])
        self.assertFalse(np.isfinite(rays(np.array([[100., 100.]]), cam)).all())


if __name__ == '__main__':
    unittest.main()
