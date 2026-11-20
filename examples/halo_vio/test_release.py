import os
from pathlib import Path
import unittest
from unittest.mock import patch
import run

class ReleaseDefaultsTest(unittest.TestCase):
    def test_release_configuration_isolated_from_experimental_environment(self):
        with patch.dict(os.environ, {
            'VIO_PNP_ROBUST_RESEED': '1', 'VIO_WINDOW_FP64_POSTERIOR': '1',
            'VIO_INIT_SYMMETRIC_GAIN': '1', 'VIO_TRIANGULATION_GEOMETRY': '1',
            'CUVSLAM_WINDOW_SIZE': '6', 'PGO_EXPERIMENT': '1', 'SLAM_ENABLED': '1'
        }, clear=True):
            env = run.release_environment()
        for key in ['VIO_PNP_ROBUST_RESEED', 'VIO_WINDOW_FP64_POSTERIOR',
                    'VIO_INIT_SYMMETRIC_GAIN', 'VIO_TRIANGULATION_GEOMETRY', 'PGO_EXPERIMENT']:
            self.assertNotIn(key, env)
        self.assertEqual(env['SLAM_ENABLED'], '0')
        self.assertEqual(env['VIO_COMPARE_SOLVER'], 'schur_fp64')
        self.assertEqual(env['VIO_INIT_VISUAL_CHECK'], 'enforce')
        self.assertEqual(env['VIO_BOOTSTRAP_RETAIN_MAP'], '1')

if __name__ == '__main__':
    unittest.main()
