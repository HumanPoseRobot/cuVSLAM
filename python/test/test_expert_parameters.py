import unittest
import cuvslam as v

class ExpertParameterBindingTest(unittest.TestCase):
    def make_tracker(self, inertial=False):
        cams=[v.Camera(size=[640,480],principal=[320,240],focal=[320,320]),
              v.Camera(size=[640,480],principal=[320,240],focal=[320,320],
                       rig_from_camera=v.Pose(rotation=[0,0,0,1],translation=[.2,0,0]))]
        mode=v.Tracker.OdometryMode.Multisensor if inertial else v.Tracker.OdometryMode.Multicamera
        rig=v.Rig(cams,[v.ImuCalibration()] if inertial else [])
        return v.Tracker(rig,v.Tracker.OdometryConfig(odometry_mode=mode,async_sba=False))
    def test_known_nonempty_string_dictionary(self):
        self.make_tracker().odom.apply_expert_parameters({'sba.num_sba_frames':'9','sba.num_sba_iterations':'11'})
    def test_multisensor_inertial_parameters(self):
        self.make_tracker(True).odom.apply_expert_parameters({
            'sba.num_inertial_sba_frames':'12','sba.num_inertial_fixed_sba_frames':'2',
            'sba.num_inertial_sba_iterations':'13','sba.inertial_robustifier_scale':'0.7',
            'sm.min_num_kf_for_gravity':'20'})
    def test_empty_dictionary(self):
        self.make_tracker().odom.apply_expert_parameters({})

if __name__=='__main__':unittest.main(verbosity=2)
