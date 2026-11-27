import unittest,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import sealing
class SealingTests(unittest.TestCase):
    def test_write_once_and_tamper(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'stage';root.mkdir();(root/'core').write_bytes(b'fixed core')
            sealing.seal(root,{'kind':'test'});sealing.verify(root)
            with self.assertRaises(FileExistsError):sealing.seal(root,{})
            (root/'core').write_bytes(b'changed core')
            with self.assertRaises(ValueError):sealing.verify(root)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory()as d:
            a=Path(d)/'stage';b=Path(d)/'published';a.mkdir();b.mkdir();(a/'core').write_bytes(b'a');(b/'core').write_bytes(b'b');sealing.seal(a,{})
            with self.assertRaises(FileExistsError):sealing.publish(a,b)
            self.assertEqual((b/'core').read_bytes(),b'b')
    def test_relocation_and_extra_file(self):
        with tempfile.TemporaryDirectory()as d:
            a=Path(d)/'stage';b=Path(d)/'candidate';a.mkdir();(a/'core').write_bytes(b'a');sealing.seal(a,{})
            sealing.publish(a,b);sealing.verify(b);(b/'extra').write_bytes(b'x')
            with self.assertRaises(ValueError):sealing.verify(b)
    def test_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'stage';root.mkdir();outside=Path(d)/'outside';outside.write_bytes(b'x');(root/'link').symlink_to(outside)
            with self.assertRaises(ValueError):sealing.seal(root,{})
if __name__=='__main__':unittest.main()
