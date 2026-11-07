import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
import replay_benchmark as bench


class ReplayTests(unittest.TestCase):
    def test_route_image_reaches_secondary_ocr_without_original_reload(self):
        image=Image.new('RGB',(100,100),(170,170,170))
        self.assertTrue(callable(getattr(bench,'group_route',None)))
        with patch('worker.group_rows',return_value=[]) as group:
            bench.group_route({'version':1,'pageId':'x','imageHash':'a'*64,'width':100,'height':100},image,[],[])
        self.assertIs(group.call_args.args[0],image)

    def test_existing_artifact_folder_is_never_overwritten_or_reused(self):
        self.assertTrue(callable(getattr(bench,'prepare_case_folder',None)))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'p1';folder.mkdir()
            (folder/'upload.jpg').write_bytes(b'retained-original')
            (folder/'local.rows.json').write_text('retained-model-result')
            with self.assertRaisesRegex(ValueError,'OUTPUT_ALREADY_EXISTS'):
                bench.prepare_case_folder(folder)
            self.assertEqual((folder/'upload.jpg').read_bytes(),b'retained-original')
            self.assertEqual((folder/'local.rows.json').read_text(),'retained-model-result')

if __name__=='__main__':unittest.main()
