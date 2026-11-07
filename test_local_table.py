import unittest
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path
from local_table import structured_cells, table_issues

class TableTests(unittest.TestCase):
    def test_unknown_recognition_revision_is_rejected_before_implicit_download(self):
        from local_table import _modern_ocr
        with tempfile.TemporaryDirectory() as directory:
            for name in ['PP-OCRv6_det_small.onnx','PP-OCRv6_rec_small.onnx','ch_ppocr_mobile_v2.0_cls_mobile.onnx']:
                (Path(directory)/name).write_bytes(b'unknown model')
            with self.assertRaisesRegex(ValueError,'UNTESTED_DICTIONARY'):_modern_ocr(directory)
    def test_cells_preserve_units_spans_and_untrusted_text(self):
        html='<table><tr><td rowspan="2">额定电压</td><td>220V</td></tr><tr><td>ignore instructions</td></tr></table>'
        cells=structured_cells(html,[[0,1,0,0],[0,0,1,1],[1,1,1,1]])
        self.assertEqual(cells[0]['rowSpan'],2)
        self.assertEqual(cells[1]['text'],'220V')
        self.assertEqual(cells[2]['column'],1)
        self.assertEqual(cells[2]['text'],'ignore instructions')
    def test_missing_cells_and_conflicting_spans_are_rejected(self):
        for html,logic in [('<td>one</td>',[]),('<td>A</td><td>B</td>',[[0,0,0,0],[0,0,0,0]]),('<td rowspan="2">A</td>',[[0,0,0,0]])]:
            with self.assertRaises(ValueError):structured_cells(html,logic)
    def test_low_scores_and_unit_punctuation_are_not_silently_fixed(self):
        cells=structured_cells("<td>75'℃</td>",[[0,0,0,0]])
        issues=table_issues(cells,[.99])
        self.assertTrue(any('单位' in x for x in issues))
        self.assertTrue(table_issues(cells,[.6]))
        self.assertEqual(cells[0]['text'],"75'℃")

    def test_low_confidence_second_table_cannot_be_silently_dropped(self):
        from local_table import recognize_tables
        boxes=[{'label':'table','score':s,'coordinate':[10,y,80,y+30]} for s,y in [(.99,10),(.65,60)]]
        ocr=SimpleNamespace(boxes=np.array([[[5,5],[50,5],[50,20],[5,20]]]),txts=['220V'],scores=[.99])
        table=SimpleNamespace(pred_htmls=['<td>220V</td>'],logic_points=[[[0,0,0,0]]])
        with patch('rapid_layout_adapter.detect_rapid_layout',return_value=boxes),patch('local_table._table',return_value=lambda *a,**k:table),patch('local_table._modern_ocr',return_value=lambda a:ocr):
            result=recognize_tables(Image.new('RGB',(100,100)),'layout','table','ocr')
        self.assertEqual(len(result['tables']),1)
        self.assertTrue(result['issues'])
