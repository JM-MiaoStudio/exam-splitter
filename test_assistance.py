import copy
import unittest
from unittest.mock import patch
import assistance
import worker
from test_worker import request


def block(n,y=100):
    box={'x':0,'y':y,'width':1000,'height':100}
    return {'id':'b'+str(n),'ordinal':n if isinstance(n,int) else 1,'printedNumber':str(n) if n is not None else None,
            'bbox':box,'questionText':'','options':[],'studentAnswer':'','subject':'','questionType':'','hasFigure':True,'unreadableParts':[]}


class AssistanceTests(unittest.TestCase):
    def test_candidate_gate_exists(self):
        self.assertTrue(callable(getattr(assistance,'choose_assisted',None)))

    def test_existing_geometry_never_replaced_by_more_cloud_blocks(self):
        original=[block(12),block(13,500)];before=copy.deepcopy(original)
        chosen=assistance.choose_assisted(original,[block(12),block(65,300),block(13,500)],'云端')
        self.assertEqual([b['bbox'] for b in chosen],[b['bbox'] for b in original])
        self.assertEqual([b['printedNumber'] for b in chosen],['12','13'])
        self.assertEqual(original,before)
        self.assertTrue(chosen[0]['unreadableParts'])

    def test_partial_or_irregular_results_do_not_masquerade_as_whole_page(self):
        for candidates in [[block(3)],[block(1),block(4),block(5),block(16)], [block(1),block(1)]]:
            self.assertEqual(assistance.choose_assisted([],candidates,'增强'),[])

    def test_reviewable_rescue_requires_multiple_consistent_candidates(self):
        chosen=assistance.choose_assisted([],[block(8),block(9,300),block(10,600)],'云端')
        self.assertEqual(len(chosen),3)
        self.assertTrue(all(b['unreadableParts'] and b['regions'][0]==b['bbox'] for b in chosen))
        self.assertEqual(len(assistance.choose_assisted([],[block(None),block(None,300)],'增强')),2)

    def test_worker_no_extra_ocr_for_existing_local_result(self):
        rows=[{'quad':[[40,100],[600,100],[600,130],[40,130]],'text':'12.求质量','score':.99}]
        with patch('worker.recognize',return_value=rows) as recognize:
            page=worker.process_page(request(),enhance_fallback=True)
        self.assertEqual(recognize.call_count,1);self.assertEqual(len(page['blocks']),1)

    def test_worker_cloud_evidence_bound_to_image_even_if_local_good(self):
        from test_cloud_geometry import envelope
        with patch('worker.recognize',return_value=[]):
            with self.assertRaises(ValueError):worker.process_page(request(),cloud_ocr=envelope())

if __name__=='__main__':unittest.main()
