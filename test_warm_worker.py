import io
import json
import unittest
from unittest.mock import patch
import worker


class WarmWorkerTests(unittest.TestCase):
    def test_stream_multiple_requests_errors_do_not_contaminate_next_page(self):
        requests=[{'id':'a','input':{'pageId':'first'}}, {'id':'b','input':{}}, {'id':'c','input':{'pageId':'third'}}]
        source=io.BytesIO(b''.join(json.dumps(p).encode()+b'\n' for p in requests))
        output=io.StringIO()
        def process(page,**kwargs):
            print('library noise')
            if not page:raise ValueError('private student text')
            return page
        with patch('worker.process_page',side_effect=process),patch('sys.stderr',io.StringIO()):
            worker.serve(source,output)
        replies=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r['id'] for r in replies],['a','b','c'])
        self.assertEqual(replies[0]['result']['pageId'],'first')
        self.assertEqual(replies[1]['error'],'SPLITTER_RUNTIME_FAILED')
        self.assertEqual(replies[2]['result']['pageId'],'third')
        self.assertNotIn('private',output.getvalue())

    def test_oversized_or_invalid_envelope_stops_before_inference(self):
        for payload in [b'x'*(worker.MAX_INPUT_BYTES+2049),b'{}\n',b'not-json\n']:
            with patch('worker.process_page') as process,patch('sys.stderr',io.StringIO()):
                worker.serve(io.BytesIO(payload),io.StringIO())
            process.assert_not_called()

    def test_engine_is_constructed_once(self):
        worker.ocr_engine.cache_clear()
        with patch('rapidocr_onnxruntime.RapidOCR') as factory:
            self.assertIs(worker.ocr_engine(),worker.ocr_engine())
            factory.assert_called_once_with(intra_op_num_threads=2,inter_op_num_threads=1)
        worker.ocr_engine.cache_clear()
