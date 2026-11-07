import unittest
from unittest.mock import patch
from PIL import Image
from dense_english import split_dense_english


def line(text,x,y,width=650,height=26,slope=-.04,score=.97):
    return {'text':text,'score':score,'quad':[[x,y],[x+width,y+slope*width],[x+width,y+slope*width+height],[x,y+height]]}


def worksheet():
    return [line(f'{i}. Complete the English sentence',90,100+(i-1)*42) for i in range(1,9)]


class DenseEnglishTests(unittest.TestCase):
    def test_page_and_recovery_share_one_model_initialization(self):
        import worker
        from dense_english import _recognize_strip
        worker.ocr_engine.cache_clear()
        try:
            with patch('rapidocr_onnxruntime.RapidOCR') as factory:
                factory.return_value.return_value=([],None)
                image=Image.new('RGB',(100,80),'white')
                self.assertEqual(worker.recognize(image),[])
                self.assertEqual(_recognize_strip(image),[])
                factory.assert_called_once_with(intra_op_num_threads=2,inter_op_num_threads=1)
                self.assertEqual(factory.return_value.call_count,4)
        finally:
            worker.ocr_engine.cache_clear()

    def test_corridor_ocr_uses_existing_worker_engine(self):
        from dense_english import _recognize_strip
        import numpy as np
        expected=[(np.array([[0,0],[20,0],[20,10],[0,10]]),'1. English text',.98)]
        with patch('ocr_runtime.ocr_engine') as factory:
            factory.return_value.return_value=(expected,None)
            rows=_recognize_strip(Image.new('RGB',(100,80),'white'))
        factory.assert_called_once()
        self.assertEqual(rows[0]['text'],'1. English text')
        self.assertEqual(rows[0]['quad'],[[0.,0.],[20.,0.],[20.,10.],[0.,10.]])

    def test_skewed_dense_lines_keep_all_original_quad_vertices(self):
        rows=worksheet()
        blocks=split_dense_english(Image.new('RGB',(1000,600),'white'),rows,recognize_strip=lambda _: [])
        self.assertIsNotNone(blocks)
        self.assertEqual([b['printedNumber'] for b in blocks],list(map(str,range(1,9))))
        for row,block in zip(rows,blocks):
            b=block['bbox']
            for x,y in row['quad']:
                self.assertTrue(b['x']<=x<=b['x']+b['width'])
                self.assertTrue(b['y']<=y<=b['y']+b['height'])
        self.assertTrue(any(b['unreadableParts'] for b in blocks))

    def test_strip_coordinates_map_back_and_do_not_invent_missing_number(self):
        rows=worksheet()[1:]
        im=Image.new('RGB',(1000,600),'white')
        self.assertIsNone(split_dense_english(im,rows,recognize_strip=lambda _: []))
        # The strip origin is unknown to the OCR callback; use a real token at
        # local x=40, and verify its position via the computed crop boundary.
        seen=[]
        def recognize(crop):
            seen.append(crop.size)
            return [line('1. Complete the English sentence',40,100,width=150)]
        result=split_dense_english(im,rows,recognize_strip=recognize)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]['printedNumber'],'1')
        self.assertLess(result[0]['bbox']['x'],90)
        self.assertEqual(len(seen),1)

    def test_handwriting_extending_above_anchor_is_preserved(self):
        rows=worksheet()
        rows.append(line('handwritten response',760,58,width=100,height=32))
        blocks=split_dense_english(Image.new('RGB',(1000,600),'white'),rows,recognize_strip=lambda _: [])
        b=blocks[0]['bbox']
        self.assertGreaterEqual(b['x']+b['width'],860)
        self.assertLessEqual(b['y'],54)
        self.assertGreaterEqual(b['y']+b['height'],90)

    def test_gaps_duplicate_numbers_and_non_english_abstain(self):
        image=Image.new('RGB',(1000,600),'white')
        for rows in [worksheet()[:3]+worksheet()[4:],
                     worksheet()+[line('3. Other question',600,510,width=200)],
                     [line(f'{i}. 求下列函数的解析式',90,100+i*42) for i in range(1,9)]]:
            self.assertIsNone(split_dense_english(image,rows,recognize_strip=lambda _: []))

    def test_explicit_single_region_preserves_intentional_overlap_in_consumers(self):
        blocks=split_dense_english(Image.new('RGB',(1000,600),'white'),worksheet(),recognize_strip=lambda _: [])
        for block in blocks:
            self.assertEqual(block.get('regions'),[block['bbox']])

    def test_partial_strip_duplicate_does_not_pull_previous_sentence_into_next_crop(self):
        rows=[line(f'{i}. Complete the English sentence',90,100+(i-1)*42,width=400) for i in range(1,9)]
        rows[0]=line('1. Complete the English sentence',90,100,width=650,slope=0)
        rows.append(line('sentence tail',650,117,width=80,height=9,slope=0))
        blocks=split_dense_english(Image.new('RGB',(1000,600),'white'),rows,recognize_strip=lambda _: [])
        self.assertGreaterEqual(blocks[1]['bbox']['y'],120)

    def test_last_question_retains_all_recognized_continuation_lines(self):
        rows=worksheet()+[line('This is the rest of the final question',90,440,width=650),
                          line('And this is the last answer line',90,480,width=780)]
        blocks=split_dense_english(Image.new('RGB',(1000,600),'white'),rows,recognize_strip=lambda _: [])
        b=blocks[-1]['bbox']
        for row in rows[-2:]:
            for x,y in row['quad']:
                self.assertTrue(b['x']<=x<=b['x']+b['width'])
                self.assertTrue(b['y']<=y<=b['y']+b['height'])

    def test_last_question_keeps_unrecognized_handwriting_down_to_page_bottom(self):
        image=Image.new('RGB',(1000,600),'white')
        image.putpixel((500,575),(0,0,0))
        blocks=split_dense_english(image,worksheet(),recognize_strip=lambda _: [])
        b=blocks[-1]['bbox']
        self.assertEqual(b['y']+b['height'],600)
        self.assertTrue(any('页底' in warning for warning in blocks[-1]['unreadableParts']))

    def test_declared_count_requires_every_printed_anchor(self):
        rows=worksheet()+[line('本题共 9 小题',90,55,slope=0)]
        self.assertIsNone(split_dense_english(Image.new('RGB',(1000,600),'white'),rows,recognize_strip=lambda _: []))

if __name__=='__main__':
    unittest.main()

