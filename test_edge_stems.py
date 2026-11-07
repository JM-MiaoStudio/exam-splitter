import unittest
from edge_stems import split_cropped_questions


def row(text,x,y,w=700,h=30):
    return {'text':text,'score':.98,'quad':[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]}


def region(label,x,y,right,bottom,score=.9):
    return {'label':label,'score':score,'coordinate':[x,y,right,bottom]}


class EdgeStemTests(unittest.TestCase):
    def test_edge_anchor_survives_dedup_and_inline_math_need_not_have_formula_box(self):
        rows,boxes=self.fixture()
        rows[1]['score']=.85
        rows.append(row('已知sin(a+b)=',3,230,698,30))
        rows.append(row('sin(a-b)=1/3',300,260,200,30))
        boxes=[b for b in boxes if b['label']!='formula']
        result=split_cropped_questions(rows,boxes,1000,1300)
        self.assertIsNotNone(result)
        self.assertEqual(len(result),4)
        self.assertTrue(all(b['printedNumber'] is None for b in result))
    def fixture(self):
        rows=[row('已知向量a与向量b求投影',8,100),row('4.已知sin(a+b)=',1,230),
              row('P的起始位置在最低点求时间',20,370),
              row('锐二面角中直线与平面的夹角',2,800),row('当最大值为给定值时求体积',2,1020),
              row('解答题（本题共六小题）',1,1180)]
        boxes=[region('text',0,95,850,140),region('text',0,220,800,280),
               region('text',0,330,960,410),region('image',400,430,700,790),
               region('text',0,795,1000,960),region('text',0,1000,850,1070),
               region('formula',300,1020,390,1110),region('text',0,1170,700,1220)]
        return rows,boxes

    def test_edge_candidates_split_independent_stems_without_guessing_cut_numbers(self):
        rows,boxes=self.fixture()
        result=split_cropped_questions(rows,boxes,1000,1300)
        self.assertIsNotNone(result)
        self.assertEqual(len(result),4)
        self.assertTrue(all(b['printedNumber'] is None and b['unreadableParts'] for b in result))
        self.assertEqual([b['ordinal'] for b in result],[1,2,3,4])
        self.assertTrue(all(b.get('regions')==[b['bbox']] for b in result))
        q3=result[2]['bbox'];q4=result[3]['bbox']
        self.assertLessEqual(q3['y'],330)
        self.assertGreaterEqual(q3['y']+q3['height'],790)
        self.assertGreaterEqual(q4['y']+q4['height'],1110)
        self.assertLess(q4['y']+q4['height'],1170)

    def test_normal_margin_and_unanchored_prose_do_not_trigger_fallback(self):
        rows,boxes=self.fixture()
        for r in rows:
            r['quad']=[[x+80,y] for x,y in r['quad']]
        self.assertIsNone(split_cropped_questions(rows,boxes,1000,1300))
        rows,boxes=self.fixture();rows[1]['text']='设三角函数已知条件如下'
        self.assertIsNone(split_cropped_questions(rows,boxes,1000,1300))

    def test_nested_subquestions_are_not_promoted_and_overlapping_detector_duplicates_merge(self):
        rows,boxes=self.fixture()
        rows.append(row('（1）计算角的大小',2,900))
        boxes.extend([region('text',0,890,850,940),region('text',1,800,999,1110,.65)])
        result=split_cropped_questions(rows,boxes,1000,1300)
        self.assertEqual(len(result),4)

    def test_fraction_numerator_does_not_replace_left_stem_or_continuation(self):
        rows,boxes=self.fixture()
        rows.extend([row('4',250,220,20,25),row('时，则三棱锥的体积为',250,1000,450,30)])
        result=split_cropped_questions(rows,boxes,1000,1300)
        self.assertIsNotNone(result)
        self.assertEqual(len(result),4)

    def test_skewed_diagram_boundary_keeps_diagram_and_warns_about_overlap(self):
        rows,boxes=self.fixture();boxes[3]['coordinate'][3]=820
        result=split_cropped_questions(rows,boxes,1000,1300)
        self.assertEqual(len(result),4)
        q3=result[2]['bbox'];q4=result[3]['bbox']
        self.assertGreaterEqual(q3['y']+q3['height'],820)
        self.assertLessEqual(q4['y'],795)
        self.assertTrue(any('重叠' in w for w in result[2]['unreadableParts']))

    def test_low_layout_score_formula_is_still_preserved_by_ocr_evidence(self):
        rows,boxes=self.fixture();rows.append(row('tanβ',600,280,80,58))
        result=split_cropped_questions(rows,boxes,1000,1300)
        box=result[1]['bbox']
        self.assertGreaterEqual(box['y']+box['height'],338)

    def test_graphic_above_owning_stem_is_kept_whole_with_overlap_warning(self):
        rows,boxes=self.fixture();boxes[3]['coordinate'][1]=300
        result=split_cropped_questions(rows,boxes,1000,1300)
        box=result[2]['bbox']
        self.assertLessEqual(box['y'],300)
        self.assertGreaterEqual(box['y']+box['height'],790)
        self.assertTrue(any('重叠' in w for w in result[2]['unreadableParts']))


if __name__=='__main__': unittest.main()
