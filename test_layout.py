import unittest

from layout import LayoutError, split_questions, question_number


def line(text, x, y, width=400, height=20, score=.98):
    return {'text': text, 'score': score, 'quad': [[x,y], [x+width,y], [x+width,y+height], [x,y+height]]}


class LayoutTests(unittest.TestCase):
    def test_fraction_numerator_survives_missing_equals_after_function_variable(self):
        rows=[line('4.求三角形面积',35,450),
              line('5.若三点都在反比例函数y',35,615,331,18),
              line('m²+1',373,607,39,18),line('图象上，则大小关系为',412,617,180,15)]
        blocks=split_questions(rows,661,935)
        self.assertLessEqual(blocks[1]['bbox']['y'],607)
        self.assertLessEqual(blocks[0]['bbox']['y']+blocks[0]['bbox']['height'],607)

    def test_chinese_exam_number_forms_not_subquestions_or_decimals(self):
        for text, number in [('12．求函数的值',12),('２、下列说法正确的是',2),('例 8. 如图实验',8),('第15题 求面积',15),('3: 求周长',3)]:
            self.assertEqual(question_number(text), number)
        for text in ['（1）求面积','(2)说明理由','①加入稀硫酸','一、选择题','1.74','2.5g金属','A. ①②','第5页（共7页）','2026.10.28','1. x=2']:
            self.assertIsNone(question_number(text), text)

    def test_numeric_and_short_chinese_stems_are_real_questions(self):
        for text,number in [('2.2025年开展实验',2),('6.2024年，“卫生间冬季温水洗手改',6),('2、2mol氢气与氧气反应',2),('4.求值',4),('1. x=2时求平方',1),('1. 2+3=（ ）',1)]:
            self.assertEqual(question_number(text),number,text)
        rows=[line('1.求质量',40,100),line('2.2025年开展实验',40,500),line('3.求电流',40,900)]
        self.assertEqual([b['printedNumber'] for b in split_questions(rows,1000,1400)],['1','2','3'])

    def test_single_column_keeps_subquestions_and_answer_space(self):
        rows=[line('12.求合金成分',100,200),line('（1）计算质量',100,500),line('（2）说明理由',100,700),line('13.求浓度',100,1000)]
        blocks=split_questions(rows,1000,1600)
        self.assertEqual([b['printedNumber'] for b in blocks],['12','13'])
        a,b=[item['bbox'] for item in blocks]
        self.assertLess(a['y'],200)
        self.assertGreater(a['y']+a['height'],900) # preserves unrecognized diagram/handwriting
        self.assertLessEqual(a['y']+a['height'],b['y'])
        self.assertEqual(a['width'],1000)
        self.assertEqual(b['y']+b['height'],1600)

    def test_two_columns_read_down_then_across(self):
        rows=[line('1.下列说法正确的是',40,100,400),line('3.求三角形面积',560,100,400),line('2.如图求电流',40,500,400),line('4.求溶液质量',560,550,400)]
        blocks=split_questions(rows,1040,1400)
        self.assertEqual([b['printedNumber'] for b in blocks],['1','2','3','4'])
        self.assertLessEqual(blocks[0]['bbox']['width'],blocks[2]['bbox']['x'])
        self.assertLessEqual(blocks[2]['bbox']['x'],560)
        self.assertGreaterEqual(blocks[0]['bbox']['width'],450)

    def test_separate_number_token_is_joined_to_same_line(self):
        blocks=split_questions([line('8.',40,100,22),line('如图所示，求电阻',75,100,500)],1000,1200)
        self.assertEqual(blocks[0]['printedNumber'],'8')

    def test_continuation_fragment_never_becomes_question(self):
        rows=[line('（4）防止生锈的方法很多',50,20),line('考点4：金属离子推断',50,60),line('例8.如图所示，求质量',50,160)]
        blocks=split_questions(rows,1000,1200)
        self.assertEqual(len(blocks),1)
        self.assertGreater(blocks[0]['bbox']['y'],100)

    def test_tall_slanted_anchor_does_not_pull_previous_figure_into_crop(self):
        rows=[line('上一题图中坐标轴',300,145,300,38),line('12.求合金质量',100,185,700,56)]
        box=split_questions(rows,1080,1919)[0]['bbox']
        self.assertGreaterEqual(box['y'],183)
        self.assertLessEqual(box['y'],185)

    def test_no_anchor_requires_manual_selection(self):
        with self.assertRaisesRegex(LayoutError,'NO_QUESTION_ANCHORS'):
            split_questions([line('（1）求阻值',50,100)],1000,1200)

    def test_low_confidence_number_is_not_used(self):
        with self.assertRaises(LayoutError):
            split_questions([line('12.求质量',50,100,score=.35)],1000,1200)

    def test_duplicate_ocr_does_not_split_twice(self):
        rows=[line('1.如图求电流',40,100),line('1.如图求电流',41,101),line('2.如图求电压',40,500)]
        self.assertEqual(len(split_questions(rows,1000,1200)),2)

    def test_partial_strip_and_full_line_form_one_anchor(self):
        rows=[line('1.如图求电流和电压',40,100,800),line('1.如图求电流',40,101,280),line('2.如图求电压',40,500)]
        self.assertEqual(len(split_questions(rows,1000,1200)),2)

    def test_wide_question_crossing_columns_is_rejected(self):
        rows=[line('1.下列说法正确的是',40,100,900),line('2.求电流',40,500,400),line('3.求面积',560,550,400)]
        with self.assertRaisesRegex(LayoutError,'AMBIGUOUS_COLUMNS'):
            split_questions(rows,1040,1400)

    def test_number_inside_stem_is_not_new_question(self):
        rows=[line('1.如图所示求物质质量',40,100),line('2.实验结果如下表所示',300,200),line('2.求溶液质量',40,500)]
        self.assertEqual([b['printedNumber'] for b in split_questions(rows,1000,1200)],['1','2'])

    def test_indented_sequential_question_is_not_silently_dropped(self):
        rows=[line('1.求电流',40,100),line('2.求面积',140,500),line('3.求质量',40,900)]
        self.assertEqual([b['printedNumber'] for b in split_questions(rows,1040,1400)],['1','2','3'])

    def test_unsupported_unique_indented_anchor_abstains(self):
        rows=[line('1.求电流',40,100),line('7.求面积',140,500),line('3.求质量',40,900)]
        with self.assertRaises(LayoutError):split_questions(rows,1040,1400)

    def test_repeat_number_on_separate_sections_keeps_distinct_ids(self):
        rows=[line('1.求电流',40,100),line('二、填空题',40,500),line('1.求电压',40,600)]
        blocks=split_questions(rows,1000,1200)
        self.assertEqual([b['id'] for b in blocks],['b1','b2'])
        self.assertLessEqual(blocks[0]['bbox']['y']+blocks[0]['bbox']['height'],500)
        self.assertGreater(blocks[1]['bbox']['y'],520)

    def test_only_boxes_are_claimed_not_formula_or_student_transcription(self):
        block=split_questions([line('1.求电流',40,100),line('学生算式2.4A',40,400)],1000,1200)[0]
        self.assertEqual(block['questionText'],'')
        self.assertEqual(block['studentAnswer'],'')
        self.assertEqual(block['options'],[])

    def test_figure_captions_are_not_question_starts(self):
        for text in ['第7题图', '第 10 题 图', '(第6题图)', '（第5题图）', '第2题图（示意图）',
                     '第7题图。', '第7题图 第8题图']:
            self.assertIsNone(question_number(text), text)
        self.assertEqual(question_number('第7题 图中电流是多少'),7)

    def test_ruled_border_noise_in_captions_does_not_hide_ownership(self):
        from layout import figure_numbers
        for text,expected in [('第-2-题图-',2),('-第-5-题图-',5),('第—6—题图',6)]:
            self.assertEqual(figure_numbers(text),[expected])
            self.assertIsNone(question_number(text))
        for text in ['第5-6题图','第5题图中求电阻','第5题图 计算过程如下']:
            self.assertEqual(figure_numbers(text),[])

    def test_detached_figure_captions_do_not_create_false_columns(self):
        rows=[line('1.求电流',40,100),line('第1题图',100,400,90),
              line('第3题图',520,400,90),line('第4题图',800,400,90),
              line('2.求质量',40,500),line('3.如图求电压',40,700),line('4.如图求电阻',40,900)]
        blocks=split_questions(rows,1000,1200)
        self.assertEqual([b['printedNumber'] for b in blocks],['1','2','3','4'])
        self.assertTrue(blocks[0]['unreadableParts']) # foreign diagrams inside this crop
        self.assertFalse(blocks[1]['unreadableParts'])
        self.assertTrue(blocks[2]['unreadableParts']) # its diagram is elsewhere
        self.assertTrue(blocks[3]['unreadableParts'])

    def test_own_figure_inside_crop_needs_no_detached_warning(self):
        blocks=split_questions([line('1.求面积',40,100),line('第1题图',100,400,90),line('2.求质量',40,500)],1000,1200)
        self.assertEqual(blocks[0]['unreadableParts'],[])

    def test_orphan_diagram_is_not_fabricated_as_a_question(self):
        blocks=split_questions([line('6.如图求面积',40,100),line('第6题图',100,500,90),
                                line('第7题图',500,500,90),line('第8题图',800,500,90)],1000,1200)
        self.assertEqual([b['printedNumber'] for b in blocks],['6'])
        self.assertTrue(blocks[0]['unreadableParts'])

    def test_fraction_numerator_above_stem_moves_shared_boundary(self):
        rows=[line('4.求面积',85,999,1100,27),line('5.若三点都在反比例函数y=',85,1349,732,32),
              line('m2+1',823,1330,74,31),line('图象上求关系',903,1351,392,27),
              line('6.如图求值',86,1492,1273,29)]
        blocks=split_questions(rows,1448,2048)
        first,second=[b['bbox'] for b in blocks[:2]]
        self.assertLessEqual(second['y'],1330)
        self.assertLessEqual(first['y']+first['height'],second['y'])

    def test_unrelated_high_figure_label_does_not_raise_question_top(self):
        rows=[line('1.求电流',40,100,400),line('A',460,490,20,20),line('2.求质量',40,500,400,30)]
        self.assertGreaterEqual(split_questions(rows,1000,1200)[1]['bbox']['y'],498)

    def test_repeated_question_numbers_do_not_claim_other_sections_diagrams(self):
        rows=[line('1.求电流',40,100),line('二、填空题',40,500),
              line('1.如图求电压',40,600),line('第1题图',100,800,90)]
        blocks=split_questions(rows,1000,1200)
        self.assertEqual([b['unreadableParts'] for b in blocks],[[],[]])


if __name__=='__main__':
    unittest.main()
