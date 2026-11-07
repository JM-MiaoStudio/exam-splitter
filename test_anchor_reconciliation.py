import copy
import unittest
from layout import split_questions
from test_layout import line

class AnchorReconciliationTests(unittest.TestCase):
    def test_adjacent_skewed_lines_with_large_bbox_overlap_are_not_duplicates(self):
        from layout import question_starts
        rows=[{'text':f'{n}. Complete the English sentence','score':.99,
               'quad':[[40,y],[640,y-65],[640,y-45],[40,y+20]]} for n,y in [(1,100),(2,130)]]
        self.assertEqual(len(question_starts(rows,1000,400)),2)
    def test_same_ink_with_small_ocr_body_differences_merges(self):
        rows=[line('5.求质量',74,100),line('6、图为一轻质弹簧的长度',75,200,500,18),
              line('6、为轻质弹簧的长度',80,202,180,12),line('7.求速度',75,300)]
        self.assertEqual([b['printedNumber'] for b in split_questions(rows,661,500)],['5','6','7'])
    def test_lower_score_numbered_prefix_survives_text_deduplication(self):
        rows=[dict(line('4.已知sin(a+b)=',0,137,233,44,.88),source='strip'),
              line('已知sin(a+b)=',11,137,222,44,.90),line('5.求面积',0,300)]
        self.assertEqual([b['printedNumber'] for b in split_questions(rows,1000,500)],['4','5'])

    def test_conflicting_same_line_numbers_use_unique_observed_sequence_and_warn(self):
        rows=[line('3.求面积',75,100,400),line('14.爬杆时求摩擦力方向',74,200,512,18,.98),
              dict(line('4.爬杆时求摩擦力',78,203,184,12,.95),source='strip'),line('5.如图求受力',75,300,400)]
        original=copy.deepcopy(rows)
        result=split_questions(rows,661,500)
        self.assertEqual([b['printedNumber'] for b in result],['3','4','5'])
        self.assertTrue(any('题号' in s and '冲突' in s for s in result[1]['unreadableParts']))
        self.assertEqual(rows,original)

    def test_unresolved_conflict_preserves_one_box_without_guessing_number(self):
        result=split_questions([line('14.如图求面积',40,100,600),line('4.如图求面积',42,102,250)],1000,500)
        self.assertEqual(len(result),1)
        self.assertIsNone(result[0]['printedNumber'])
        self.assertTrue(result[0]['unreadableParts'])

    def test_same_number_shifted_strip_is_not_another_question(self):
        rows=[line('2.用手拿住石子的绳子',75,138,513,20),line('2.用手拿住石子',84,142,178,12),line('3.春节将至',77,295,400)]
        self.assertEqual([b['printedNumber'] for b in split_questions(rows,661,500)],['2','3'])

    def test_leading_border_mark_does_not_hide_observed_two_digit_number(self):
        rows=[line('9.求质量',75,100),line('10.求速度',75,200),line('！11、下列关于力的说法正确的是',75,300),line('1、下列关于力的说法',89,302,170,12)]
        result=split_questions(rows,661,500)
        self.assertEqual([b['printedNumber'] for b in result],['9','10','11'])
        self.assertTrue(result[-1]['unreadableParts'])

    def test_partial_caption_does_not_become_question_anchor(self):
        rows=[line('1.如图求面积',40,100),line('第5题图',100,300,90),line('第5题',101,301,65),line('2.求速度',40,500)]
        self.assertEqual([b['printedNumber'] for b in split_questions(rows,1000,700)],['1','2'])
