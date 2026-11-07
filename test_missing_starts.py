import unittest
from PIL import Image
from test_layout import line
from test_layout_regions import figure
from worker import group_rows

class MissingStartTests(unittest.TestCase):
    def test_independent_english_regions_keep_unknown_number_without_inventing_it(self):
        rows=[line('完成句子，一空一词。（本题共8小题）',40,20)]
        boxes=[]
        for i in range(8):
            rows.append(line(('' if i==0 else str(i+1)+'. ')+'You have finished the work, have you?',60,100+i*50,700,32))
            boxes.append(figure(50,95+i*50,800,142+i*50,label='text'))
        # Force the dense fallback to abstain by omitting the first number.
        from missing_starts import english_region_candidates
        blocks=english_region_candidates(Image.new('RGB',(1000,600)),rows,boxes)
        self.assertEqual(len(blocks),8)
        self.assertIsNone(blocks[0]['printedNumber'])
        self.assertEqual([b['printedNumber'] for b in blocks[1:]],list(map(str,range(2,9))))
        self.assertTrue(all(b['unreadableParts'] for b in blocks))
        self.assertIsNone(english_region_candidates(Image.new('RGB',(1000,600)),rows[1:],boxes))
        self.assertIsNone(english_region_candidates(Image.new('RGB',(1000,600)),rows,boxes[:-1]))

    def test_choice_regions_inside_a_number_gap_split_without_guessing_numbers(self):
        from missing_starts import recover_choice_gaps
        from layout import split_questions
        rows=[line('1.下列说法正确的是',40,50),line('4.如图求面积',40,500)]
        boxes=[]
        for y in [200,350]:
            rows += [line('已知函数求正确选项',40,y),line('A.甲 B.乙 C.丙 D.丁',40,y+30)]
            boxes.append(figure(35,y-5,700,y+55,label='text'))
        original=split_questions(rows,1000,700)
        out=recover_choice_gaps(original,rows,boxes,1000,700)
        self.assertEqual([b['printedNumber'] for b in out],['1',None,None,'4'])
        self.assertTrue(all(b['unreadableParts'] for b in out[1:3]))
        self.assertEqual(out[-1]['bbox'],original[-1]['bbox'])
        # Numbered subparts / arbitrary paragraphs cannot create new questions.
        altered=[dict(r,text='（1）以下是一个说明段落') if r['text'].startswith('已知') else r for r in rows]
        self.assertEqual(recover_choice_gaps(original,altered,boxes,1000,700),original)
