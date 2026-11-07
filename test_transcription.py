import unittest
import transcription

def row(text,y,x=40,right=700,score=.99):
    return {'text':text,'score':score,'quad':[[x,y],[right,y],[right,y+20],[x,y+20]]}

def block():
    return {'id':'b1','ordinal':1,'printedNumber':'1','bbox':{'x':0,'y':80,'width':800,'height':500},
            'questionText':'','options':[],'studentAnswer':'','subject':'','questionType':'','unreadableParts':[],'hasFigure':True}

def clean_rows():
    return [row('1. 下列哪种做法符合文明礼仪？',100),row('A. 排队等候',160),row('B. 随意插队',200),
            row('C. 大声喧哗',240),row('D. 乱扔垃圾',280)]

class TranscriptionTests(unittest.TestCase):
    def assemble(self,rows,b=None):
        return transcription.assemble_questions([b or block()],rows,800,600)[0]

    def test_complete_plain_choices_are_observed_not_invented(self):
        b=self.assemble(clean_rows())
        self.assertEqual(b['questionText'],'1. 下列哪种做法符合文明礼仪?')
        self.assertEqual(b['options'],['A. 排队等候','B. 随意插队','C. 大声喧哗','D. 乱扔垃圾'])
        self.assertEqual(b['unreadableParts'],[])
        self.assertFalse(b['hasFigure']);self.assertEqual(b['studentAnswer'],'')

    def test_missing_d_is_not_accepted(self):
        self.assertTrue(self.assemble(clean_rows()[:-1])['unreadableParts'])

    def test_low_confidence_and_formula_require_cloud(self):
        rows=clean_rows();rows[1]=row('A. x² + 1',160,score=.99)
        self.assertTrue(self.assemble(rows)['unreadableParts'])
        rows=clean_rows();rows[1]['score']=.7
        self.assertTrue(self.assemble(rows)['unreadableParts'])

    def test_student_solution_and_inline_answer_are_not_confident_stem(self):
        rows=clean_rows()+[row('解：选B',320)]
        b=self.assemble(rows)
        self.assertTrue(b['unreadableParts']);self.assertNotIn('解：',b['options'][-1])
        rows=clean_rows();rows[0]['text']='1. 下列哪种做法正确（B）'
        self.assertTrue(self.assemble(rows)['unreadableParts'])

    def test_wrapped_options_and_strip_duplicates(self):
        rows=clean_rows();rows.insert(2,row('并保持安静',185));rows.append(dict(rows[0]))
        b=self.assemble(rows)
        self.assertEqual(b['questionText'].count('下列'),1)
        self.assertEqual(b['options'][0],'A. 排队等候\n并保持安静')

    def test_partial_strips_are_position_evidence_not_duplicate_question_text(self):
        rows=[{**r,'source':'page'} for r in clean_rows()]
        rows += [{**row('A. 排队',160,right=300),'source':'strip'}, {**row('礼仪?',100,x=600),'source':'strip'}]
        b=self.assemble(rows)
        self.assertEqual(len(b['options']),4);self.assertEqual(b['unreadableParts'],[])
        self.assertEqual(b['questionText'].count('礼仪'),1)

    def test_anchor_seen_only_in_strip_is_not_ready(self):
        rows=[{**r,'source':'page'} for r in clean_rows()[1:]]+[dict(clean_rows()[0],source='strip')]
        self.assertTrue(self.assemble(rows)['unreadableParts'])

    def test_standalone_handwritten_letter_cannot_become_option_text(self):
        b=self.assemble(clean_rows()+[row('B',330,x=300,right=325)])
        self.assertTrue(b['unreadableParts'])
        self.assertEqual(b['options'][-1],'D. 乱扔垃圾')

    def test_wrapped_option_prefix_before_next_label_is_one_option(self):
        rows=clean_rows();rows[2]['text']='并保持安静 B. 随意插队'
        b=self.assemble(rows)
        self.assertEqual(len(b['options']),4)
        self.assertEqual(b['options'][0],'A. 排队等候\n并保持安静')

    def test_shared_material_reference_requires_cloud(self):
        rows=clean_rows();rows[0]['text']='1. 根据上文，下列哪种做法符合文明礼仪？'
        self.assertTrue(self.assemble(rows)['unreadableParts'])

    def test_negative_units_superscripts_and_chemical_subscripts_require_cloud(self):
        for text in ('A. -2℃','A. x²','A. H₂O','A. 0.05','A. 10%'):
            rows=clean_rows();rows[1]['text']=text
            with self.subTest(text=text):self.assertTrue(self.assemble(rows)['unreadableParts'])

    def test_crossing_frame_and_foreign_anchor_trigger_cloud(self):
        rows=clean_rows()+[row('2. 另一道题目',350)]
        self.assertTrue(self.assemble(rows)['unreadableParts'])
        rows=clean_rows();rows[-1]['quad'][1][0]=820;rows[-1]['quad'][2][0]=820
        self.assertTrue(self.assemble(rows)['unreadableParts'])

    def test_figures_and_existing_risk_survive(self):
        rows=clean_rows();rows[0]['text']='1. 如图所示，下列哪种做法正确？'
        b=self.assemble(rows);self.assertTrue(b['hasFigure']);self.assertTrue(b['unreadableParts'])
        b=block();b['unreadableParts']=['题框待核对']
        self.assertIn('题框待核对',self.assemble(clean_rows(),b)['unreadableParts'])

    def test_inline_choices_and_no_choice_free_text_remain_reviewable(self):
        rows=[clean_rows()[0],row('A. 排队 B. 插队 C. 喧哗 D. 扔垃圾',160)]
        self.assertEqual(len(self.assemble(rows)['options']),4)
        self.assertTrue(self.assemble([row('1. 简述文明礼仪的意义。',100)])['unreadableParts'])

if __name__=='__main__':unittest.main()
