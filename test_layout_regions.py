import copy
import unittest

from layout_regions import refine_with_layout
from test_layout import line


REMOTE = '发现同题号图注位于题框外，请核对并补充配图'
FOREIGN = '题框内有其他题号的配图（2），请核对配图归属'


def block(number=1, x=0, y=0, width=1000, height=200, warnings=()):
    return {'id':f'b{number}', 'ordinal':number, 'printedNumber':str(number),
            'bbox':{'x':x,'y':y,'width':width,'height':height},
            'questionText':'','options':[],'studentAnswer':'','subject':'','questionType':'',
            'hasFigure':True,'unreadableParts':list(warnings)}


def figure(x=200, y=400, right=400, bottom=500, score=.9, label='image'):
    return {'label':label,'score':score,'coordinate':[x,y,right,bottom]}


def contains(box, x, y):
    return box['x'] <= x < box['x']+box['width'] and box['y'] <= y < box['y']+box['height']


class LayoutRegionTests(unittest.TestCase):
    def test_unique_referenced_remote_figure_is_appended_without_union(self):
        original=[block(warnings=[REMOTE,'原有公式无法辨认'])]
        snapshot=copy.deepcopy(original)
        rows=[line('1.如图求面积',20,20),line('第1题图',250,520,100)]
        result=refine_with_layout(original,rows,[figure()],1000,1000)
        self.assertEqual(original,snapshot)
        self.assertEqual(result[0]['bbox'],original[0]['bbox'])
        self.assertEqual(len(result[0]['regions']),2)
        self.assertTrue(contains(result[0]['regions'][1],300,450))
        self.assertTrue(contains(result[0]['regions'][1],300,530))
        self.assertFalse(any(contains(r,300,300) for r in result[0]['regions']))
        self.assertEqual(result[0]['unreadableParts'],['原有公式无法辨认'])
        self.assertEqual(result[0]['questionText'],'')

    def test_remote_caption_number_without_figure_reference_is_not_trusted(self):
        rows=[line('1.两个铝块质量之比为',20,20),line('第1题图',250,520,100)]
        result=refine_with_layout([block(warnings=[REMOTE])],rows,[figure()],1000,1000)
        self.assertNotIn('regions',result[0])
        self.assertIn(REMOTE,result[0]['unreadableParts'])
        self.assertTrue(any('图引用' in warning for warning in result[0]['unreadableParts']))

    def test_image_and_caption_confidence_thresholds_and_supported_labels(self):
        for score,caption_score,label in [(.64,.99,'image'),(.9,.71,'image'),(.9,.99,'text')]:
            with self.subTest(score=score,caption_score=caption_score,label=label):
                result=refine_with_layout([block(warnings=[REMOTE])],
                    [line('1.如图求面积',20,20),line('第1题图',250,520,100,score=caption_score)],
                    [figure(score=score,label=label)],1000,1000)
                self.assertNotIn('regions',result[0])
                self.assertIn(REMOTE,result[0]['unreadableParts'])
        result=refine_with_layout([block()],
            [line('1.图为函数图像',20,20),line('第1题图',250,520,100,score=.72)],
            [figure(score=.65,label='chart')],1000,1000)
        self.assertEqual(len(result[0]['regions']),2)

    def test_caption_must_be_close_and_horizontally_aligned(self):
        for caption in [line('第1题图',600,520,100),line('第1题图',250,250,100),line('第1题图',250,800,100)]:
            result=refine_with_layout([block(warnings=[REMOTE])],[line('1.如图求面积',20,20),caption],[figure()],1000,1000)
            self.assertNotIn('regions',result[0])

    def test_unique_caption_above_remote_image_keeps_whole_image_and_caption(self):
        rows=[line('1.如图求面积',20,20),line('第1题图',250,350,100)]
        result=refine_with_layout([block(warnings=[REMOTE])],rows,[figure()],1000,1000)
        regions=result[0].get('regions',[])
        self.assertEqual(len(regions),2)
        self.assertTrue(contains(regions[1],300,355))
        self.assertTrue(contains(regions[1],300,499))
        self.assertFalse(any(contains(r,300,300) for r in regions))
        self.assertNotIn(REMOTE,result[0]['unreadableParts'])

    def test_caption_between_equally_near_images_abstains_in_both_orders(self):
        rows=[line('1.如图求面积',20,20),line('第1题图',250,520,100)]
        figures=[figure(),figure(200,560,400,660)]
        for boxes in [figures,list(reversed(figures))]:
            result=refine_with_layout([block(warnings=[REMOTE])],rows,boxes,1000,1000)
            self.assertNotIn('regions',result[0])
            self.assertIn(REMOTE,result[0]['unreadableParts'])

    def test_upper_caption_cannot_cross_text_into_remote_image(self):
        for text in ['2.如图求周长','A. x=2','学生手写说明','x+2=3']:
            rows=[line('1.如图求面积',20,20),line('第1题图',250,350,100),line(text,250,380,150,12)]
            result=refine_with_layout([block(warnings=[REMOTE])],rows,[figure()],1000,1000)
            self.assertNotIn('regions',result[0])

    def test_one_image_between_two_captions_stays_unresolved(self):
        rows=[line('1.如图求面积',20,20),line('2.如图求周长',20,220),
              line('第1题图',250,350,100),line('第2题图',250,520,100)]
        result=refine_with_layout([block(warnings=[REMOTE]),block(2,y=200,warnings=[REMOTE])],rows,[figure()],1000,1000)
        self.assertTrue(all('regions' not in b for b in result))

    def test_foreign_upper_caption_subtraction_preserves_unrelated_ink(self):
        rows=[line('1.如图求面积',20,20),line('第2题图',550,350,100)]
        result=refine_with_layout([block(height=900,warnings=[FOREIGN])],rows,[figure(500,400,700,500)],1000,1000)
        regions=result[0].get('regions',[result[0]['bbox']])
        self.assertFalse(any(contains(r,600,450) for r in regions))
        self.assertFalse(any(contains(r,600,355) for r in regions))
        self.assertTrue(any(contains(r,800,450) for r in regions))

    def test_duplicate_numbers_and_competing_captions_abstain(self):
        rows=[line('1.如图求面积',20,20),line('1.如图求面积',20,220),line('第1题图',250,520,100)]
        result=refine_with_layout([block(),block(y=200)],rows,[figure()],1000,1000)
        self.assertTrue(all('regions' not in b for b in result))
        self.assertTrue(all(b['unreadableParts'] for b in result))
        rows=[line('1.如图求面积',20,20),line('第1题图',210,520,70),line('第2题图',310,520,70)]
        result=refine_with_layout([block(warnings=[REMOTE])],rows,[figure()],1000,1000)
        self.assertNotIn('regions',result[0])
        self.assertIn(REMOTE,result[0]['unreadableParts'])

    def test_foreign_figure_subtraction_preserves_all_other_source_pixels(self):
        original=block(height=900,warnings=[FOREIGN,'其他条件仍不清楚'])
        rows=[line('1.如图求面积',20,20),line('第二行公式=',20,390,150),
              line('学生手写计算',20,650,180,score=.4),line('第2题图',550,520,100)]
        result=refine_with_layout([original],rows,[figure(500,400,700,500)],1000,1000)
        self.assertEqual(len(result),1)
        regions=result[0]['regions']
        self.assertEqual(regions[0],result[0]['bbox'])
        self.assertTrue(contains(regions[0],30,30))
        self.assertFalse(any(contains(r,600,450) for r in regions))
        self.assertFalse(any(contains(r,600,530) for r in regions))
        # Undetected formulas, work and whitespace outside the figure survive,
        # not just the rectangles recognized by OCR.
        for x,y in [(30,400),(40,700),(900,450),(300,800),(550,300)]:
            self.assertTrue(any(contains(r,x,y) for r in regions),(x,y))
        # Keeping the pixels is insufficient if a text/formula line is cut
        # into two disjoint positions in the eventual composite image.
        self.assertTrue(any(contains(r,20,390) and contains(r,169,409) for r in regions))
        self.assertEqual(sum(r['width']*r['height'] for r in regions),900000-208*148)
        for i,a in enumerate(regions):
            for b in regions[i+1:]:
                self.assertFalse(max(a['x'],b['x'])<min(a['x']+a['width'],b['x']+b['width']) and
                                 max(a['y'],b['y'])<min(a['y']+a['height'],b['y']+b['height']))
        self.assertNotIn(FOREIGN,result[0]['unreadableParts'])
        self.assertIn('其他条件仍不清楚',result[0]['unreadableParts'])
        self.assertTrue(any('未找到对应题干' in w for w in result[0]['unreadableParts']))

    def test_short_axis_labels_slightly_outside_detector_are_retained(self):
        rows=[line('1.图为函数图像',20,20),line('0 5 10 L/cm',270,488,120,20),
              line('第1题图',250,520,100)]
        result=refine_with_layout([block()],rows,[figure()],1000,1000)
        self.assertEqual(len(result[0].get('regions',[])),2)
        self.assertTrue(contains(result[0]['regions'][1],389,507))

    def test_stem_or_formula_crossing_foreign_image_forces_full_fallback(self):
        original=block(height=900,warnings=[FOREIGN])
        rows=[line('1.如图求面积',20,20),line('本题的公式和选项不可切断',20,440,650),line('第2题图',550,520,100)]
        result=refine_with_layout([original],rows,[figure(500,400,700,500)],1000,1000)
        self.assertEqual(result[0]['bbox'],original['bbox'])
        self.assertNotIn('regions',result[0])
        self.assertIn(FOREIGN,result[0]['unreadableParts'])

    def test_options_or_body_formula_inside_misclassified_image_are_not_deleted(self):
        for text in ['A. x=2 B. x=3 C. x=4 D. x=5', 'B、2 mol', '(1) x + 2 = 3', 'x+2=3', '答:2+3=5']:
            with self.subTest(text=text):
                original=block(height=900,warnings=[FOREIGN])
                rows=[line('1.如图求面积',20,20),line(text,510,440,180),line('第2题图',550,520,100)]
                result=refine_with_layout([original],rows,[figure(500,400,700,500)],1000,1000)
                self.assertNotIn('regions',result[0])
                self.assertEqual(result[0]['bbox'],original['bbox'])
                self.assertIn(FOREIGN,result[0]['unreadableParts'])

    def test_conflicting_reference_of_another_real_stem_retains_host_pixels(self):
        original=[block(height=600,warnings=[FOREIGN]),block(2,y=600,height=200,warnings=[REMOTE])]
        rows=[line('1.如图摆球实验',20,20),line('第2题图',250,520,100),line('2.两个铝块质量之比为',20,620)]
        result=refine_with_layout(original,rows,[figure()],1000,1000)
        self.assertEqual(result[0]['bbox'],original[0]['bbox'])
        self.assertTrue(all('regions' not in b for b in result))
        self.assertIn(FOREIGN,result[0]['unreadableParts'])

    def test_independent_foreign_crop_is_removed_beside_retained_caption_conflict(self):
        warning='题框内有其他题号的配图（2、3），请核对配图归属'
        unobserved='题框内有其他题号的配图（2、99），请核对配图归属'
        original=[block(height=600,warnings=[warning,unobserved,'教师核对注记']),
                  block(2,y=600,height=100,warnings=[REMOTE]),
                  block(3,y=700,height=100,warnings=[REMOTE])]
        rows=[line('1.如图实验',20,20),line('第2题图',150,520,100),
              line('第3题图',550,520,100),line('2.如图求面积',20,620),
              line('3.两个物体质量之比为',20,720)]
        result=refine_with_layout(original,rows,
            [figure(100,400,300,500),figure(500,390,700,500)],1000,1000)
        regions=result[0].get('regions',[result[0]['bbox']])
        self.assertFalse(any(contains(r,200,450) for r in regions))
        self.assertTrue(any(contains(r,496,386) and contains(r,703,543) for r in regions))
        self.assertEqual(regions[0],result[0]['bbox'])
        self.assertLessEqual(len(regions),8)
        self.assertEqual(sum(r['width']*r['height'] for r in regions),600000-208*148)
        self.assertIn('题框内有其他题号的配图（3），请核对配图归属',result[0]['unreadableParts'])
        self.assertIn('题框内有其他题号的配图（99），请核对配图归属',result[0]['unreadableParts'])
        self.assertNotIn(warning,result[0]['unreadableParts'])
        self.assertIn('教师核对注记',result[0]['unreadableParts'])
        self.assertTrue(any('关联不确定' in w for w in result[0]['unreadableParts']))
        self.assertEqual(len(result[1]['regions']),2)
        self.assertNotIn('regions',result[2])
        self.assertTrue(any('图引用' in w for w in result[2]['unreadableParts']))
        self.assertIn(REMOTE,result[2]['unreadableParts'])
        self.assertEqual(refine_with_layout(list(reversed(original)),rows,
            [figure(100,400,300,500),figure(500,390,700,500)],1000,1000),list(reversed(result)))

    def test_partial_resolution_preserves_unmatched_caption_and_body_formula(self):
        original=block(height=600,warnings=['题框内有其他题号的配图（2、3），请核对配图归属'])
        rows=[line('1.如图实验',20,20),line('第2题图',150,520,100),
              line('第3题图',550,520,100),line('B. x+2=3',510,440,180)]
        result=refine_with_layout([original],rows,
            [figure(100,400,300,500),figure(500,400,700,500)],1000,1000)
        regions=result[0].get('regions',[result[0]['bbox']])
        self.assertFalse(any(contains(r,200,450) for r in regions))
        self.assertTrue(any(contains(r,510,440) and contains(r,689,459) for r in regions))
        self.assertTrue(any(contains(r,550,520) and contains(r,649,539) for r in regions))
        self.assertIn('题框内有其他题号的配图（3），请核对配图归属',result[0]['unreadableParts'])

    def test_partial_resolution_does_not_cut_overlapping_unresolved_figure(self):
        original=[block(height=600,warnings=['题框内有其他题号的配图（2、3），请核对配图归属']),
                  block(2,y=600,height=100),block(3,y=700,height=100)]
        # The two image boxes overlap; each caption uniquely pairs by its x.
        rows=[line('1.如图实验',20,20),line('第2题图',150,520,100),
              line('第3题图',550,520,100),line('2.如图求面积',20,620),
              line('3.两个物体质量之比为',20,720)]
        result=refine_with_layout(original,rows,
            [figure(100,400,450,500),figure(400,390,700,500)],1000,1000)
        self.assertEqual(result[0]['bbox'],original[0]['bbox'])
        self.assertNotIn('regions',result[0])
        self.assertTrue(any('关联不确定' in w for w in result[0]['unreadableParts']))

    def test_rejected_pair_retains_overlapping_raw_figure_pixels(self):
        original=block(height=600,warnings=['题框内有其他题号的配图（2、3），请核对配图归属'])
        rows=[line('1.如图实验',20,20),line('第2题图',150,520,100),
              line('第3题图',550,520,100),line('B. x+2=3',520,440,170)]
        result=refine_with_layout([original],rows,
            [figure(100,400,450,500),figure(400,390,700,500)],1000,1000)
        regions=result[0].get('regions',[result[0]['bbox']])
        self.assertTrue(any(contains(r,420,450) for r in regions))
        self.assertTrue(any(contains(r,396,386) and contains(r,703,543) for r in regions))
        self.assertEqual(result[0]['bbox'],original['bbox'])
        self.assertNotIn('regions',result[0])
        self.assertIn(original['unreadableParts'][0],result[0]['unreadableParts'])
        self.assertTrue(any('关联不确定' in w for w in result[0]['unreadableParts']))

    def test_region_limit_falls_back_without_dropping_crops(self):
        rows=[line('1.如图求面积',20,20)]
        figures=[]
        for i in range(8):
            y=100+i*100
            figures.append(figure(500,y,650,y+30))
            rows.append(line(f'第{i+2}题图',530,y+40,90))
        original=block(height=1000)
        result=refine_with_layout([original],rows,figures,1000,1000)
        self.assertEqual(result[0]['bbox'],original['bbox'])
        self.assertNotIn('regions',result[0])
        self.assertTrue(any('8个区域' in w for w in result[0]['unreadableParts']))

    def test_unmatched_foreign_caption_or_existing_regions_are_never_removed(self):
        original=block(height=900,warnings=[FOREIGN])
        rows=[line('1.如图求面积',20,20),line('第2题图',550,520,100)]
        result=refine_with_layout([original],rows,[],1000,1000)
        self.assertEqual(result[0]['bbox'],original['bbox'])
        self.assertNotIn('regions',result[0])
        self.assertIn(FOREIGN,result[0]['unreadableParts'])
        original['regions']=[original['bbox'],{'x':0,'y':950,'width':20,'height':20}]
        result=refine_with_layout([original],rows,[figure(500,400,700,500)],1000,1000)
        self.assertEqual(result,[original])
        self.assertIsNot(result[0],original)

    def test_malformed_or_out_of_page_layout_detections_are_ignored(self):
        rows=[line('1.如图求面积',20,20),line('第1题图',250,520,100)]
        bad=[{},None,figure(score=float('nan')),figure(x=-10),figure(right=float('inf'))]
        result=refine_with_layout([block(warnings=[REMOTE])],rows,bad,1000,1000)
        self.assertNotIn('regions',result[0])
        self.assertIn(REMOTE,result[0]['unreadableParts'])


if __name__ == '__main__':
    unittest.main()
