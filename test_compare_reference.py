import unittest
from compare_reference import match_boxes, overlap, reference_boxes


def box(x, y, w, h):
    return {'bbox':dict(x=x, y=y, width=w, height=h)}


class ComparisonTests(unittest.TestCase):
    def test_rescaling_uses_response_pixel_dimensions(self):
        response = {'QuestionInfo':[{'Angle':0,'Width':2000,'Height':1000,'ResultList':[
            {'Coord':[{'LeftTop':{'X':200,'Y':100}, 'RightBottom':{'X':1600,'Y':700}}]}]}]}
        self.assertEqual(reference_boxes(response,(1000,500))[0]['bbox'], dict(x=100,y=50,width=700,height=300))

    def test_rotated_reference_cannot_silently_compare_different_spaces(self):
        with self.assertRaises(ValueError):
            reference_boxes({'QuestionInfo':[{'Angle':90}]},(1000,500))

    def test_large_box_cannot_claim_multiple_reference_matches(self):
        result = match_boxes([box(0,0,100,200)], [box(0,0,100,100),box(0,100,100,100)])
        self.assertEqual(len(result['pairs']),1)
        self.assertEqual(len(result['unmatched_reference']),1)
        self.assertEqual(result['pairs'][0]['extra_area_fraction'], .5)

    def test_coverage_is_distinct_from_iou_and_includes_extra_background(self):
        m = overlap(box(0,0,200,200)['bbox'],box(50,50,100,100)['bbox'])
        self.assertEqual(m, dict(iou=.25,reference_coverage=1,extra_area_fraction=.75))

    def test_abstention_is_zero_matches_not_perfect_accuracy(self):
        self.assertEqual(match_boxes([], [box(0,0,100,100)])['unmatched_reference'], [1])
        self.assertEqual(match_boxes([], [box(0,0,100,100)])['matched_iou_50'], 0)


if __name__ == '__main__': unittest.main()
