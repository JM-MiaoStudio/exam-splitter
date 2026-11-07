import unittest
import cloud_geometry as cg


def envelope(size=(4000,3000), sent=(2000,1500), matrix=None):
    return {'version':1, 'polygonSpace':'sent-image', 'imageHash':'a'*64, 'originalSize':list(size),
            'processedSize':list(size), 'sentSize':list(sent),
            'originalToProcessed':matrix or [[1,0,0],[0,1,0],[0,0,1]],
            'response':{'TextDetections':[{'DetectedText':'12. 已知条件', 'Confidence':98,
                'Polygon':[{'X':400,'Y':300},{'X':600,'Y':300},{'X':600,'Y':350},{'X':400,'Y':350}]}]}}


class CloudGeometryTests(unittest.TestCase):
    def adapt(self, e):
        return cg.cloud_rows(e,image_hash='a'*64,width=e['originalSize'][0],height=e['originalSize'][1])

    def test_adapter_exists(self):
        self.assertTrue(callable(getattr(cg,'cloud_rows',None)))

    def test_resize_inverse_uses_actual_sent_dimensions(self):
        rows=self.adapt(envelope())
        self.assertEqual(rows[0]['quad'],[[800,600],[1200,600],[1200,700],[800,700]])
        self.assertEqual(rows[0]['score'],.98)
        e=envelope((4001,3001),(2000,1500))
        p=self.adapt(e)[0]['quad'][0]
        self.assertAlmostEqual(p[0],800.2);self.assertAlmostEqual(p[1],600.2)

    def test_crop_translation_and_resize_are_inverted_in_order(self):
        e=envelope(matrix=[[1,0,-100],[0,1,-200],[0,0,1]])
        e['processedSize']=[2000,1500];e['sentSize']=[1000,750]
        self.assertEqual(self.adapt(e)[0]['quad'][0],[900,800])

    def test_perspective_inverse_round_trip(self):
        import numpy as np
        H=np.array([[.8,.1,-10],[.02,.9,-20],[.0001,.0002,1.]])
        expected=[[200,300],[400,300],[400,350],[200,350]]
        e=envelope(matrix=H.tolist());e['sentSize']=e['processedSize']
        pts=[]
        for x,y in expected:
            p=H@np.array([x,y,1.]);pts.append({'X':float(p[0]/p[2]),'Y':float(p[1]/p[2])})
        e['response']['TextDetections'][0]['Polygon']=pts
        self.assertTrue(np.allclose(self.adapt(e)[0]['quad'],expected))

    def test_wrong_hash_and_dimensions_rejected(self):
        for changes in [{'imageHash':'b'*64},{'originalSize':[0,3000]}, {'sentSize':[0,1]},
                        {'originalToProcessed':[[0,0,0]]*3},{'version':2}]:
            e=envelope();e.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.adapt(e)
        with self.assertRaises(ValueError):cg.cloud_rows(envelope(),image_hash='a'*64,width=1000,height=1000)

    def test_provider_errors_unknown_angle_and_bad_polygons_rejected(self):
        for mutate in [lambda e:e['response'].update(Error={'Code':'FailedOperation'}),
                       lambda e:e.update(polygonSpace='rotation-corrected-image'),
                       lambda e:e['response']['TextDetections'][0].update(Confidence=101),
                       lambda e:e['response']['TextDetections'][0]['Polygon'][0].update(X=float('nan')),
                       lambda e:e['response']['TextDetections'][0]['Polygon'][0].update(X=-500)]:
            e=envelope();mutate(e)
            with self.assertRaises(ValueError):self.adapt(e)

    def test_missing_metadata_does_not_guess(self):
        e=envelope();del e['originalToProcessed']
        with self.assertRaises(ValueError):self.adapt(e)

    def test_small_provider_border_overflow_is_clipped_and_marked(self):
        e=envelope((1080,1920),(1080,1920))
        e['response']['TextDetections'][0]['Polygon']=[{'X':1030,'Y':223},{'X':1090,'Y':216},{'X':1095,'Y':251},{'X':1035,'Y':259}]
        row=self.adapt(e)[0]
        self.assertTrue(row['clipped'])
        self.assertEqual(max(p[0] for p in row['quad']),1080)

    def test_empty_text_does_not_create_anchor(self):
        e=envelope();e['response']['TextDetections'][0]['DetectedText']=' '
        self.assertEqual(self.adapt(e),[])

    def test_reported_text_angle_is_not_a_second_pixel_rotation(self):
        e=envelope();e['response']['Angle']=359.9
        self.assertEqual(self.adapt(e)[0]['quad'][0],[800,600])

if __name__=='__main__':unittest.main()
