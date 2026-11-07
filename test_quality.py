import unittest
from unittest.mock import patch
from test_worker import request
import worker

def row(text,y):return {'quad':[[40,y],[700,y],[700,y+25],[40,y+25]],'text':text,'score':.9}

class QualityTests(unittest.TestCase):
    def test_uncovered_stems_above_three_to_five_warn_without_inventing_questions(self):
        rows=[row('函数f(x)零点所在的大致区间为',90),row('函数h(x)有两个不同的零点',190),row('3.若函数有两个不同的零点',320),row('4.已知函数满足下列条件',550),row('5.讨论函数零点个数',850)]
        with patch('worker.recognize',return_value=rows):page=worker.process_page(request())
        self.assertEqual([b['printedNumber'] for b in page['blocks']],['3','4','5'])
        self.assertTrue(page.get('warnings'))
        self.assertTrue(all(any(n.startswith('本地转写待补齐：') for n in b['unreadableParts']) for b in page['blocks']))
        self.assertFalse(any('漏题' in n for b in page['blocks'] for n in b['unreadableParts']))

    def test_following_page_header_does_not_mean_missing_one_to_eleven(self):
        rows=[row('数学单元测验',25),row('12.求质量是多少',100),row('13.已知函数求零点',500)]
        with patch('worker.recognize',return_value=rows):page=worker.process_page(request())
        self.assertFalse(page.get('warnings'))

    def test_empty_page_warns_to_retake_or_manually_select(self):
        with patch('worker.recognize',return_value=[]):page=worker.process_page(request())
        self.assertEqual(page['blocks'],[])
        self.assertTrue(page.get('warnings'))

if __name__=='__main__':unittest.main()
