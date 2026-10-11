import unittest
from pathlib import Path
from scripts.search_player_evidence import search_records


class EvidenceSearchTests(unittest.TestCase):
    def data(self):
        return {'tracklets':[{'clip_id':'rus_cro_vida','segment_id':3,'tracklet_id':0,
            'start_time_sec':12,'end_time_sec':14,'expected_query':{'number':21,'surname':'vida'},
            'upright_evidence':{'number':10,'name_tokens':[{'text':'MODRIC','frames':[1,9],'span_sec':1}]},
            'upright_query':{'status':'number_conflict'},'keyframes':[{'image':'crops/a.jpg'}]}]}
    def test_cross_clip_match_not_lost_to_containing_clip_conflict(self):
        r=search_records(self.data(),Path('/tmp'),number=10,name='Modrić')[0]
        self.assertTrue(r['requested_number_match']);self.assertTrue(r['requested_name_match'])
        self.assertEqual(r['containing_clip_query_status'],'number_conflict')
        self.assertFalse(r['identity_verified'])
    def test_both_fields_required(self):
        self.assertEqual(search_records(self.data(),Path('/tmp'),number=21,name='Modric'),[])
    def test_clip_filter(self):
        self.assertEqual(search_records(self.data(),Path('/tmp'),clip='other',number=10),[])
    def test_unknown_number_not_filled_from_metadata(self):
        d=self.data();d['tracklets'][0]['upright_evidence']['number']=None
        self.assertEqual(search_records(d,Path('/tmp'),number=21),[])
    def test_clip_only_conflicts_are_opt_in(self):
        self.assertEqual(search_records(self.data(),Path('/tmp'),clip='vida'),[])
        self.assertEqual(len(search_records(self.data(),Path('/tmp'),clip='vida',include_conflicts=True)),1)


if __name__=='__main__':unittest.main()
