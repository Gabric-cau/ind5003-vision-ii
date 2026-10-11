import unittest
from scripts.apply_goalkeeper_user_review import apply_reviews


class TestUserReview(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(clip_id='a', dive_direction='unknown', dive_height='unknown', review_required=True),
                     dict(clip_id='b', dive_direction='image_left', review_required=True)]
        self.annotation = dict(source='explicit_user_chat_video_review', reviews=[
            dict(clip_id='a', reviewed_direction='not_applicable', display_direction=False)])

    def test_preserve_auto_and_qa(self):
        result = apply_reviews(self.rows, self.annotation)
        self.assertEqual(result[0]['dive_direction'], 'unknown')
        self.assertEqual(result[0]['dive_height'], 'unknown')
        self.assertTrue(result[0]['review_required'])
        self.assertNotIn('final_dive_direction', self.rows[0])
        self.assertEqual(result[1]['final_dive_direction'], 'image_left')
        self.assertFalse(result[1]['direction_user_reviewed'])

    def test_protest_not_center(self):
        result = apply_reviews(self.rows, self.annotation)[0]
        self.assertEqual(result['final_dive_direction'], 'not_applicable')
        self.assertIsNone(result['display_dive_direction'])

    def test_small_dive_can_be_right(self):
        self.annotation['reviews'][0].update(reviewed_direction='image_right', display_direction=True)
        result = apply_reviews(self.rows, self.annotation)[0]
        self.assertEqual(result['display_dive_direction'], 'image_right')
        self.assertEqual(result['final_direction_source'], 'explicit_user_chat_video_review')

    def test_unknown_id_rejected(self):
        self.annotation['reviews'][0]['clip_id'] = 'unseen'
        with self.assertRaises(ValueError):
            apply_reviews(self.rows, self.annotation)

    def test_duplicate_rejected(self):
        self.annotation['reviews'] *= 2
        with self.assertRaises(ValueError):
            apply_reviews(self.rows, self.annotation)


if __name__ == '__main__':
    unittest.main()
