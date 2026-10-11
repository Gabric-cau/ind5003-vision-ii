import unittest
import numpy as np
from src.ball_mask_seed import white_mask, measure_mask_support, choose_seed, associate_with_seed


def det(f, x, y, ident, confidence=.6):
    return {'frame': f, 'x': x, 'y': y, 'width': 12, 'height': 12,
        'stabilized_xy': [x, y], 'confidence': confidence,
        'sources': ['full'], 'detection_id': ident, 'motion_mean_abs_gray': 30.}


class MaskSeedTests(unittest.TestCase):
    def test_repeated_grass_false_positive_does_not_defeat_visible_ball(self):
        frame = np.full((140, 180, 3), (40, 140, 35), dtype=np.uint8)
        frame[96:104, 96:104] = 220
        mask = white_mask(frame)
        ds = {str(f): [det(f, 100, 20, f'{f}:grass')] for f in range(13)}
        for f in range(7):
            ds[str(f)].append(det(f, 100, 100, f'{f}:ball'))
        evidence = {d['detection_id']: measure_mask_support(mask, d) for rows in ds.values() for d in rows}
        old, _ = choose_seed(ds, 10, [[100, 100]], evidence, use_mask=False, radius=85)
        new, _ = choose_seed(ds, 10, [[100, 100]], evidence, use_mask=True, radius=85)
        self.assertTrue(all(d['y'] == 20 for d in old))
        self.assertTrue(all(d['y'] == 100 for d in new))

    def test_mask_loss_cannot_remove_flight_observations(self):
        ds = {str(f): [det(f, 1000, 400, f'{f}:ball')] for f in range(11)}
        for f in range(11, 26):
            ds[str(f)] = [det(f, 1000 - 30 * (f - 10), 400 - 10 * (f - 10), f'{f}:flight')]
        evidence = {d['detection_id']: {'mask_support': d['frame'] <= 10} for rows in ds.values() for d in rows}
        seed, diag = choose_seed(ds, 10, [[1000, 400]], evidence, use_mask=True, radius=35)
        path, _ = associate_with_seed(ds, 10, [[1000, 400]], [550, 250], seed, diag)
        self.assertEqual(path[-1]['frame'], 25)
        self.assertFalse(evidence[path[-1]['detection_id']]['mask_support'])

    def test_foreign_seed_cannot_be_injected(self):
        with self.assertRaises(ValueError):
            associate_with_seed({'0': [det(0, 100, 100, 'observed')]}, 0,
                [[100, 100]], [20, 20], [det(0, 100, 100, 'invented')],
                {'pixel_mask_used_for_seed': True})


if __name__ == '__main__':
    unittest.main()
