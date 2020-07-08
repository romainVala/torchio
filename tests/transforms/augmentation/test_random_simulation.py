import torch
from torchio.transforms import RandomSimulation
from torchio import DATA
from ...utils import TorchioTestCase


class TestRandomSimulation(TorchioTestCase):
    """Tests for `RandomSimulation`."""
    def test_random_simulation(self):
        transform = RandomSimulation()
        transformed = transform(self.sample)
        self.assertIn('image', transformed)

    def test_deterministic_simulation(self):
        transform = RandomSimulation(
            label_keys='label',
            coefficients={1: {'mean': 0.5, 'std': 0}}
        )
        transformed = transform(self.sample)
        self.assertTrue(torch.eq(transformed['image'][DATA] == 0.5, self.sample['label'][DATA] == 1).prod())

    def test_inpainting(self):
        transform = RandomSimulation(
            label_keys='label',
            image_key='t1',
        )
        t1_indices = self.sample['label'][DATA] == 0
        transformed = transform(self.sample)
        self.assertTrue(torch.eq(transformed['t1'][DATA][t1_indices], self.sample['t1'][DATA][t1_indices]).prod())
