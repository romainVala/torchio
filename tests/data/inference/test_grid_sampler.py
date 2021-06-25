#!/usr/bin/env python

from copy import copy
import torchio as tio
from ...utils import TorchioTestCase


class TestGridSampler(TorchioTestCase):
    """Tests for `GridSampler`."""

    def test_locations(self):
        patch_size = 5, 20, 20
        patch_overlap = 2, 4, 6
        sampler = tio.GridSampler(
            subject=self.sample_subject,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
        fixture = [
            [0, 0, 0, 5, 20, 20],
            [0, 0, 10, 5, 20, 30],
            [3, 0, 0, 8, 20, 20],
            [3, 0, 10, 8, 20, 30],
            [5, 0, 0, 10, 20, 20],
            [5, 0, 10, 10, 20, 30],
        ]
        locations = sampler.locations.tolist()
        self.assertEqual(locations, fixture)

    def test_large_patch(self):
        with self.assertRaises(ValueError):
            tio.GridSampler(self.sample_subject, (5, 21, 5), (0, 2, 0))

    def test_large_overlap(self):
        with self.assertRaises(ValueError):
            tio.GridSampler(self.sample_subject, (5, 20, 5), (2, 4, 6))

    def test_odd_overlap(self):
        with self.assertRaises(ValueError):
            tio.GridSampler(self.sample_subject, (5, 20, 5), (2, 4, 3))

    def test_single_location(self):
        sampler = tio.GridSampler(self.sample_subject, (10, 20, 30), 0)
        fixture = [[0, 0, 0, 10, 20, 30]]
        self.assertEqual(sampler.locations.tolist(), fixture)

    def test_subject_shape(self):
        patch_size = 5, 20, 20
        patch_overlap = 2, 4, 6
        initial_shape = copy(self.sample_subject.shape)
        tio.GridSampler(
            self.sample_subject,
            patch_size,
            patch_overlap,
            padding_mode='reflect',
        )
        final_shape = self.sample_subject.shape
        self.assertEqual(initial_shape, final_shape)

    def test_bad_subject(self):
        with self.assertRaises(ValueError):
            patch_size = 88
            tio.GridSampler(patch_size)
