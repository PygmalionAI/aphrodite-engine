import torch
import unittest

from aphrodite.modeling.layers.sampler import _apply_typical_sampling, _apply_clone_typical_sampling


class TestTypicalSampling(unittest.TestCase):
    def setUp(self):
        self.batch_sizes = [1, 10, 100]
        self.logits_sizes = [10, 100, 1000]

    def test_consistency(self):
        for batch_size in self.batch_sizes:
            for logits_size in self.logits_sizes:
                logits = torch.randn(batch_size, logits_size)
                typical_p = torch.rand(1)
                typical_threshold = torch.rand(1)

                original_result = _apply_typical_sampling(logits.clone(), typical_p.clone(), typical_threshold.clone())
                modified_result = _apply_clone_typical_sampling(logits.clone(), typical_p.clone(), typical_threshold.clone())

                self.assertTrue(torch.allclose(original_result, modified_result))

if __name__ == '__main__':
    unittest.main()