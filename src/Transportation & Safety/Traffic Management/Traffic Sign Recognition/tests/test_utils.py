import unittest
from utils import preprocess_image
import numpy as np

class TestUtils(unittest.TestCase):
    def test_preprocess_image(self):
        img = preprocess_image('test_image.jpg')
        self.assertEqual(img.shape, (30, 30, 3))
        self.assertEqual(img.dtype, np.uint8)