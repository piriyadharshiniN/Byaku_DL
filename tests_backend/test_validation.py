import unittest
import cv2
import numpy as np
from pathlib import Path
from backend.validation import validate_image_bytes, validate_video_file
from backend.exceptions import ValidationException

class TestValidation(unittest.TestCase):
    def setUp(self):
        # Create a valid JPEG
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = [255, 100, 50]
        _, self.valid_jpeg_bytes = cv2.imencode(".jpg", img)
        self.valid_jpeg_bytes = self.valid_jpeg_bytes.tobytes()

        # Create a valid PNG
        _, self.valid_png_bytes = cv2.imencode(".png", img)
        self.valid_png_bytes = self.valid_png_bytes.tobytes()

    def test_valid_image(self):
        img, w, h = validate_image_bytes(self.valid_jpeg_bytes, filename="test.jpg")
        self.assertEqual(w, 100)
        self.assertEqual(h, 100)
        self.assertIsNotNone(img)

        img_png, w_png, h_png = validate_image_bytes(self.valid_png_bytes, filename="test.png")
        self.assertEqual(w_png, 100)
        self.assertEqual(h_png, 100)

    def test_empty_image(self):
        with self.assertRaises(ValidationException) as ctx:
            validate_image_bytes(b"", filename="empty.jpg")
        self.assertEqual(ctx.exception.code, "EMPTY_FILE")

    def test_wrong_extension(self):
        with self.assertRaises(ValidationException) as ctx:
            validate_image_bytes(self.valid_jpeg_bytes, filename="test.txt")
        self.assertEqual(ctx.exception.code, "UNSUPPORTED_EXTENSION")

    def test_corrupt_magic_bytes(self):
        corrupt_bytes = b"NOT_AN_IMAGE_HEADER_BUT_LONG_ENOUGH"
        with self.assertRaises(ValidationException) as ctx:
            validate_image_bytes(corrupt_bytes, filename="corrupt.jpg")
        self.assertEqual(ctx.exception.code, "INVALID_MAGIC_BYTES")

    def test_corrupt_payload(self):
        # Starts with valid JPEG header but corrupt payload
        corrupt_jpeg = b"\xff\xd8\xff\xe0" + b"random_corrupted_payload" * 10
        with self.assertRaises(ValidationException) as ctx:
            validate_image_bytes(corrupt_jpeg, filename="corrupt.jpg")
        self.assertEqual(ctx.exception.code, "IMAGE_DECODE_FAILED")

    def test_image_size_limit(self):
        # Oversized dummy bytes
        large_bytes = b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024)
        with self.assertRaises(ValidationException) as ctx:
            validate_image_bytes(large_bytes, filename="large.jpg")
        self.assertEqual(ctx.exception.status_code, 413)

if __name__ == "__main__":
    unittest.main()
