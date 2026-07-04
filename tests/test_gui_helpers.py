import unittest
from unittest.mock import patch

from gui import _browse_file, _browse_folder


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class TestGuiBrowseHelpers(unittest.TestCase):
    @patch("gui.filedialog.askopenfilename", return_value="/pictures/photo.jpg")
    def test_convert_file_browse_opens_one_file_dialog(self, askopenfilename):
        input_var = FakeVar()

        _browse_file(input_var, object())

        askopenfilename.assert_called_once()
        self.assertEqual(input_var.get(), "/pictures/photo.jpg")

    @patch("gui.filedialog.askopenfilename", return_value="/pictures/photo.jpg")
    def test_judge_file_browse_suggests_output(self, askopenfilename):
        input_var = FakeVar()
        output_var = FakeVar()

        _browse_file(input_var, object(), output_var)

        askopenfilename.assert_called_once()
        self.assertEqual(input_var.get(), "/pictures/photo.jpg")
        self.assertEqual(output_var.get(), "/pictures/photo-judged")

    @patch("gui.filedialog.askdirectory", return_value="/pictures/to-review")
    def test_judge_browse_opens_one_folder_dialog_and_suggests_output(self, askdirectory):
        input_var = FakeVar()
        output_var = FakeVar()

        _browse_folder(input_var, object(), output_var)

        askdirectory.assert_called_once()
        self.assertEqual(input_var.get(), "/pictures/to-review")
        self.assertEqual(output_var.get(), "/pictures/to-review-judged")


if __name__ == "__main__":
    unittest.main()
