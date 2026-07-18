import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault(
    "customtkinter",
    SimpleNamespace(
        StringVar=object,
        BooleanVar=object,
        CTk=object,
        CTkFrame=object,
        CTkTextbox=object,
        CTkProgressBar=object,
        CTkLabel=object,
        CTkButton=object,
        CTkEntry=object,
        CTkOptionMenu=object,
        CTkCheckBox=object,
        CTkScrollbar=object,
        CTkTabview=object,
        CTkFont=object,
        set_appearance_mode=lambda *_args, **_kwargs: None,
        set_default_color_theme=lambda *_args, **_kwargs: None,
    ),
)

from gui import (
    _decorate_model_menu_values,
    _OutputAutofillController,
    _browse_file,
    _browse_folder,
    _input_kind,
    _merge_backend_model_choices,
    _models_url,
    _progress_display,
    _split_labels,
    _strip_model_menu_label,
    _video_option_int,
)


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeTraceVar(FakeVar):
    def __init__(self, value=""):
        super().__init__(value)
        self._callbacks = []

    def trace_add(self, _mode, callback):
        self._callbacks.append(callback)

    def set(self, value):
        self.value = value
        for callback in list(self._callbacks):
            callback()


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

    @patch("gui.filedialog.askdirectory", return_value="/pictures/second")
    def test_judge_browse_updates_auto_suggested_output_when_input_changes(self, askdirectory):
        input_var = FakeVar("/pictures/first")
        output_var = FakeVar("/pictures/first-judged")

        _browse_folder(input_var, object(), output_var)

        askdirectory.assert_called_once()
        self.assertEqual(input_var.get(), "/pictures/second")
        self.assertEqual(output_var.get(), "/pictures/second-judged")

    @patch("gui.filedialog.askdirectory", return_value="/pictures/second")
    def test_judge_browse_preserves_manual_output_override(self, askdirectory):
        input_var = FakeVar("/pictures/first")
        output_var = FakeVar("/custom/output")

        _browse_folder(input_var, object(), output_var)

        askdirectory.assert_called_once()
        self.assertEqual(input_var.get(), "/pictures/second")
        self.assertEqual(output_var.get(), "/custom/output")

    @patch("gui.filedialog.askdirectory", return_value="/pictures/to-sort")
    def test_organizer_browse_uses_organized_suffix(self, askdirectory):
        input_var = FakeVar()
        output_var = FakeVar()

        _browse_folder(input_var, object(), output_var, "organized")

        self.assertEqual(output_var.get(), "/pictures/to-sort-organized")

    def test_output_autofill_replaces_default_output_until_user_overrides(self):
        input_var = FakeTraceVar()
        output_var = FakeTraceVar("/workspace/output")
        _OutputAutofillController(input_var, output_var, "judged")

        input_var.set("/pictures/to-review")
        self.assertEqual(output_var.get(), "/pictures/to-review-judged")

        output_var.set("/custom/output")
        input_var.set("/pictures/second")
        self.assertEqual(output_var.get(), "/custom/output")

    def test_output_autofill_reenables_when_user_returns_to_suggested_path(self):
        input_var = FakeTraceVar("/pictures/first")
        output_var = FakeTraceVar("/pictures/first-judged")
        _OutputAutofillController(input_var, output_var, "judged")

        output_var.set("/custom/output")
        input_var.set("/pictures/second")
        self.assertEqual(output_var.get(), "/custom/output")

        output_var.set("/pictures/first-judged")
        input_var.set("/pictures/third")
        self.assertEqual(output_var.get(), "/pictures/third-judged")

    def test_split_labels_accepts_commas_and_newlines(self):
        self.assertEqual(
            _split_labels(" outdoors, indoors\nnight , "),
            ["outdoors", "indoors", "night"],
        )

    def test_models_url_accepts_base_or_chat_endpoint(self):
        self.assertEqual(_models_url("http://127.0.0.1:8001"), "http://127.0.0.1:8001/v1/models")
        self.assertEqual(_models_url("http://127.0.0.1:8001/v1"), "http://127.0.0.1:8001/v1/models")
        self.assertEqual(
            _models_url("http://127.0.0.1:8001/v1/chat/completions"),
            "http://127.0.0.1:8001/v1/models",
        )

    def test_merge_backend_model_choices_preserves_order_and_dedupes(self):
        merged = _merge_backend_model_choices(
            ["Qwen3.6-35B-A3B-MLX-4bit", "gemma-3-27b-it-8bit"],
            ["gemma-3-27b-it-8bit", "custom-model"],
            ["CUSTOM-model"],
        )

        self.assertEqual(
            merged,
            [
                "Qwen3.6-35B-A3B-MLX-4bit",
                "gemma-3-27b-it-8bit",
                "custom-model",
            ],
        )

    def test_decorate_model_menu_values_marks_selected_item_only(self):
        values = _decorate_model_menu_values(
            ["model-a", "model-b", "model-c"],
            "model-b",
        )

        self.assertEqual(
            values,
            ["model-a", "[Selected] model-b", "model-c"],
        )

    def test_strip_model_menu_label_removes_selected_prefix(self):
        self.assertEqual(_strip_model_menu_label("[Selected] model-b"), "model-b")
        self.assertEqual(_strip_model_menu_label("model-a"), "model-a")

    def test_progress_display_caps_in_flight_work_at_ninety_nine_percent(self):
        frac, percent = _progress_display(999, 1000)

        self.assertEqual(frac, 0.999)
        self.assertEqual(percent, 99)

    def test_progress_display_allows_one_hundred_percent_when_complete(self):
        frac, percent = _progress_display(1000, 1000)

        self.assertEqual(frac, 1.0)
        self.assertEqual(percent, 100)

    def test_video_option_int_accepts_original_and_numeric_labels(self):
        self.assertEqual(_video_option_int("Original"), -1)
        self.assertEqual(_video_option_int("720"), 720)
        self.assertEqual(_video_option_int("30 fps"), 30)

    def test_input_kind_detects_file_extensions_and_mixed_folders(self):
        import os
        import shutil
        import tempfile

        tmpdir = tempfile.mkdtemp()
        try:
            image = os.path.join(tmpdir, "photo.jpg")
            video = os.path.join(tmpdir, "clip.mp4")
            open(image, "w").close()
            open(video, "w").close()

            self.assertEqual(_input_kind(image), "image")
            self.assertEqual(_input_kind(video), "video")
            self.assertEqual(_input_kind(tmpdir), "mixed")
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
