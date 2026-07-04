from pathlib import Path

from PIL import Image

from upscaler import collect_images, fit_dimensions, target_dimensions


def test_target_dimensions_preserves_landscape_aspect():
    assert target_dimensions(1920, 1080, 3840) == (3840, 2160)


def test_target_dimensions_preserves_portrait_aspect():
    assert target_dimensions(1080, 1920, 3840) == (2160, 3840)


def test_target_dimensions_never_downscales():
    assert target_dimensions(5000, 3000, 3840) == (5000, 3000)


def test_quest_box_fits_landscape_without_distortion():
    assert fit_dimensions(1920, 1080, 2064, 2208) == (2064, 1161)


def test_quest_box_fits_portrait_without_distortion():
    assert fit_dimensions(1080, 1920, 2064, 2208) == (1242, 2208)


def test_quest_box_downsizes_oversized_source():
    assert fit_dimensions(3840, 2160, 2064, 2208) == (2064, 1161)


def test_collect_images_accepts_file_and_non_recursive_folder(tmp_path: Path):
    jpg = tmp_path / "a.jpg"
    png = tmp_path / "b.PNG"
    Image.new("RGB", (2, 2)).save(jpg)
    Image.new("RGB", (2, 2)).save(png)
    (tmp_path / "notes.txt").write_text("no")
    assert collect_images(str(jpg)) == [str(jpg)]
    assert collect_images(str(tmp_path)) == [str(jpg), str(png)]
