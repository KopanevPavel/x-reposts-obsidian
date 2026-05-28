from xreposts_obsidian.utils import safe_filename


def test_safe_filename_removes_bad_chars():
    assert safe_filename('a/b:c*"d<e>f|g') == "a b c d e f g"
