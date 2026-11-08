import pytest

from promoai.timecode import TimecodeError, format_timecode, parse_timecode


@pytest.mark.parametrize(
    "text, ms",
    [
        ("00:00.00", 0),
        ("00:04.80", 4800),
        ("00:04.8", 4800),       # one fractional digit means tenths, not hundredths
        ("00:04.080", 4080),
        ("00:04", 4000),         # no fraction
        ("1:02.5", 62500),       # short minutes
        ("(00:11.30)", 11300),   # parentheses as in the real files
        ("02:18.00", 138000),
        ("00:04,80", 4800),      # comma decimal separator
        ("01:00:00.00", 3600000),  # HH:MM:SS
        ("66:40.00", 4000000),
    ],
)
def test_parse_timecode(text, ms):
    assert parse_timecode(text) == ms


@pytest.mark.parametrize("bad", ["", "abc", "00:99.00", "12", "--:--", "00:04.8.1", "4.8"])
def test_parse_timecode_rejects_garbage(bad):
    with pytest.raises(TimecodeError):
        parse_timecode(bad)


def test_integer_milliseconds_are_exact():
    # 66.7 and 66.8 as floats are not exact; as ints the difference is exactly 100 ms
    assert parse_timecode("01:06.80") - parse_timecode("01:06.70") == 100


def test_format_roundtrip():
    for ms in (0, 4800, 62500, 138000):
        assert parse_timecode(format_timecode(ms)) == ms
