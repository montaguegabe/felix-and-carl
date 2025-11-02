from main import CantusFirmus


def test_consecutive_leaps_in_same_direction():
    # Consecutive leaps not allowed in the same direction.
    cf = CantusFirmus("C4 D4 G3 E3")
    assert not cf.could_be_valid_adding_notes


def test_leap_4th_deflection():
    # Leap of 4th needs a deflection
    cf = CantusFirmus("C4 D4 G4 E3")
    assert cf.penalty > 0

    cf2 = CantusFirmus("C4 D4 G4 A4")
    assert not cf2.could_be_valid_adding_notes


def test_leap_4th_same_direction():
    # Leap of 4th in same direction is penalized
    cf = CantusFirmus("C4 D4 C4 G3 A3")
    assert cf.penalty > 0


def test_outlined_dissonance():
    cf = CantusFirmus("C4 B3 C4 E4 F4 E4 D4 C4")
    assert not cf.could_be_valid_adding_notes
    assert cf.outlined_intervals[1].interval not in CantusFirmus.allowed_outlined_intervals


def test_leading_tone_in_top_of_outlined():
    cf = CantusFirmus("C4 A4 G4 B4 A4")
    assert not cf.could_be_valid_adding_notes


def test_normal_cf_beginning():
    cf = CantusFirmus("C4 D4")
    assert cf.could_be_valid_adding_notes
