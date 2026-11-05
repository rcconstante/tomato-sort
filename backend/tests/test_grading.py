from app.grading import UNCERTAIN, TrackVotes, category, crossed, side_of_line, within_segment


def test_category_checks_unripe_before_ripe():
    assert category("unripe") == "unripe"
    assert category("Ripe") == "ripe"
    assert category("semi") == "semi"


def test_votes_pick_stable_majority():
    v = TrackVotes()
    for _ in range(8):
        v.add("ripe", 0.9, 0.5)
    v.add("unripe", 0.6, 0.5)
    g = v.finalize(min_observations=3, min_vote_share=0.65, certain_conf=0.5)
    assert g.label == "ripe"
    assert g.observations == 9


def test_votes_uncertain_when_too_few_or_split():
    v = TrackVotes()
    v.add("ripe", 0.9, 0.5)
    assert v.finalize(3, 0.65, 0.5).label == UNCERTAIN

    v = TrackVotes()
    for _ in range(3):
        v.add("ripe", 0.8, 0.5)
        v.add("unripe", 0.8, 0.5)
    assert v.finalize(3, 0.65, 0.5).label == UNCERTAIN


def test_votes_uncertain_when_low_confidence():
    v = TrackVotes()
    for _ in range(5):
        v.add("ripe", 0.3, 0.5)
    g = v.finalize(3, 0.65, 0.5)
    assert g.label == UNCERTAIN and "low confidence" in g.reason
    assert g.predicted_label == "ripe"
    assert g.vote_share == 1.0
    assert "30% / 50% required" in g.reason


def test_simple_mode_uses_winning_label_without_confidence_gate():
    votes = TrackVotes()
    for _ in range(4):
        votes.add("unripe", 0.35, 0.5)
    grade = votes.finalize(2, 0.65, 0.5, require_certainty=False)
    assert grade.label == "unripe" and grade.confidence == 0.35


def test_simple_mode_still_requires_observations():
    votes = TrackVotes()
    votes.add("ripe", 0.9, 0.5)
    assert votes.finalize(2, 0.65, 0.5, require_certainty=False).label == UNCERTAIN


def test_simple_mode_does_not_guess_on_equal_votes():
    votes = TrackVotes()
    votes.add("ripe", 0.8, 0.5)
    votes.add("unripe", 0.8, 0.5)
    grade = votes.finalize(2, 0.65, 0.5, require_certainty=False)
    assert grade.label == UNCERTAIN and "no winning class" in grade.reason


def test_line_crossing_direction():
    # horizontal line left→right at y=10; positive side is below (image y grows downward)
    above = side_of_line(5, 5, 0, 10, 20, 10)
    below = side_of_line(5, 15, 0, 10, 20, 10)
    assert above < 0 < below
    assert crossed(above, below, "forward")
    assert not crossed(above, below, "reverse")
    assert crossed(below, above, "reverse")
    assert crossed(below, above, "both")
    assert not crossed(above, above, "both")


def test_counted_exactly_once_over_many_frames():
    counted = 0
    prev = None
    done = False
    for y in range(0, 30):             # centroid moves down through the line and stays below
        s = side_of_line(10, y, 0, 10, 20, 10)
        if not done and prev is not None and crossed(prev, s, "forward"):
            counted += 1
            done = True
        if s != 0:
            prev = s
    assert counted == 1


def test_within_segment():
    assert within_segment(10, 10, 0, 10, 20, 10)
    assert not within_segment(40, 10, 0, 10, 20, 10)
