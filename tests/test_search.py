from queue import PriorityQueue

from music21.pitch import Pitch

from main import CantusFirmus, PrioritizedCantusFirmus


def test_prio_queue():
    cf1 = CantusFirmus("C4 D4 G3")
    cf2 = CantusFirmus("C4 D4 E3")
    queue = PriorityQueue()
    queue.put(PrioritizedCantusFirmus(cf1, 1))
    queue.put(PrioritizedCantusFirmus(cf2, 0))
    assert cf1.could_be_valid_adding_notes
    assert cf2.could_be_valid_adding_notes
    assert queue.get().item.s[-1] == Pitch("E3")