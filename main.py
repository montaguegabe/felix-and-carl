import copy
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
from time import sleep
from typing import Union, Any

import numpy as np
from music21.interval import GenericInterval, Interval
from music21.key import Key
from music21.note import Note
from music21.pitch import Pitch
from music21.stream import Part

INVALID = float('-inf')

VERBOSE = False
UPDATE_EVERY = 100


def log(string: str):
    if VERBOSE:
        print(string)


def info(string: str):
    print(string)


class Motion(Enum):
    DESC = -1
    NONE = 0
    ASC = 1

    @staticmethod
    def from_interval(interval: Interval) -> 'Motion':
        if interval.semitones > 0:
            return Motion.ASC
        elif interval.semitones < 0:
            return Motion.DESC
        else:
            return Motion.NONE


@dataclass
class OutlinedInterval:
    interval: Interval
    num_notes: int
    top_pitch_ind: int

    @property
    def is_asc(self):
        return self.interval.semitones > 0



c_major_pitch_classes = [pitch.pitchClass for pitch in Key('C').getScale().pitches]


class CantusFirmus:
    s: list[Pitch]  # must be only single pitches in this stream

    # Validity: 0 is best, negative float = not ideal, None = -inf = not valid.
    validity: float
    max_validity_adding_notes: float

    # Implied in page 7-8
    allowed_outlined_intervals = [
        Interval('m2'),
        Interval('M2'),
        Interval('m3'),
        Interval('M3'),
        Interval('P4'),
        Interval('P5'),
        Interval('m6'),
        Interval('M6'),
        Interval('P8')
    ]

    def __init__(self, notes: str = None):

        # Starting base case
        self.s = []

        self.has_out_of_scale_notes = False
        self.num_notes = 0
        self.num_leaps = 0
        self.num_leaps_larger_than_4th = 0
        self.num_deflections = 0
        self.highest_pitch = None
        self.highest_pitch_note_count = 0
        self.lowest_pitch = None
        self.lowest_pitch_note_count = 0
        self.num_leaps_gt_3rd_without_deflection = 0
        self.num_leaps_gt_3rd_without_stepwise_deflection = 0
        self.num_repeat_leap_in_same_direction = 0  # C E G -> 1, C E G B -> 2, C E G A F D -> 2
        self.max_num_consecutive_leaps = 0
        self.max_num_consecutive_steps_in_same_dir = 0
        self.max_num_consecutive_any_in_same_dir = 0
        self.num_leaps_gte_5th_in_same_direction = 0
        self.num_leaps_4th_in_same_direction = 0
        self.num_leaps_3rd_in_same_direction = 0
        self.outlined_intervals = []

        self.running_interval = None
        self.running_did_deflect = False
        self.running_num_consecutive_leaps = 0
        self.running_num_consecutive_steps_in_same_dir = 0
        self.running_num_any_in_same_dir = 0
        self.running_movement_start_pitch = None
        self.running_movement_end_pitch = None

        self.validity = INVALID
        self.max_validity_adding_notes = 0
        self.validation_comments = []

        # Add notes one at a time
        if notes:
            pitches = [Pitch(pitch_str) for pitch_str in notes.split(' ')]
            for pitch in pitches:
                self.add_note(pitch, inplace=True)

    def add_note(self, pitch: Pitch, inplace=False) -> Union['CantusFirmus', None]:
        target = self if inplace else copy.deepcopy(self)
        target._add_note_inplace(pitch)
        if not inplace:
            return target

    def _add_note_inplace(self, pitch):

        assert isinstance(pitch, Pitch)

        # Pre-append: gather info that will be overwritten
        previous_pitch: Pitch = self.s[-1] if self.s else None
        previous_pitch_ind = len(self.s)
        previous_interval: Interval = self.running_interval
        previous_motion = Motion.from_interval(previous_interval) if previous_interval is not None else None
        previous_num_any_in_same_dir = self.running_num_any_in_same_dir

        # Append note/pitch
        self.s.append(pitch)
        # pitch_ind = len(self.s)

        # Update number of notes
        self.num_notes += 1

        # Update whether any notes are out of scale
        if pitch.pitchClass not in c_major_pitch_classes:
            self.has_out_of_scale_notes = True

        # Update highest and lowest pitches, and their counts
        if self.highest_pitch is None or pitch > self.highest_pitch:
            self.highest_pitch = pitch
            self.highest_pitch_note_count = 1
        elif pitch == self.highest_pitch:
            self.highest_pitch_note_count += 1
        if self.lowest_pitch is None or pitch < self.lowest_pitch:
            self.lowest_pitch = pitch
            self.lowest_pitch_note_count = 1
        elif pitch == self.lowest_pitch:
            self.lowest_pitch_note_count += 1

        # Update running movement start pitch
        if self.running_movement_start_pitch is None:
            self.running_movement_start_pitch = pitch

        if previous_pitch is not None:

            # Update running interval
            self.running_interval = Interval(pitch, previous_pitch)
            running_motion = Motion.from_interval(self.running_interval)

            # Update number of leaps, number of consecutive leaps
            if not self.running_interval.isStep:
                self.num_leaps += 1
                self.running_num_consecutive_leaps += 1
                if self.running_num_consecutive_leaps > self.max_num_consecutive_leaps:
                    self.max_num_consecutive_leaps = self.running_num_consecutive_leaps
            else:
                self.running_num_consecutive_leaps = 0

            # Update number of leaps greater than a fourth
            if abs(self.running_interval.semitones) > Interval('P4').semitones:
                self.num_leaps_larger_than_4th += 1

            if previous_interval is not None:

                # Update deflections
                assert running_motion != Motion.NONE
                assert previous_motion != Motion.NONE
                if running_motion != previous_motion:
                    self.running_did_deflect = True
                    self.num_deflections += 1
                else:
                    self.running_did_deflect = False

                # Update presence of a leap greater than a third without a deflection (page 6)
                # Update the number of leaps greater than a third without a stepwise deflection (page 6)
                if abs(previous_interval.semitones) >= Interval('P4').semitones:
                    if not self.running_did_deflect:
                        self.num_leaps_gt_3rd_without_deflection += 1
                    elif not self.running_interval.isStep:
                        self.num_leaps_gt_3rd_without_stepwise_deflection += 1

            # Update the number of repeat leaps in the same direction (e.g. C E G would be 1 repeat leap in same direction)
            if self.running_num_consecutive_leaps > 1 and not self.running_did_deflect:
                self.num_repeat_leap_in_same_direction += 1

            # Update the number of consecutive steps in the same direction
            if self.running_interval.isStep and not self.running_did_deflect:
                self.running_num_consecutive_steps_in_same_dir += 1
                if self.running_num_consecutive_steps_in_same_dir > self.max_num_consecutive_steps_in_same_dir:
                    self.max_num_consecutive_steps_in_same_dir = self.running_num_consecutive_steps_in_same_dir
            else:
                self.running_num_consecutive_steps_in_same_dir = 0

            # Update the number of either steps or leaps (any) that don't change direction
            # Update running movement start and end pitches (for outlines)
            if not self.running_did_deflect:
                self.running_num_any_in_same_dir += 1
                if self.running_num_any_in_same_dir > self.max_num_consecutive_any_in_same_dir:
                    self.max_num_consecutive_steps_in_same_dir = self.running_num_any_in_same_dir
            else:
                self.running_num_any_in_same_dir = 0

            # Update the number of leaps a fifth or larger that don't change direction
            # Update the same counts for 3rds and 4ths
            if not self.running_did_deflect:
                if self.running_interval.semitones >= Interval('P5').semitones:
                    self.num_leaps_gte_5th_in_same_direction += 1
                elif self.running_interval == Interval('P4'):
                    self.num_leaps_4th_in_same_direction += 1
                elif self.running_interval == Interval('m3') or self.running_interval == Interval('M3'):
                    self.num_leaps_3rd_in_same_direction += 1

            # Update outlined intervals
            if self.running_did_deflect:
                self.running_movement_end_pitch = previous_pitch
                outlined_interval = OutlinedInterval(
                    interval=Interval(self.running_movement_start_pitch, self.running_movement_end_pitch),
                    num_notes=previous_num_any_in_same_dir,
                    top_pitch_ind=previous_pitch_ind
                )
                self.outlined_intervals.append(outlined_interval)
                self.running_movement_start_pitch = previous_pitch

        # Update validity
        (self.validity, self.max_validity_adding_notes, self.validation_comments) = self._compute_validity()

    def _compute_validity(self) -> tuple[float, float, list[str]]:
        """
        Let's make this function non-reliant on "running" variables.
        :returns: Tuple of current validity, validity upper bound if adding notes
        """

        class Penalties:
            leap_gt_3rd_without_stepwise_deflection = 1.0  # key word is "stepwise" - you always have to deflect in this case
            leap_4th_in_same_direction = 1.5
            leap_3rd_in_same_direction = 0.0

        validity = 0
        max_validity_adding_notes = 0
        comments = []

        # Check if in key (movements by P4 and P5 can introduce this)
        if self.has_out_of_scale_notes:
            return INVALID, INVALID, ["Out of scale"]

        # Page 4: No more than 16 notes, and no less than 8
        if self.num_notes > 16:
            return INVALID, INVALID, ["Too many notes"]
        if self.num_notes < 8:
            validity = INVALID

        # Page 4: Vocal range of a cantus firmus should be at most a 10th.
        # TODO: 5th or 6th is good.
        vocal_range = Interval(self.lowest_pitch, self.highest_pitch)
        if vocal_range.semitones > Interval('M10').semitones:
            return INVALID, INVALID, ["Cannot span more than a tenth"]

        # Page 5: Climax should not be repeated
        if self.highest_pitch_note_count > 1:
            validity = INVALID
            comments.append("Right now the climax pitch is repeated")

        # Page 6: Two to four leaps
        # TODO: Most contain 2 to 3 leaps
        if self.num_leaps < 2:
            validity = INVALID
        if self.num_leaps > 4:
            return INVALID, INVALID, ["Too many leaps"]

        # Page 6: Must change direction "several" times
        # TODO: Param for several
        several = 3
        if self.num_deflections < several:
            validity = INVALID
            comments.append("Eventually needs more change of direction")

        # Page 6: No more than 2 leaps larger than 4th
        if self.num_leaps_larger_than_4th > 2:
            return INVALID, INVALID, ["Too many leaps larger than a 4th"]

        # Page 6: Leaps followed by a change in direction
        # Most of the leaps should have a stepwise motion following
        if self.num_leaps_gt_3rd_without_deflection > 0:
            return INVALID, INVALID, ["Leaps should be followed by a change of direction"]
        leaps_gt_3rd_without_stepwise_deflection_decrement = self.num_leaps_gt_3rd_without_stepwise_deflection * Penalties.leap_gt_3rd_without_stepwise_deflection
        validity -= leaps_gt_3rd_without_stepwise_deflection_decrement
        max_validity_adding_notes -= leaps_gt_3rd_without_stepwise_deflection_decrement

        # Page 6: No two leaps in the same direction
        # Not more than two consecutive leaps
        if self.num_repeat_leap_in_same_direction > 0:
            return INVALID, INVALID, ["Two leaps in the same direction"]
        if self.max_num_consecutive_leaps > 2:
            return INVALID, INVALID, ["More than 2 leaps in a row is not allowed."]

        # Page 7: No more than 5 consecutive steps in same direction
        # TODO: "in most cases" - depends on length of cantus firmus
        if self.max_num_consecutive_steps_in_same_dir > 5:
            return INVALID, INVALID, ["No more than 5 steps in a row"]

        # Page 7: Leaps of a fifth or larger should change direction
        # "Indeed, even smaller leaps seem to function with maximum effectiveness when they reverse the preceding direction."
        if self.num_leaps_gte_5th_in_same_direction > 0:
            return INVALID, INVALID, ["Leaps greater than a 5th should change direction"]
        leaps_in_same_direction_decrement = self.num_leaps_4th_in_same_direction * Penalties.leap_4th_in_same_direction + self.num_leaps_3rd_in_same_direction * Penalties.leap_3rd_in_same_direction
        validity -= leaps_in_same_direction_decrement
        max_validity_adding_notes -= leaps_in_same_direction_decrement

        # Page 7: Guard against Example 1-9
        # We will say that 6 is the maximum you can go without deflection
        # TODO: Param for 6
        if self.max_num_consecutive_any_in_same_dir > 6:
            return INVALID, INVALID, ["More than 6 notes going in the same direction"]

        # Page 7-8: Avoid outlined dissonances
        # (Page 8: Do not leave the leading tone unresolved)
        leading_tone: Pitch = Interval('-m2').transposePitch(self.s[0])
        for outlined_interval in self.outlined_intervals:
            interval = outlined_interval.interval
            if interval.semitones < 0:
                interval = interval.reverse()
            if interval not in CantusFirmus.allowed_outlined_intervals:
                return INVALID, INVALID, ["Outlined a dissonance"]
            top_pitch = self.s[outlined_interval.top_pitch_ind]
            if top_pitch.pitchClass == leading_tone.pitchClass:
                validity = INVALID
                comments.append("Outlined interval with leading tone at top.")

        if self.s[-1] != self.s[0]:
            validity = INVALID

        if self.num_notes >= 2 and self.s[-2] != GenericInterval('second').transposePitch(self.s[0]):
            validity = INVALID

        return validity, max_validity_adding_notes, comments

    # Page 4
    step_up_intervals = [
        GenericInterval('second'),
    ]
    step_down_intervals = [interval.reverse() for interval in step_up_intervals]
    leap_up_intervals = [
        GenericInterval('third'),
        Interval('P4'),
        Interval('P5'),
        GenericInterval('sixth'),
        Interval('P8')
    ]
    leap_down_intervals = [interval.reverse() for interval in leap_up_intervals]
    all_allowed_intervals = step_up_intervals + step_down_intervals + leap_up_intervals + leap_down_intervals

    def next_interval_distribution(self) -> tuple[list[Union[Interval, GenericInterval]], list[float]]:
        """
        Most CF have 2-4 leaps. For length 8 this is 25% to 50%. For length 16 this is 12.5% to 25%. Seems like 25% is a good number for the probability of leaping.

        Independently there is whether to go up or down.  Let's say the probability of going away from the starting tonic should be 1/4 at 1 octave away, but should only asymptotically vanish to 0 with distance. The formula a / distance does this.  a / 12 semitones = 1/4, so a = 3.
        """
        p_leap = 0.33
        a = 3.0

        # Compute probability of going up/down
        tonic_offset = Interval(self.s[0], self.s[-1]).semitones
        p_asc = 0.5
        p_desc = 0.5
        if tonic_offset > 0:
            p_asc = min(a / abs(tonic_offset), 0.5)
            p_desc = 1.0 - p_asc
        elif tonic_offset < 0:
            p_desc = min(a / abs(tonic_offset), 0.5)
            p_asc = 1.0 - p_desc

        # Compute probability of leaping
        # TODO: If we are out of leaps don't leap anymore
        p_step = 1.0 - p_leap

        # Return the distribution
        all_allowed = CantusFirmus.all_allowed_intervals
        n_step_up = len(CantusFirmus.step_up_intervals)
        n_step_down = n_step_up  # you could assert this to be careful
        n_leap_up = len(CantusFirmus.leap_up_intervals)
        n_leap_down = n_leap_up

        # Operates on invariant:
        # all_allowed_intervals = step_up_intervals + step_down_intervals + leap_up_intervals + leap_down_intervals
        assert p_asc >= 0.0
        assert p_desc >= 0.0
        return (
            all_allowed,
            [p_step * p_asc / n_step_up] * n_step_up +
            [p_step * p_desc / n_step_down] * n_step_down +
            [p_leap * p_asc / n_leap_up] * n_leap_up +
            [p_leap * p_desc / n_leap_down] * n_leap_down
        )

    def with_adding_note_from_interval(self, interval: Interval) -> 'CantusFirmus':
        assert len(self.s) > 0
        return self.add_note(interval.transposePitch(self.s[-1]), inplace=False)

    @property
    def is_valid(self) -> bool:
        return self.validity != INVALID

    @property
    def could_be_valid_adding_notes(self) -> bool:
        return self.max_validity_adding_notes != INVALID

    @property
    def penalty(self) -> float:
        return -self.validity

    @property
    def min_penalty_adding_notes(self) -> float:
        return -self.max_validity_adding_notes

    def show(self, fmt=None):
        p = Part()
        for pitch in self.s:
            note = Note()
            note.pitch = pitch
            p.append(note)
        p.show(fmt)

    def add_notes_search(self) -> 'CantusFirmus':

        frontier = PriorityQueue()
        self_copy = copy.deepcopy(self)
        frontier.put(PrioritizedCantusFirmus(self_copy, 0))

        i = 0
        while not frontier.empty():
            prio_cf: PrioritizedCantusFirmus = frontier.get()
            to_explore: CantusFirmus = prio_cf.item
            log(f"Exploring {to_explore}")
            if i % UPDATE_EVERY == 0:
                info(str(to_explore))
            # sleep(1)
            (possible_next_pitches, weights) = to_explore.next_interval_distribution()
            sampled_intervals = np.random.choice(possible_next_pitches, len(possible_next_pitches), replace=False,
                                                 p=weights)
            for draw_order, interval in enumerate(sampled_intervals):
                new_state: CantusFirmus = to_explore.with_adding_note_from_interval(interval)
                if new_state.is_valid:
                    return new_state
                # log(f"Considering {new_state}")
                if new_state.could_be_valid_adding_notes:
                    # log(f"Adding to frontier: {new_state}")
                    frontier.put(PrioritizedCantusFirmus(new_state, draw_order))

            i += 1

    def __repr__(self):
        return ' '.join([str(pitch) for pitch in self.s])


@dataclass(order=True)
class PrioritizedCantusFirmus:
    min_penalty: float
    neg_num_notes: int  # longer cantus firmi get higher priority (DFS)
    draw_order: int  # weighted randomized
    item: Any = field(compare=False)

    def __init__(self, cantus_firmus: CantusFirmus, draw_order: int):
        self.min_penalty = cantus_firmus.min_penalty_adding_notes
        self.neg_num_notes = -cantus_firmus.num_notes
        self.draw_order = draw_order
        self.item = cantus_firmus


def main():
    # np.random.seed(2)
    start = CantusFirmus('C4')
    result = start.add_notes_search()
    print(f"Penalty: {result.penalty}")
    # result.show('text')
    result.show()


if __name__ == "__main__":
    main()
