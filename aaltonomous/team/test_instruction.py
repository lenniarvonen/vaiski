"""python -m unittest aaltonomous/team/test_instruction.py -v"""
import itertools
import unittest

from aaltonomous.interface import tasks
from aaltonomous.team.instruction import parse

R, Y, B = "red", "yellow", "blue"
L, C, Rt = "left", "center", "right"

# Held-out style paraphrases nobody trained on: (text, expected mapping subset)
UNSEEN = [
    ("Sort the objects: red into right, blue into left, yellow into center.", {R: Rt, B: L, Y: C}),
    ("Reds on the left, yellows in the middle, blues on the right please.", {R: L, Y: C, B: Rt}),
    ("Blue things belong in the centre box; red ones go to the right-hand box and yellow ones to the left one.",
     {B: C, R: Rt, Y: L}),
    ("Left bin: yellow. Middle bin: blue. Right bin: red.", {Y: L, B: C, R: Rt}),
    ("Red, blue and yellow go to the left, center and right bins respectively.", {R: L, B: C, Y: Rt}),
    ("Put red in the rightmost bin and blue in the leftmost; yellow goes in the remaining one.", {R: Rt, B: L, Y: C}),
    ("Yellow -> left, red -> right, blue -> middle", {Y: L, R: Rt, B: C}),
    ("Could you place the crimson blocks in the central container, the golden ones on the far left, and "
     "everything else on the right?", {R: C, Y: L, B: Rt}),
    ("Fill the left box with blue pieces, the right box with yellow pieces and the middle box with red pieces.",
     {B: L, Y: Rt, R: C}),
    ("Red doesn't go left, it goes right. Blue goes left. Yellow in the middle.", {R: Rt, B: L, Y: C}),
    ("The blue ones go in the center bin, red ones in the left, whatever is left goes right.", {B: C, R: L, Y: Rt}),
    ("Everything goes into the middle bin except red, which goes on the right.", {"default": C, R: Rt}),
    ("Put all the objects into the left bin.", {"default": L}),
    ("Cubes into the left bin, cylinders into the center bin, triangles into the right bin.",
     {"cube": L, "cylinder": C, "triangle": Rt}),
    ("Put the red cubes in the right bin and everything else in the left bin.", {(R, "cube"): Rt, "default": L}),
    ("Sort by colour. Red: center. Blue: right. Yellow: left.", {R: C, B: Rt, Y: L}),
    ("For the right bin, take the yellow objects; for the left bin, the blue ones; red goes in the middle.",
     {Y: Rt, B: L, R: C}),
    ("Move each red item to the left container and each blue item to the right container. Yellow stays in the "
     "middle.", {R: L, B: Rt, Y: C}),
    ("Yellow objects should be placed in the bin on the right, blue objects in the bin on the left.",
     {Y: Rt, B: L, R: C}),
    ("Please sort the blocks into the right bins: red left, yellow right, blue center.", {R: L, Y: Rt, B: C}),
    ("Drop scarlet stuff in the left bin, navy stuff in the middle one, and the yellow stuff into the right one.",
     {R: L, B: C, Y: Rt}),
    ("The center bin takes blue. The left bin takes yellow. Red goes to the last bin.", {B: C, Y: L, R: Rt}),
    ("Blue left. Red center. Yellow right.", {B: L, R: C, Y: Rt}),
    ("I want the red objects on the right side, and the blue and yellow ones... no wait: blue in the middle, "
     "yellow on the left.", {R: Rt, B: C, Y: L}),
    ("Put blue and red in the left bin and yellow in the right bin.", {B: L, R: L, Y: Rt}),
    ("The left bin is for red and blue, the right bin for yellow.", {R: L, B: L, Y: Rt}),
    ("Red to the left. Do not put blue in the center bin. Yellow goes to the center.", {R: L, B: Rt, Y: C}),
    ("Sort them: red ➜ right, blue ➜ center, yellow ➜ left.".replace("➜", "->"), {R: Rt, B: C, Y: L}),
    ("Place the yellow objects into the bin on the left, put the blue ones into the right one, and red goes in "
     "between.", {Y: L, B: Rt, R: C}),
    ("Red objects: center bin. Blue objects: left bin. Yellow objects: right bin.", {R: C, B: L, Y: Rt}),
    ("Sort the objects into the bins: blue right, red middle, yellow left.", {B: Rt, R: C, Y: L}),
    ("Every blue object goes to the right; every red object goes to the left; the yellow objects go into the "
     "middle.", {B: Rt, R: L, Y: C}),
    ("Put red objects in the bin to the left, and blue objects in the right bin. The yellow ones go to the center "
     "bin.", {R: L, B: Rt, Y: C}),
    ("Left gets blue, centre gets red, right gets yellow.", {B: L, R: C, Y: Rt}),
    ("All red items to the right box. All yellow items to the left box. All blue items to the middle box.",
     {R: Rt, Y: L, B: C}),
    ("Put the red ones on your left, the blue ones on your right and yellow ones in the middle.", {R: L, B: Rt, Y: C}),
    ("The yellow objects go into the right bin, but the red objects should not go into the left bin.",
     {Y: Rt, R: C, B: L}),
]


class TestTemplates(unittest.TestCase):
    def test_every_training_template_and_permutation(self):
        n = 0
        for tpl in tasks.TEMPLATES:
            for cols in itertools.permutations(tasks.COLORS):
                for bins in itertools.permutations(tasks.BINS):
                    w = {f"c{i}": c for i, c in enumerate(cols)}
                    w.update({f"C{i}": c.capitalize() for i, c in enumerate(cols)})
                    w.update({f"b{i}": b for i, b in enumerate(bins)})
                    text = tpl.format(**w)
                    r = parse(text)
                    self.assertEqual(r.mapping, dict(zip(cols, bins)), f"{text!r}: {r.notes}")
                    self.assertTrue(r.ok)
                    n += 1
        self.assertEqual(n, 8 * 36)

    def test_fixed(self):
        self.assertEqual(parse(tasks.FIXED_INSTRUCTION).mapping, tasks.FIXED_MAPPING)
        self.assertEqual(parse("", task="sort_fixed").mapping, tasks.FIXED_MAPPING)

    def test_make_instruction_roundtrip(self):
        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(500):
            text, m = tasks.make_instruction("sort_language", rng)
            self.assertEqual(parse(text).mapping, m, text)


class TestUnseen(unittest.TestCase):
    def test_paraphrases(self):
        bad = []
        for text, exp in UNSEEN:
            r = parse(text)
            got = {k: r.mapping.get(k) for k in exp}
            if got != exp:
                bad.append(f"{text!r}\n   want {exp}\n   got  {r.mapping}  notes={r.notes}")
        self.assertFalse(bad, "\n" + "\n".join(bad))

    def test_shape_resolution(self):
        from aaltonomous.scene.scene import resolve_bin
        m = parse("Put the red cubes in the right bin and everything else in the left bin.").mapping
        self.assertEqual(resolve_bin(m, "red", "cube"), "right")
        self.assertEqual(resolve_bin(m, "red", "cylinder"), "left")
        self.assertEqual(resolve_bin(m, "blue", "cube"), "left")

    def test_garbage_is_not_ok(self):
        self.assertFalse(parse("Do a little dance.").ok)
        self.assertFalse(parse("Sort the objects.").ok)


if __name__ == "__main__":
    unittest.main()
