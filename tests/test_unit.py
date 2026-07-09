import unittest
from posture.correction_engine import CorrectionEngine


class DummyPerson:
    def __init__(
        self, pitch=0.0, spine=0.0, shoulder=0.0, ratio=0.5, baseline=0.5, cal_pitch=0.0
    ):
        self.pitch = pitch
        self.spine_alignment = spine
        self.shoulder_alignment = shoulder
        self.smoothed_ratio = ratio
        self.posture_baseline = baseline
        self.calibrated_baseline_neck_pitch = cal_pitch


class TestCorrectionEngine(unittest.TestCase):
    def test_forward_lean(self):
        # Torso ratio < 80% of baseline
        person = DummyPerson(ratio=0.35, baseline=0.5)
        advice = CorrectionEngine.get_advice(person)
        self.assertEqual(advice, "Avoid leaning forward.")

    def test_spine_deviation(self):
        # abs(spine_alignment) > 20
        person = DummyPerson(spine=25.0)
        advice = CorrectionEngine.get_advice(person)
        self.assertEqual(advice, "Straighten your back.")

    def test_neck_pitch(self):
        # pitch - calibrated > 35
        person = DummyPerson(pitch=40.0, cal_pitch=0.0)
        advice = CorrectionEngine.get_advice(person)
        self.assertEqual(advice, "Lift your head.")

    def test_shoulder_roll(self):
        # abs(shoulder) > 15
        person = DummyPerson(shoulder=-16.0)
        advice = CorrectionEngine.get_advice(person)
        self.assertEqual(advice, "Relax your shoulders.")

    def test_ergonomic_fallback(self):
        person = DummyPerson()  # defaults to perfect posture
        advice = CorrectionEngine.get_advice(person)
        self.assertEqual(advice, "Maintain ergonomic posture.")


if __name__ == "__main__":
    unittest.main()
