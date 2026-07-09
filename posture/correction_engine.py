class CorrectionEngine:
    """
    Stateless evaluation engine that translates geometric deviations into contextual TTS string alerts.
    Analyzes advanced biometric variables extracted by PostureAnalyzer.
    """

    @staticmethod
    def get_advice(person) -> str:
        """
        Calculates severity and assigns the proper alert string.
        """
        from config.settings import settings

        torso_ratio = getattr(person, "smoothed_ratio", 0.5)
        baseline_ratio = getattr(person, "posture_baseline", 0.5)
        spine_alignment = getattr(person, "spine_alignment", 0.0)
        shoulder_alignment = getattr(person, "shoulder_alignment", 0.0)

        # 1. Forward Slouch Depth
        if torso_ratio < (baseline_ratio * settings.slouch_ratio_threshold):
            return "Avoid leaning forward."

        # 2. Lateral Spine Deviation
        if abs(spine_alignment) > settings.spine_tolerance_degrees:
            return "Straighten your back."

        # 3. Severe Neck Pitch
        cal_pitch = getattr(person, "calibrated_baseline_neck_pitch", 0.0)
        relative_slouch = getattr(person, "pitch", 0.0) - cal_pitch
        if relative_slouch > settings.neck_pitch_tolerance_degrees:
            return "Lift your head."

        # 4. Shoulder Roll/Asymmetry
        if abs(shoulder_alignment) > settings.shoulder_roll_tolerance_degrees:
            return "Relax your shoulders."

        return "Maintain ergonomic posture."
