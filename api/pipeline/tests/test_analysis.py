import unittest
from app.analysis import (
    apply_tukey_outlier_detection_frame,
    update_plant_cleaning_state,
    MORPHOLOGY_FEATURES,
)


class TestAnalysis(unittest.TestCase):
    def test_apply_tukey_outlier_detection(self):
        # Create a list of stats mimicking a frame
        # Normal areas: 100, 105, 95, 102
        # Outlier: 1000 (masked pot)

        # We need at least 4 points for our logic to kick in
        stats_list = [
            {"area": 100.0, "id": 1},
            {"area": 105.0, "id": 2},
            {"area": 95.0, "id": 3},
            {"area": 102.0, "id": 4},
            {"area": 1000.0, "id": 5},  # Massive outlier
        ]

        result = apply_tukey_outlier_detection_frame(stats_list)

        # Check normal
        self.assertFalse(result[0]["tukey_outlier"])  # 100
        self.assertEqual(result[0]["area_after_tukey"], 100.0)

        # Check outlier
        self.assertTrue(result[4]["tukey_outlier"])  # 1000
        self.assertIsNone(result[4]["area_after_tukey"])

    def test_ewm_cleaning_flow(self):
        # Simulate a single plant over time
        # Sequence of areas: 100, 102, 105, 500 (outlier), 108
        areas = [100.0, 102.0, 105.0, 500.0, 108.0]

        # Assume valid other features for simplicity
        base_features = {f: 1.0 for f in MORPHOLOGY_FEATURES}

        state = None

        results = []

        for i, area in enumerate(areas):
            # 1. Create stats input (mimicking output of Tukey, so area_after_tukey is set)
            # If 500 is outlier, let's say it WASN'T caught by Tukey to test EWM
            # or it WAS caught.
            # Let's test EWM specifically: so assume it passed Tukey or is within-plant outlier.
            # 500 vs ~100 is definitely an EWM outlier.

            stats = base_features.copy()
            stats["area"] = area
            stats["area_after_tukey"] = area  # Assume passed Tukey

            # Simulate mask object
            current_mask = {"id": i, "content": "mask"}

            new_state, clean_stats, clean_mask = update_plant_cleaning_state(
                stats, state, current_mask
            )
            state = new_state

            # Store clean_mask in result for checking
            clean_stats["returned_mask"] = clean_mask
            results.append(clean_stats)

            # Additional check: modifying stats in place is how utils.py does it,
            # but here we check the return

        # Step 0: 100. First value.
        self.assertFalse(results[0]["is_outlier"])
        self.assertAlmostEqual(results[0]["clean_area"], 100.0)
        self.assertEqual(results[0]["returned_mask"]["id"], 0)

        # Step 1: 102. Normal.
        self.assertFalse(results[1]["is_outlier"])
        self.assertAlmostEqual(results[1]["clean_area"], 102.0)

        # Step 2: 105. Normal.
        self.assertFalse(results[2]["is_outlier"])
        self.assertAlmostEqual(results[2]["clean_area"], 105.0)

        # Step 3: 500. Outlier.
        # Should be flagged.
        # CLEAN_AREA_UPPER_THRESHOLD = 1.5.
        # Mean is around 105. 1.5 * 105 = 157 (plus mean).
        # Wait, (1+1.5)*mean = 2.5*mean ~ 260. 500 > 260.
        self.assertTrue(results[3]["is_outlier"])
        # Should use previous clean value (105 from step 2)
        self.assertAlmostEqual(results[3]["clean_area"], 105.0)
        # Should return mask from step 2 (id=2)
        self.assertEqual(results[3]["returned_mask"]["id"], 2)

        # Step 4: 108. Normal.
        self.assertFalse(results[4]["is_outlier"])
        self.assertAlmostEqual(results[4]["clean_area"], 108.0)
        self.assertEqual(results[4]["returned_mask"]["id"], 4)


if __name__ == "__main__":
    unittest.main()
