"""
Unit tests for Method M5: Reliable Identity Transplant (face_swap.py).
Uses Python standard library unittest.
"""
import unittest
from consistency_v2.face_swap import ReActorPayloadBuilder, DirectFaceSwapper


class TestFaceSwap(unittest.TestCase):

    def test_reactor_payload_31_arguments(self):
        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        payload = ReActorPayloadBuilder.build_reactor_payload(
            source_image_b64=fake_b64,
            target_face_index=0,
            codeformer_weight=0.65,
            restorer_visibility=0.80
        )

        self.assertIn("reactor", payload)
        args = payload["reactor"]["args"]

        # Exactly 31 arguments required by Forge API contract
        self.assertEqual(len(args), 31)

        # Verify resolution of Defect D2
        self.assertEqual(args[0], fake_b64)
        self.assertIs(args[1], True)               # Enable
        self.assertEqual(args[4], "inswapper_128.onnx")
        self.assertEqual(args[5], "CodeFormer")
        self.assertEqual(args[6], 0.80)            # Visibility
        self.assertEqual(args[13], "Minimum")      # String, not int 0!
        self.assertEqual(args[14], "No")           # String, not int 0!
        self.assertEqual(args[15], "No")           # String, not int 0!
        self.assertEqual(args[17], 0.65)           # CodeFormer fidelity
        self.assertEqual(args[22], "Image(s)")     # String, not int 0!
        self.assertEqual(args[30], "tab_single")   # Mandatory 31st argument!

    def test_direct_swapper_validates_missing_file(self):
        res = DirectFaceSwapper.execute_face_swap(
            target_image_path="non_existent_image.png",
            swap_specs=[],
            out_image_path="out.png"
        )
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
