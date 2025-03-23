import os
import unittest
import numpy as np
import open3d as o3d
import logging


class TestRANSAC(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - [%(message)s]",
        )
        logging.info(f"Running test: {self._testMethodName}")

    def test_ransac(self):
        noise_points_path = os.path.join(".", "HM1_ransac_points.txt")
        pred_plane_path = os.path.join(".", "result", "HM1_RANSAC_plane.txt")

        if not os.path.exists(noise_points_path):
            raise FileNotFoundError(f"File not found: {noise_points_path}")
        if not os.path.exists(pred_plane_path):
            raise FileNotFoundError(f"File not found: {pred_plane_path}")

        noise_points = np.loadtxt(noise_points_path)
        pred_plane = np.loadtxt(pred_plane_path)

        np.random.seed(0)

        p, w, n = 0.999, 10 / 13, 3
        sample_time = np.ceil(np.log(1 - p) / np.log(1 - w**n)).astype(int)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(noise_points)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=sample_time
        )

        # Normalize the plane model by the last element
        plane_model = np.array(plane_model) / np.array(plane_model)[-1]
        logging.info(f"Plane model estimated by open3d: {plane_model}")

        pred_plane = pred_plane / pred_plane[-1]
        logging.info(f"Predicted plane: {pred_plane}")
        logging.info(f"Diff between open3d and prediction: {plane_model - pred_plane}")

        np.testing.assert_allclose(plane_model, pred_plane, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
