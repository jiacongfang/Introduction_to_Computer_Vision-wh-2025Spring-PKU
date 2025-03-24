import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":
    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")
    print("Noise points shape: ", noise_points.shape)

    # RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0

    # Calculate the minimal sample time first
    # the minimal time that can guarantee the probability of at least one hypothesis
    # does not contain any outliers is larger than 99.9%
    num_inliers = 100
    p, w, n = 0.999, num_inliers / noise_points.shape[0], 3
    sample_time = np.ceil(np.log(1 - p) / np.log(1 - w**n)).astype(int)
    print("Minimal sample time: ", sample_time)

    distance_threshold = 0.05

    # sample points group: (k, n) from total points
    random_index = np.random.choice(
        len(noise_points), (sample_time, n), replace=False
    )  # (sample_time, n)
    random_index_flat = random_index.flatten()  # (sample_time*n,)

    sample_points = noise_points[random_index_flat].reshape(
        sample_time, n, -1
    )  # (sample_time, n, 3)

    # estimate the plane with sampled points group
    v1 = sample_points[:, 1, :] - sample_points[:, 0, :]  # [sample_time, 3]
    v2 = sample_points[:, 2, :] - sample_points[:, 0, :]  # [sample_time, 3]
    normal_vectors = np.cross(v1, v2, axis=1)
    A, B, C = normal_vectors[:, 0], normal_vectors[:, 1], normal_vectors[:, 2]
    D = -(
        A * sample_points[:, 0, 0]
        + B * sample_points[:, 0, 1]
        + C * sample_points[:, 0, 2]
    )

    pf = np.stack([A, B, C, D], axis=1)  # (sample_time, 4)
    print("Plane function shape: ", pf.shape)

    # normalize the normal vectors
    normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=1)[:, None]

    # evaluate inliers (with point-to-plance distance < distance_threshold)
    vec = (
        noise_points[:, None, :] - sample_points[None, :, 0, :]
    )  # (130, sample_time, 3)
    distance = np.abs(np.sum(vec * normal_vectors[None, :, :], axis=2))
    inliers = distance < distance_threshold
    print("Inliers shape: ", inliers.shape)

    # count the number of inliers for each hypothesis
    inliers_count = np.sum(inliers, axis=0)  # (sample_time,)
    best_hypothesis_index = np.argmax(inliers_count)
    inliers_best_index = np.where(inliers[:, best_hypothesis_index])[0]

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    inliers_best = noise_points[inliers_best_index]
    v1 = inliers_best[1] - inliers_best[0]
    v2 = inliers_best[2] - inliers_best[0]
    normal_vector = np.cross(v1, v2)
    A, B, C = normal_vector
    D = -np.sum(normal_vector * inliers_best[0])
    pf = np.array([A, B, C, D])

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt("result/HM1_RANSAC_sample_time.txt", np.array([sample_time]))
