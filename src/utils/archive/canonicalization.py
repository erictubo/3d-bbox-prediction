import torch
import math


def quat_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiplication in [w, x, y, z] format.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    cw = aw * bw - ax * bx - ay * by - az * bz
    cx = aw * bx + ax * bw + ay * bz - az * by
    cy = aw * by - ax * bz + ay * bw + az * bx
    cz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([cw, cx, cy, cz], dim=-1)

def canonicalize_params(params: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize to make params unique for loss computation.
    Sorts dims descending, adjusts quat for new order, ensures w > 0.
    Handles batch (B, 10).
    """
    B = params.size(0)
    center = params[:, :3]
    dims = params[:, 3:6]
    quat = params[:, 6:]

    # Normalize quat (input format: [x, y, z, w])
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-6)

    # Sort dims descending and get permutation indices
    sorted_dims, perm_idx = dims.sort(dim=1, descending=True)  # (B, 3), (B, 3)

    # Convert input quat to standard [w, x, y, z] format
    x, y, z, w = quat.unbind(-1)
    quat_standard = torch.stack([w, x, y, z], dim=-1)  # (B, 4)

    # Define the 6 possible permutations and corresponding s quaternions ([w, x, y, z])
    perms = [
        [0, 1, 2],  # a
        [0, 2, 1],  # b
        [1, 0, 2],  # c
        [2, 1, 0],  # d
        [1, 2, 0],  # e
        [2, 0, 1],  # f
    ]
    sqrt2_2 = math.sqrt(2) / 2
    s_list = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),          # a
        torch.tensor([sqrt2_2, sqrt2_2, 0.0, 0.0]),  # b
        torch.tensor([sqrt2_2, 0.0, 0.0, sqrt2_2]),  # c
        torch.tensor([sqrt2_2, 0.0, sqrt2_2, 0.0]),  # d
        torch.tensor([-0.5, 0.5, 0.5, 0.5]),         # e
        torch.tensor([0.5, 0.5, 0.5, 0.5]),          # f
    ]

    # Assign s for each batch element based on perm_idx
    s = torch.zeros((B, 4), dtype=torch.float, device=params.device)
    for i, perm in enumerate(perms):
        perm_t = torch.tensor(perm, dtype=torch.long, device=perm_idx.device)
        mask = torch.all(perm_idx == perm_t, dim=1)
        s[mask] = s_list[i].to(params.device)

    # Compute s_inv (conjugate of s, since unit quaternion)
    s_inv = torch.cat([s[:, :1], -s[:, 1:]], dim=1)  # [w, -x, -y, -z]

    # Compute canonical_quat_standard = s * quat_standard * s_inv
    temp = quat_mult(quat_standard, s_inv)
    canonical_quat_standard = quat_mult(s, temp)

    # Ensure w >= 0 (flip sign if necessary)
    sign = torch.sign(canonical_quat_standard[:, 0])
    sign[sign == 0] = 1.0  # Arbitrary choice if w == 0
    canonical_quat_standard = canonical_quat_standard * sign.unsqueeze(1)

    # Convert back to [x, y, z, w] format
    canonical_quat = canonical_quat_standard[:, [1, 2, 3, 0]]

    return torch.cat([center, sorted_dims, canonical_quat], dim=1)




def test_canonicalize_params():
    """
    Test function for canonicalize_params to ensure correct sorting of dimensions
    and quaternion adjustment for axis permutations.
    """

    # Helper function to rotate a vector using a quaternion ([w, x, y, z] format)
    def rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # Quaternion must be in [w, x, y, z]
        quat_conj = torch.cat([quat[:, :1], -quat[:, 1:]], dim=-1)
        vec_quat = torch.cat([torch.zeros_like(vec[:, :1]), vec], dim=-1)
        temp = quat_mult(quat, vec_quat)
        result = quat_mult(temp, quat_conj)
        return result[:, 1:]

    print("Running tests for canonicalize_params...")

    # Test setup: Create a batch of 6 examples, one for each permutation
    B = 6
    center = torch.zeros(B, 3)  # Center at origin for simplicity
    # Dimensions for each permutation (to be sorted to [3, 2, 1])
    dims_list = [
        torch.tensor([3.0, 2.0, 1.0]),  # already sorted: [0,1,2]
        torch.tensor([3.0, 1.0, 2.0]),  # to [0,2,1]
        torch.tensor([2.0, 3.0, 1.0]),  # to [1,0,2]
        torch.tensor([1.0, 2.0, 3.0]),  # to [2,1,0]
        torch.tensor([2.0, 1.0, 3.0]),  # to [1,2,0]
        torch.tensor([1.0, 3.0, 2.0]),  # to [2,0,1]
    ]
    dims = torch.stack(dims_list, dim=0)  # (B, 3)

    # Quaternion: Use a simple rotation (e.g., 90 degrees around x-axis)
    # Input format [x, y, z, w]
    quat_raw = torch.tensor([math.sqrt(2)/2, 0.0, 0.0, math.sqrt(2)/2])  # 90 deg around x
    quat = quat_raw.repeat(B, 1)  # Same quaternion for all

    # Input tensor: (B, 10) = [center(3), dims(3), quat(4)]
    params = torch.cat([center, dims, quat], dim=1)

    # Run the canonicalization
    result = canonicalize_params(params)
    result_center = result[:, :3]
    result_dims = result[:, 3:6]
    result_quat = result[:, 6:]  # [x, y, z, w]

    # Convert result_quat to [w, x, y, z] for rotation test
    result_quat_standard = torch.stack([result_quat[:, 3], result_quat[:, 0], result_quat[:, 1], result_quat[:, 2]], dim=-1)

    # Expected sorted dimensions
    expected_dims = torch.tensor([3.0, 2.0, 1.0]).repeat(B, 1)

    # Test 1: Check if center is unchanged
    assert torch.allclose(result_center, center, atol=1e-5), "Center should remain unchanged"

    # Test 2: Check if dimensions are sorted descending
    assert torch.allclose(result_dims, expected_dims, atol=1e-5), "Dimensions should be sorted descending [3,2,1]"

    # Test 3: Check if quaternion is normalized
    assert torch.allclose(torch.norm(result_quat, dim=-1), torch.ones(B), atol=1e-5), "Quaternion should be normalized"

    # Test 4: Check if w >= 0
    assert torch.all(result_quat[:, 3] >= 0), "Quaternion scalar part w should be >= 0"

    # Test 5: Check if the physical orientation is preserved (rotate a test vector)
    # Original rotation: 90 deg around x, should map (0,1,0) to (0,0,1)
    test_vec = torch.tensor([0.0, 1.0, 0.0]).repeat(B, 1)
    orig_quat_standard = torch.tensor([math.sqrt(2)/2, math.sqrt(2)/2, 0.0, 0.0]).repeat(B, 1)  # [w,x,y,z]
    expected_rotated = torch.tensor([0.0, 0.0, 1.0]).repeat(B, 1)
    # Since axes are permuted, we need to account for the permutation in the expected vector
    perms = [
        [0, 1, 2],  # no change
        [0, 2, 1],  # y->z, z->y
        [1, 0, 2],  # x->y, y->x
        [2, 1, 0],  # x->z, y->x, z->y
        [1, 2, 0],  # x->y, y->z, z->x
        [2, 0, 1],  # x->z, y->x, z->y
    ]
    expected_rotated_permuted = torch.zeros(B, 3)
    for i, perm in enumerate(perms):
        expected_rotated_permuted[i, perm[0]] = expected_rotated[i, 0]
        expected_rotated_permuted[i, perm[1]] = expected_rotated[i, 1]
        expected_rotated_permuted[i, perm[2]] = expected_rotated[i, 2]

    rotated_vec = rotate_vector(result_quat_standard, test_vec)

    for i in range(B):
        if not torch.allclose(rotated_vec, expected_rotated_permuted, atol=1e-4):
            print(f"Test failed for permutation {i}")
            print(f"rotated_vec: {rotated_vec}")
            print(f"expected_rotated_permuted: {expected_rotated_permuted}")
            print(f"result_quat_standard: {result_quat_standard}")
            print(f"test_vec: {test_vec}")
            print(f"quat_raw: {quat_raw}")

        else:
            print(f"Test passed for permutation {i}")

# Run the test
if __name__ == "__main__":
    test_canonicalize_params()
