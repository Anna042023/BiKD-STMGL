import torch


def mask_lane_noise_speeds(lane_speed_vector, road_speed_vector, lane_counts, k):
    """
    For each road, find corresponding lanes and mask the lane speeds that are not within a threshold k of the road speed.
    Lane speeds within the threshold remain unchanged, others are masked.

    Parameters:
    - lane_speed_vector: torch.Tensor with shape (batch_size, seq_len, num_lane_node)
    - road_speed_vector: torch.Tensor with shape (batch_size, seq_len, num_road_node)
    - lane_counts: list of integers, length num_road_node, indicating the number of lanes per road.
    - k: float, the threshold for determining if a lane speed is close enough to the road speed.

    Returns:
    - masked_lane_speed_vector: torch.Tensor, the lane_speed_vector with speeds not within k of the road speed masked.
    """
    batch_size, seq_len, num_lane_node = lane_speed_vector.shape
    num_road_node = len (lane_counts)

    # Create a clone of the lane_speed_vector to perform masking operations
    masked_lane_speed_vector = lane_speed_vector.clone ()

    # Compute cumulative sum of lane_counts to find indices ranges for lanes per road
    lane_indices = torch.cumsum (torch.tensor ([0] + lane_counts), dim=0)

    for road_idx in range (num_road_node):
        start_idx, end_idx = lane_indices[road_idx].item (), lane_indices[road_idx + 1].item ()

        # Extract the lane speeds for the current road across all batches and sequences
        current_lanes_speeds = lane_speed_vector[:, :, start_idx:end_idx]

        # Compute the condition for masking: where the absolute difference with road speed is greater than k
        road_speed_expanded = road_speed_vector[:, :, road_idx].unsqueeze (-1).expand (-1, -1, end_idx - start_idx)
        mask_condition = torch.abs (current_lanes_speeds - road_speed_expanded) > k

        # Apply mask: set speeds not within k of the road speed to a specific value (e.g., 0)
        masked_lane_speed_vector[:, :, start_idx:end_idx][mask_condition] = 0  # Masking with 0

    return masked_lane_speed_vector


def random_mask_lane_speeds(lane_speed_vector, mask_percentage):
    """
    Randomly mask a specified percentage of the lane speed vector.

    Parameters:
    - lane_speed_vector: torch.Tensor with shape (batch_size, seq_len, num_lane_node)
    - mask_percentage: float, the percentage of elements to be masked in the lane speed vector.

    Returns:
    - masked_lane_speed_vector: torch.Tensor, the lane_speed_vector with specified percentage of elements randomly masked.
    """
    # Generate a random tensor with the same shape as lane_speed_vector
    random_tensor = torch.rand_like (lane_speed_vector)

    # Determine the threshold for masking based on mask_percentage
    mask_threshold = mask_percentage / 100.0

    # Apply mask: set elements to 0 where the corresponding element in random_tensor is less than the threshold
    masked_lane_speed_vector = lane_speed_vector.clone ()
    masked_lane_speed_vector[random_tensor < mask_threshold] = 0

    return masked_lane_speed_vector


def adjust_lane_speeds_torch(lane_speed_vector, road_speed_vector, lane_counts, k):
    """
    For each road, find corresponding lanes. If a lane's speed is within a threshold k of the road speed,
    it remains unchanged. Otherwise, set the lane's speed to the corresponding road speed.

    Parameters:
    - lane_speed_vector: torch.Tensor with shape (batch_size, seq_len, num_lane_node)
    - road_speed_vector: torch.Tensor with shape (batch_size, seq_len, num_road_node)
    - lane_counts: list of integers, length num_road_node, indicating the number of lanes per road.
    - k: float, the threshold for determining if a lane speed is close enough to the road speed.

    Returns:
    - adjusted_lane_speed_vector: torch.Tensor, the lane_speed_vector with speeds not within k of the road speed adjusted.
    """
    batch_size, seq_len, num_lane_node = lane_speed_vector.shape
    num_road_node = len (lane_counts)

    # Create a clone of the lane_speed_vector to perform adjustments
    adjusted_lane_speed_vector = lane_speed_vector.clone ()

    # Compute cumulative sum of lane_counts to find indices ranges for lanes per road
    lane_indices = torch.cumsum (torch.tensor ([0] + lane_counts), dim=0)

    for road_idx in range (num_road_node):
        start_idx, end_idx = lane_indices[road_idx].item (), lane_indices[road_idx + 1].item ()

        # Extract the lane speeds for the current road across all batches and sequences
        current_lanes_speeds = lane_speed_vector[:, :, start_idx:end_idx]

        # Compute the condition for adjustment: where the absolute difference with road speed is greater than k
        road_speed_expanded = road_speed_vector[:, :, road_idx].unsqueeze (-1).expand (-1, -1, end_idx - start_idx)
        adjustment_condition = torch.abs (current_lanes_speeds - road_speed_expanded) > k

        # Apply adjustment: set speeds not within k of the road speed to the corresponding road speed
        adjusted_lane_speed_vector[:, :, start_idx:end_idx][adjustment_condition] = road_speed_expanded[
            adjustment_condition]

    return adjusted_lane_speed_vector