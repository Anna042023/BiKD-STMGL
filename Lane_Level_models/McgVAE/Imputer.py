import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneSpeedImputer (nn.Module):
    def __init__(self, seq_len, num_lane_nodes, num_road_nodes, hidden_dim):
        super (LaneSpeedImputer, self).__init__ ()
        self.seq_len = seq_len
        self.num_lane_nodes = num_lane_nodes
        self.num_road_nodes = num_road_nodes
        self.hidden_dim = hidden_dim

        # Define the network layers
        self.fc1 = nn.Linear (seq_len * (num_lane_nodes + num_road_nodes), hidden_dim)
        self.fc2 = nn.Linear (hidden_dim, seq_len * num_lane_nodes)

    def forward(self, lane_speed_vector, road_speed_vector, lane_counts):
        # Preprocess input: Assuming some method to appropriately combine lane and road speeds
        # This example directly concatenates them for simplicity
        combined_input = torch.cat ((lane_speed_vector, road_speed_vector), dim=2)
        combined_input_flattened = combined_input.view (combined_input.size (0), -1)

        # Network forward pass
        hidden = F.relu (self.fc1 (combined_input_flattened))
        output = self.fc2 (hidden).view (-1, self.seq_len, self.num_lane_nodes)

        # Post-processing: Only update masked (zero) values in the original lane_speed_vector
        # Create a mask for positions where lane_speed_vector is zero
        mask = lane_speed_vector == 0
        # Update these positions with the network's output
        adjusted_lane_speed_vector = torch.where (mask, output, lane_speed_vector)

        return adjusted_lane_speed_vector