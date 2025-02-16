import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN1D(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes=4):
        super(FPN1D, self).__init__()

        # Bottom-Up Pathway
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)  # Downsample (23 → 12)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)  # Downsample (12 → 6)

        # Lateral Connections
        self.lateral2 = nn.Conv1d(128, 128, kernel_size=1)
        self.lateral1 = nn.Conv1d(64, 64, kernel_size=1)

        # Top-Down Pathway (Interpolation-based Upsampling)
        self.up2 = nn.Conv1d(256, 128, kernel_size=1)  # Reduce channels
        self.up1 = nn.Conv1d(128, 64, kernel_size=1)   # Reduce channels

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(64 * sequence_length, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch, channels, time)

        # Bottom-Up Pathway
        c1 = F.relu(self.conv1(x))  # Shape: (batch, 64, 23)
        c2 = F.relu(self.conv2(c1))  # Shape: (batch, 128, 12)
        c3 = F.relu(self.conv3(c2))  # Shape: (batch, 256, 6)

        # Top-Down Pathway with Lateral Connections
        p2 = self.up2(c3)  # Reduce channels (batch, 128, 6)
        p2 = F.interpolate(p2, size=c2.shape[2], mode="linear", align_corners=False)  # Match c2 length (12)
        p2 = F.relu(p2 + self.lateral2(c2))  

        p1 = self.up1(p2)  # Reduce channels (batch, 64, 12)
        p1 = F.interpolate(p1, size=c1.shape[2], mode="linear", align_corners=False)  # Match c1 length (23)
        p1 = F.relu(p1 + self.lateral1(c1))  

        # Flatten and Classification Layer
        r1 = p1.view(p1.shape[0], -1)  # (batch, 64 * 23)
        features = self.fc(p1)
        output = self.out(features)

        return output

class FPN1D_V1(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes=4):
        super(FPN1D_V1, self).__init__()

        # Bottom-Up Pathway
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)  # Downsample (23 → 12)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)  # Downsample (12 → 6)

        # Lateral Connections
        self.lateral2 = nn.Conv1d(128, 128, kernel_size=1)
        self.lateral1 = nn.Conv1d(64, 64, kernel_size=1)

        # Top-Down Pathway
        self.up2 = nn.Conv1d(256, 128, kernel_size=1)  
        self.up1 = nn.Conv1d(128, 64, kernel_size=1)   

        # Feature and Classification Layers
        self.fc = nn.Linear(64 * sequence_length, 128)  # Feature layer
        self.out = nn.Linear(128, num_classes)  # Classification layer

    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)  # Reshape to (batch, channels, time)

        # Bottom-Up Pathway
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))

        # Top-Down Pathway
        p2 = self.up2(c3)
        p2 = F.interpolate(p2, size=c2.shape[2], mode="linear", align_corners=False)
        p2 = F.relu(p2 + self.lateral2(c2))

        p1 = self.up1(p2)
        p1 = F.interpolate(p1, size=c1.shape[2], mode="linear", align_corners=False)
        p1 = F.relu(p1 + self.lateral1(c1))

        # Feature Extraction
        r1 = p1.view(p1.shape[0], -1)  # Flatten features
        features = self.fc(r1)  # Features before classification

        if return_features:
            return features  # Return feature values instead of class output

        output = self.out(features)  # Class predictions
        return output