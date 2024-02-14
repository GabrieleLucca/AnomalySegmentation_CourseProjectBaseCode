'''imported from dlmacedo/entropic-out-of-distribution-detection repository'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class IsoMaxPlusLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def init(self, num_features, num_classes):
        super(IsoMaxPlusLossFirstPart, self).init()
        self.num_features = num_features
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features), F.normalize(self.prototypes),
            p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        return logits

class IsoMaxPlusLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def init(self, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).init()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets):
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        return loss