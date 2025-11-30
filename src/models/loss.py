"""
Multi-task loss function that uses cross-entropy for Caption Prediction
and binary cross-entropy with logits for Concept Detection.
"""

import torch
import torch.nn as nn


class UnifiedVLMLoss(nn.Module):
    def __init__(self, caption_weight=1.0, concept_weight=1.0, concept_pos_weight=None):
        """
        Multi-task loss module for simultaneous Caption Generation and Concept Detection.

        This loss function is designed for a UNIFIED model that outputs both:
        1. Language modeling logits (lm_logits)
        2. Concept classification logits (concept_logits)

        Args:
            caption_weight (float): Weight for the language modeling loss (alpha).
            concept_weight (float): Weight for the concept classification loss (beta).
            concept_pos_weight (torch.Tensor): Optional weights for positive classes in BCE
                                               (useful for imbalanced concept tags).
        """
        super(UnifiedVLMLoss, self).__init__()
        self.caption_weight = caption_weight
        self.concept_weight = concept_weight

        # Standard Language Modeling Loss (Cross Entropy)
        # ignore_index=-100 allows us to mask out padding tokens in labels
        self.caption_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Multi-label Classification Loss (Binary Cross Entropy with Logits)
        self.concept_loss_fn = nn.BCEWithLogitsLoss(pos_weight=concept_pos_weight)

    def forward(self, lm_logits, concept_logits, lm_labels, concept_labels):
        """
        Computes the weighted sum of captioning loss (Causal LM) and concept loss (Classification).

        Args:
            lm_logits: [batch_size, seq_len, vocab_size] - Output from LLM head
            concept_logits: [batch_size, num_concepts] - Output from Visual Grounding head
            lm_labels: [batch_size, seq_len] - Token IDs for ground truth caption (labels in the collator)
            concept_labels: [batch_size, num_concepts] - Binary (0/1) matrix for concepts

        Returns:
            total_loss: Weighted sum of both losses
            loss_dict: Dictionary containing individual loss components for logging
        """
        # 1. Compute Caption Loss (Standard Causal LM Loss)
        # Shift tokens: predict the next token based on the current one
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()

        loss_cap = self.caption_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # 2. Compute Concept Loss (Multi-label Classification)
        loss_con = self.concept_loss_fn(concept_logits, concept_labels.float())

        # 3. Combine for Total Loss
        total_loss = (self.caption_weight * loss_cap) + (self.concept_weight * loss_con)

        return total_loss, {
            "loss_caption": loss_cap.item(),
            "loss_concept": loss_con.item(),
            "loss_total": total_loss.item()
        }