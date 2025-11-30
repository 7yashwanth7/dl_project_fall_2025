"""
Computes evaluation metrics for medical VLM
Concept metrics: F1, Precision, Recall (micro/macro)
Caption metrics: ROUGE, BERTScore, MedCLIP
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import CLIPProcessor, CLIPModel


class MedicalEvaluator:
    def __init__(self, device='cuda'):
        self.device = device

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Initialize MedCLIP (using a popular huggingface PubMed CLIP implementation)
        try:
            self.medclip_model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32").to(
                self.device)
            self.medclip_processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            self.use_medclip = True
        except Exception as e:
            print(f"Warning: MedCLIP model could not be loaded. MedCLIP scores will be 0. Error: {e}")
            self.use_medclip = False

    def compute_concept_metrics(self, y_true, y_pred, concept_names=None):
        """
        Computes F1, Precision, and Recall for multi-label concept detection.

        Args:
            y_true (np.array): Binary ground truth matrix [N, num_classes]
            y_pred (np.array): Binary prediction matrix [N, num_classes] (thresholded logits)
            concept_names (list): List of string names for the concepts (CUIs)

        Returns:
            dict: Overall metrics and per-class performance.
        """
        # Overall Micro/Macro metrics
        metrics = {
            "concept_f1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
            "concept_f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "concept_precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
            "concept_recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        }

        # Per-class F1 for finding "Top 5 / Worst 5"
        # We compute F1 for each label individually
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        if concept_names:
            class_performance = []
            for idx, score in enumerate(per_class_f1):
                # Only track classes that actually appeared in the ground truth to avoid noise
                if np.sum(y_true[:, idx]) > 0:
                    class_performance.append((concept_names[idx], score))

            # Sort by F1 score
            class_performance.sort(key=lambda x: x[1], reverse=True)
            metrics["top_5_concepts"] = class_performance[:5]
            metrics["worst_5_concepts"] = class_performance[-5:]

        return metrics

    def compute_caption_metrics(self, references, predictions, images=None):
        """
        Computes ROUGE, BERTScore, and MedCLIP score.

        Args:
            references (list of str): Ground truth captions.
            predictions (list of str): Generated captions.
            images (list of PIL.Image): Original images (needed for MedCLIP).
        """
        # 1. ROUGE Scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for ref, pred in zip(references, predictions):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

        avg_rouge = {k: np.mean(v) for k, v in rouge_scores.items()}

        # 2. BERTScore
        # lang='en' is standard, but for medical usually works fine.
        # For strict medical adaptation, one might use a specific checkpoint, but standard BERTScore is the baseline.
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False, device=self.device)
        avg_bertscore = F1.mean().item()

        # 3. MedCLIP Score
        # Measures semantic alignment between the Input Image and the Generated Caption
        medclip_score = 0.0
        if self.use_medclip and images:
            with torch.no_grad():
                inputs = self.medclip_processor(
                    text=predictions,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                outputs = self.medclip_model(**inputs)
                # similarity is logits_per_image (image-text similarity)
                # We normalize and take the diagonal (matching image to its own caption)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                # We want the cosine similarity really, but standard CLIP score is often raw logits or diag
                # Here we take the diagonal of the similarity matrix
                diag_sim = torch.diagonal(logits_per_image)
                medclip_score = diag_sim.mean().item()

        return {
            "rouge1": avg_rouge['rouge1'],
            "rougeL": avg_rouge['rougeL'],
            "bert_score": avg_bertscore,
            "medclip_score": medclip_score
        }