"""
Confidence-Based Routing Logic for Dual-Model IDS System

This module implements a production-ready intrusion detection system that uses
a two-stage classification pipeline with confidence-based decision routing.

Stage 1: Binary Classification (Attack vs Benign)
Stage 2: Multiclass Classification (Attack Type Identification)
Stage 3: Confidence-Based Action Routing (High/Medium/Low confidence)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class ConfidenceLevel(Enum):
    """Confidence level classification."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ActionPriority(Enum):
    """Action priority classification."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class TrafficSample:
    """Represents a network traffic sample."""
    sample_id: str
    timestamp: datetime
    features: np.ndarray
    binary_prediction: str
    binary_probability: float
    attack_type: str = None
    attack_probability: float = None
    confidence_level: ConfidenceLevel = None


@dataclass
class RoutingDecision:
    """Represents a routing decision with actions."""
    sample_id: str
    confidence_level: ConfidenceLevel
    classification: str
    attack_type: str = None
    confidence_score: float = None
    priority: ActionPriority = None
    actions: List[str] = None
    timestamp: datetime = None
    
    def __str__(self):
        """Safe string representation."""
        return (f"RoutingDecision(sample_id={self.sample_id}, "
                f"confidence_level={self.confidence_level}, "
                f"classification={self.classification}, "
                f"attack_type={self.attack_type}, "
                f"confidence_score={self.confidence_score}, "
                f"priority={self.priority}, "
                f"actions_count={len(self.actions) if self.actions else 0})")
    
    def __repr__(self):
        """Safe representation."""
        return self.__str__()


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIDENCE-BASED ROUTING SYSTEM
# ============================================================================

class ConfidenceBasedRouter:
    """
    Routes network traffic decisions based on classification confidence scores.
    Implements three-tier decision routing: High/Medium/Low confidence.
    """
    
    def __init__(self, 
                 high_confidence_threshold: float = 0.85,
                 medium_confidence_threshold: float = 0.60):
        """
        Initialize router with confidence thresholds.
        
        Parameters
        ----------
        high_confidence_threshold : float
            Threshold for high-confidence decisions (default: 90%)
        medium_confidence_threshold : float
            Threshold for medium-confidence decisions (default: 70%)
        """
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.decisions_log = []
        
    def classify_confidence(self, confidence_score: float) -> ConfidenceLevel:
        """
        Classify confidence level based on score.
        
        Parameters
        ----------
        confidence_score : float
            Confidence score between 0 and 1
            
        Returns
        -------
        ConfidenceLevel
            HIGH (>85%), MEDIUM (60-85%), or LOW (<60%)
        """
        if confidence_score >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def get_high_confidence_actions(self, 
                                    sample: TrafficSample) -> List[str]:
        """
        Generate actions for HIGH-confidence malicious traffic (>85%).
        
        Classification: Confirmed attack with specific type identification
        
        Parameters
        ----------
        sample : TrafficSample
            Network traffic sample with predictions
            
        Returns
        -------
        List[str]
            List of immediate actions to take
        """
        actions = [
            f"[CRITICAL] Immediate automated blocking at network perimeter",
            f"[ALERT] SIEM alert: {sample.attack_type} attack detected (Confidence: {(sample.attack_probability if sample.attack_probability is not None else 0):.2%})",
            f"[TICKET] Incident ticket creation: {sample.attack_type} - Sample ID: {sample.sample_id}",
            f"[SOC] Security Operations Center notification - Immediate action required",
            f"[BLOCK] IP/Port blocking rules deployed",
            f"[FORENSICS] Enable packet capture for forensic analysis",
            f"[ISOLATION] Network segment isolation if applicable"
        ]
        return actions
    
    def get_medium_confidence_actions(self, 
                                      sample: TrafficSample) -> List[str]:
        """
        Generate actions for MEDIUM-confidence suspicious traffic (60-85%).
        
        Classification: Suspicious activity requiring investigation
        
        Parameters
        ----------
        sample : TrafficSample
            Network traffic sample with predictions
            
        Returns
        -------
        List[str]
            List of investigation and containment actions
        """
        actions = [
            f"[WARNING] Traffic logging with elevated priority",
            f"[ANALYSIS] Sandbox analysis initiated for: {sample.attack_type}",
            f"[RATE_LIMIT] Temporary rate limiting applied (threshold: 50% of normal)",
            f"[ALERT] Tier-1 security analyst alert - Confidence: {(sample.attack_probability if sample.attack_probability is not None else 0):.2%}",
            f"[THREAT_INTEL] Enrichment with threat intelligence feeds in progress",
            f"[MONITOR] Enhanced monitoring enabled for source IP",
            f"[QUEUE] Queued for incident review by security team",
            f"[SAMPLE_STORE] Store full packet capture for later analysis"
        ]
        return actions
    
    def get_low_confidence_actions(self, 
                                   sample: TrafficSample) -> List[str]:
        """
        Generate actions for LOW-confidence benign traffic (<60%).
        
        Classification: Likely benign network activity
        
        Parameters
        ----------
        sample : TrafficSample
            Network traffic sample with predictions
            
        Returns
        -------
        List[str]
            List of routine logging and monitoring actions
        """
        actions = [
            f"[INFO] Standard logging for baseline monitoring",
            f"[SAMPLE] Periodic sampling (1/1000) for quality assurance",
            f"[TRAINING] Include in model retraining dataset",
            f"[NO_ALERT] No immediate operational impact",
            f"[ROUTINE] Routine network flow logging only",
            f"[STATS] Update baseline traffic statistics"
        ]
        return actions
    
    def route_decision(self, sample: TrafficSample) -> RoutingDecision:
        """
        Route traffic decision based on classification and confidence.
        
        Parameters
        ----------
        sample : TrafficSample
            Network traffic sample with dual-model predictions
            
        Returns
        -------
        RoutingDecision
            Structured routing decision with appropriate actions
        """
        # Determine confidence level
        # Use attack_probability if available (Stage 2 ran), otherwise use binary_probability
        if sample.binary_prediction == "Attack" and sample.attack_probability is not None:
            confidence_score = sample.attack_probability
        else:
            confidence_score = sample.binary_probability
        
        confidence_level = self.classify_confidence(confidence_score)
        
        # Determine priority
        priority_map = {
            ConfidenceLevel.HIGH: ActionPriority.CRITICAL,
            ConfidenceLevel.MEDIUM: ActionPriority.HIGH,
            ConfidenceLevel.LOW: ActionPriority.INFO
        }
        priority = priority_map[confidence_level]
        
        # Get appropriate actions and classification
        # If LOW confidence (below 60%), treat as BENIGN regardless of prediction
        if confidence_level == ConfidenceLevel.HIGH:
            actions = self.get_high_confidence_actions(sample)
            classification = f"CONFIRMED_ATTACK: {sample.attack_type}"
        elif confidence_level == ConfidenceLevel.MEDIUM:
            actions = self.get_medium_confidence_actions(sample)
            classification = f"SUSPICIOUS: Possible {sample.attack_type}"
        else:  # LOW - treat as BENIGN
            actions = self.get_low_confidence_actions(sample)
            classification = "BENIGN"
            sample.attack_type = None  # Clear attack type for benign classification
        
        # Create routing decision
        decision = RoutingDecision(
            sample_id=sample.sample_id,
            confidence_level=confidence_level,
            classification=classification,
            attack_type=sample.attack_type,
            confidence_score=confidence_score,
            priority=priority,
            actions=actions,
            timestamp=sample.timestamp
        )
        
        self.decisions_log.append(decision)
        return decision


# ============================================================================
# DUAL-MODEL IDS SYSTEM
# ============================================================================

class DualModelIDS:
    """
    Production-ready Intrusion Detection System with dual-model architecture:
    - Stage 1: Binary Classification (Attack vs Benign)
    - Stage 2: Multiclass Classification (Attack Type)
    - Stage 3: Confidence-based Routing
    """
    
    def __init__(self, 
                 binary_model_path: str,
                 multiclass_model_path: str,
                 scaler_path: str = None,
                 pca_model_path: str = None):
        """
        Initialize IDS system with pretrained models.
        
        Parameters
        ----------
        binary_model_path : str
            Path to binary classification model
        multiclass_model_path : str
            Path to multiclass classification model
        scaler_path : str, optional
            Path to feature scaler (if None, no scaling applied)
        pca_model_path : str, optional
            Path to PCA model for feature transformation (if None, no PCA applied)
        """
        # Load models
        self.binary_model = joblib.load(binary_model_path)
        self.multiclass_model = joblib.load(multiclass_model_path)
        
        # Load optional preprocessing models
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.pca = joblib.load(pca_model_path) if pca_model_path else None
        
        # Initialize router
        self.router = ConfidenceBasedRouter(
            high_confidence_threshold=0.90,
            medium_confidence_threshold=0.70
        )
        
        self.predictions_cache = []
    
    def preprocess_features(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Preprocess raw network features through scaling and PCA.
        
        Parameters
        ----------
        raw_features : np.ndarray
            Raw network traffic features
            
        Returns
        -------
        np.ndarray
            Preprocessed and dimensionally-reduced features
        """
        features = raw_features.reshape(1, -1)
        
        # Scale features if scaler available
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Apply PCA if available
        if self.pca:
            features = self.pca.transform(features)
        
        return features[0]
    
    def predict_sample(self, 
                      sample_id: str,
                      raw_features: np.ndarray) -> TrafficSample:
        """
        Predict network traffic sample using dual-model pipeline.
        
        Stage 1: Binary classification (Attack/Benign)
        Stage 2: If Attack -> Multiclass classification (Attack type)
        
        Parameters
        ----------
        sample_id : str
            Unique sample identifier
        raw_features : np.ndarray
            Raw network traffic features
            
        Returns
        -------
        TrafficSample
            Prediction results with confidence scores
        """
        # Preprocess
        features = self.preprocess_features(raw_features)
        
        # Stage 1: Binary Classification
        binary_pred = self.binary_model.predict(features.reshape(1, -1))[0]
        binary_proba = self.binary_model.predict_proba(features.reshape(1, -1))[0]
        
        # Get confidence for binary prediction
        if binary_pred == 1:  # Attack
            binary_confidence = binary_proba[1]
            label = "Attack"
        else:  # Benign
            binary_confidence = binary_proba[0]
            label = "Benign"
        
        # Stage 2: Multiclass Classification (only if Attack with HIGH confidence)
        attack_type = None
        attack_confidence = None
        
        if label == "Attack" and binary_confidence >= 0.90:  # Only Stage 2 if >90% sure it's attack
            multiclass_pred = self.multiclass_model.predict(features.reshape(1, -1))[0]
            multiclass_proba = self.multiclass_model.predict_proba(features.reshape(1, -1))[0]
            
            attack_type = multiclass_pred
            attack_confidence = np.max(multiclass_proba)
        
        # Create traffic sample
        sample = TrafficSample(
            sample_id=sample_id,
            timestamp=datetime.now(),
            features=features,
            binary_prediction=label,
            binary_probability=binary_confidence,
            attack_type=attack_type,
            attack_probability=attack_confidence
        )
        
        return sample
    
    def detect_and_route(self, 
                        sample_id: str,
                        raw_features: np.ndarray) -> RoutingDecision:
        """
        Complete detection and routing pipeline.
        
        Parameters
        ----------
        sample_id : str
            Unique sample identifier
        raw_features : np.ndarray
            Raw network traffic features
            
        Returns
        -------
        RoutingDecision
            Routing decision with actions
        """
        # Get predictions
        sample = self.predict_sample(sample_id, raw_features)
        
        # Route decision
        decision = self.router.route_decision(sample)
        
        self.predictions_cache.append({
            'sample_id': sample_id,
            'sample': sample,
            'decision': decision
        })
        
        return decision


# ============================================================================
# BATCH PROCESSING & REPORTING
# ============================================================================

class BatchIDS:
    """Batch processing for multiple samples with reporting."""
    
    def __init__(self, ids_system: DualModelIDS):
        """Initialize batch processor."""
        self.ids = ids_system
        self.results = []
    
    def process_batch(self, 
                     samples_df: pd.DataFrame,
                     feature_columns: List[str]) -> pd.DataFrame:
        """
        Process batch of network samples.
        
        Parameters
        ----------
        samples_df : pd.DataFrame
            DataFrame with network samples
        feature_columns : List[str]
            Column names for feature extraction
            
        Returns
        -------
        pd.DataFrame
            Results with predictions and routing decisions
        """
        results = []
        
        for idx, row in samples_df.iterrows():
            sample_id = f"sample_{idx}_{datetime.now().timestamp()}"
            raw_features = row[feature_columns].values
            
            decision = self.ids.detect_and_route(sample_id, raw_features)
            
            results.append({
                'sample_id': sample_id,
                'timestamp': decision.timestamp,
                'confidence_level': decision.confidence_level.value,
                'classification': decision.classification,
                'attack_type': decision.attack_type,
                'confidence_score': decision.confidence_score,
                'priority': decision.priority.value,
                'actions_count': len(decision.actions),
                'actions': ' | '.join(decision.actions)
            })
        
        self.results = results
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def save_results(self, output_path: str = "ids_results"):
        """
        Save detailed results and reports.
        
        Parameters
        ----------
        output_path : str
            Output directory for results
        """
        Path(output_path).mkdir(exist_ok=True)
        
        results_df = pd.DataFrame(self.results)
        
        # Save full results
        results_df.to_csv(f"{output_path}/ids_predictions.csv", index=False)
        
        # Save summary by confidence
        summary = results_df.groupby('confidence_level').agg({
            'sample_id': 'count',
            'confidence_score': ['mean', 'min', 'max']
        })
        summary.to_csv(f"{output_path}/confidence_summary.csv")
        
        # Save summary by priority
        priority_summary = results_df.groupby('priority').agg({
            'sample_id': 'count'
        })
        priority_summary.to_csv(f"{output_path}/priority_summary.csv")


# ============================================================================
# END OF IDS SYSTEM
# ============================================================================
