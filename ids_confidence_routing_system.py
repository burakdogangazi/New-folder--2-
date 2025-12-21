"""
Dual-Model IDS with Deep Inspection & Audience-Based Response Logic

Architecture:
1. Deep Inspection: If Binary predicts 'Attack', ALWAYS run Stage 2 (Multiclass).
2. Weighted Scoring: Final Score = (0.7 * Binary_Conf) + (0.3 * Multi_Conf).
3. Safe Thresholding: Benign traffic is split into 'Verified Safe' and 'Anomalies'.
4. Distinct Actions: Low confidence Attack vs Low confidence Benign have different handling.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# 1. NEW TERMINOLOGY & ENUMS
# ============================================================================

class ConfidenceLevel(Enum):
    """
    Defines the fidelity/certainty of the detection.
    """
    CONFIRMED = "CONFIRMED"     # Score >= 0.85 (High Fidelity Threat)
    SUSPICIOUS = "SUSPICIOUS"   # Score 0.70 - 0.84 (Medium Fidelity Threat)
    UNCERTAIN = "UNCERTAIN"     # Score < 0.70 (Low Fidelity / Gray Zone)
    SAFE = "SAFE"               # High Confidence Benign

class ResponseAction(Enum):
    """
    Defines the specific operational response type.
    Differentiates between SOC actions and Data Science actions.
    """
    BLOCK = "BLOCK"                 # Critical Threat -> Stop immediately.
    MITIGATE = "MITIGATE"           # High Risk -> Contain/Throttle.
    LOG_SUSPICIOUS = "LOG_SUSPICIOUS" # Attack (Low Conf) -> Log for SOC (Potential False Positive).
    FLAG_ANOMALY = "FLAG_ANOMALY"   # Benign (Low Conf) -> Flag for Retraining (Potential False Negative).
    ALLOW = "ALLOW"                 # Benign (High Conf) -> Pass traffic.

@dataclass
class TrafficSample:
    """Represents a processed network traffic sample."""
    sample_id: str
    timestamp: datetime
    features: np.ndarray
    binary_prediction: str      # "Attack" or "Benign"
    binary_probability: float
    attack_type: Optional[str] = None
    multiclass_probability: Optional[float] = None
    combined_score: Optional[float] = None  # The Final Weighted Score

@dataclass
class RoutingDecision:
    """Final decision object containing specific security actions."""
    sample_id: str
    timestamp: datetime
    confidence_level: ConfidenceLevel
    suggested_response: ResponseAction
    classification: str         # Human readable status text
    final_score: float
    actions: List[str]

    def __str__(self):
        return (f"Decision(ID={self.sample_id}, "
                f"Response={self.suggested_response.value}, "
                f"Level={self.confidence_level.value}, "
                f"Score={self.final_score:.4f})")

# ============================================================================
# 2. CONFIDENCE-BASED ROUTER
# ============================================================================

class ConfidenceBasedRouter:
    """
    Routes decisions based on scores and thresholds.
    Handles the 'Gray Zone' by splitting actions for low-confidence predictions.
    """
    
    def __init__(self, 
                 critical_threshold: float = 0.85, 
                 warning_threshold: float = 0.70,
                 safe_threshold: float = 0.80): # YENİ: Temiz trafik için güven eşiği
        
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.safe_threshold = safe_threshold
        self.decisions_log = []

    def get_specific_actions(self, response: ResponseAction, sample_id: str) -> List[str]:
        """Returns concrete actions based on the Response Category."""
        
        if response == ResponseAction.BLOCK:
            return [
                "FIREWALL: Add Source IP to Blocklist",
                "SESSION: Terminate TCP Connection (RST)",
                f"SIEM: Critical Alert (ID: {sample_id})"
            ]
        
        elif response == ResponseAction.MITIGATE:
            return [
                "TRAFFIC: Apply Rate Limiting (Throttle)",
                "ROUTING: Redirect to Honeypot Environment",
                "FORENSICS: Enable Full Packet Capture"
            ]
            
        elif response == ResponseAction.LOG_SUSPICIOUS:
            # Context: Model said Attack, but score is low.
            # Target: SOC Analysts
            return [
                "LOGGING: Record in 'Suspicious Events' Database",
                "SIEM: Send INFO-Level Notification (No PagerDuty)",
                "STATUS: Pass Traffic (Potential False Positive - Monitor)"
            ]
            
        elif response == ResponseAction.FLAG_ANOMALY:
            # Context: Model said Benign, but score is low.
            # Target: Data Science / MLOps
            return [
                "STATUS: Pass Traffic (To avoid False Positives)",
                "FORENSICS: TRIGGER_PCAP (Capture Full Packets for Zero-Day Analysis)",
                "DATASET: Tag as 'Unknown Pattern' for Analyst Review",
                "METRICS: Increment 'Anomaly' Counter"
            ]
            
        else: # ALLOW
            return [
                "FIREWALL: Allow Traffic Flow",
                "METRICS: Update Normal Traffic Counters"
            ]

    def route_decision(self, sample: TrafficSample) -> RoutingDecision:
        
        # --- PATH 1: BENIGN TRAFFIC ---
        if sample.binary_prediction == "Benign":
            score = sample.binary_probability
            
            if score >= self.safe_threshold:
                # CASE: HIGH CONFIDENCE BENIGN -> ALLOW
                return RoutingDecision(
                    sample_id=sample.sample_id,
                    timestamp=sample.timestamp,
                    confidence_level=ConfidenceLevel.SAFE,
                    suggested_response=ResponseAction.ALLOW,
                    classification="NORMAL_TRAFFIC_VERIFIED",
                    final_score=score,
                    actions=self.get_specific_actions(ResponseAction.ALLOW, sample.sample_id)
                )
            else:
                # CASE: LOW CONFIDENCE BENIGN -> FLAG_ANOMALY
                # "Model temiz dedi ama zar attı (%51-%79 arası)"
                return RoutingDecision(
                    sample_id=sample.sample_id,
                    timestamp=sample.timestamp,
                    confidence_level=ConfidenceLevel.UNCERTAIN,
                    suggested_response=ResponseAction.FLAG_ANOMALY,
                    classification="BENIGN_ANOMALY_DETECTED",
                    final_score=score,
                    actions=self.get_specific_actions(ResponseAction.FLAG_ANOMALY, sample.sample_id)
                )

        # --- PATH 2: ATTACK TRAFFIC ---
        # Calculate Weighted Score
        score = sample.combined_score if sample.combined_score is not None else 0.0
        attack_str = sample.attack_type if sample.attack_type else "Unknown"
        
        if score >= self.critical_threshold:
            # CASE: CONFIRMED ATTACK -> BLOCK
            return RoutingDecision(
                sample_id=sample.sample_id,
                timestamp=sample.timestamp,
                confidence_level=ConfidenceLevel.CONFIRMED,
                suggested_response=ResponseAction.BLOCK,
                classification=f"THREAT_CONFIRMED: {attack_str}",
                final_score=score,
                actions=self.get_specific_actions(ResponseAction.BLOCK, sample.sample_id)
            )
            
        elif score >= self.warning_threshold:
            # CASE: SUSPICIOUS ATTACK -> MITIGATE
            return RoutingDecision(
                sample_id=sample.sample_id,
                timestamp=sample.timestamp,
                confidence_level=ConfidenceLevel.SUSPICIOUS,
                suggested_response=ResponseAction.MITIGATE,
                classification=f"THREAT_SUSPECTED: {attack_str}",
                final_score=score,
                actions=self.get_specific_actions(ResponseAction.MITIGATE, sample.sample_id)
            )
            
        else:
            # CASE: LOW CONFIDENCE ATTACK -> LOG_SUSPICIOUS
            # "Model saldırı dedi ama uzman onaylamadı/emin değil"
            return RoutingDecision(
                sample_id=sample.sample_id,
                timestamp=sample.timestamp,
                confidence_level=ConfidenceLevel.UNCERTAIN,
                suggested_response=ResponseAction.LOG_SUSPICIOUS,
                classification="THREAT_UNCERTAIN_NOISE",
                final_score=score,
                actions=self.get_specific_actions(ResponseAction.LOG_SUSPICIOUS, sample.sample_id)
            )

# ============================================================================
# 3. DUAL-MODEL IDS SYSTEM
# ============================================================================

class DualModelIDS:
    """
    Production-ready Intrusion Detection System.
    """
    
    def __init__(self, 
                 binary_model_path: str,
                 multiclass_model_path: str,
                 scaler_path: str = None,
                 pca_model_path: str = None):
        
        # Load resources
        print("Loading models and preprocessors...")
        self.binary_model = joblib.load(binary_model_path)
        self.multiclass_model = joblib.load(multiclass_model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.pca = joblib.load(pca_model_path) if pca_model_path else None
        
        # Initialize Router with Defined Thresholds
        self.router = ConfidenceBasedRouter(
            critical_threshold=0.85, 
            warning_threshold=0.70,
            safe_threshold=0.80
        )
        self.predictions_cache = []
    
    def preprocess_features(self, raw_features: np.ndarray) -> np.ndarray:
        """Preprocess raw network features."""
        features = raw_features.reshape(1, -1)
        if self.scaler:
            features = self.scaler.transform(features)
        if self.pca:
            features = self.pca.transform(features)
        return features[0]
    
    def predict_sample(self, sample_id: str, raw_features: np.ndarray) -> TrafficSample:
        """
        Executes Deep Inspection Logic.
        """
        features = self.preprocess_features(raw_features)
        
        # --- STAGE 1: Binary Classification ---
        bin_pred = self.binary_model.predict(features.reshape(1, -1))[0]
        bin_probs = self.binary_model.predict_proba(features.reshape(1, -1))[0]
        
        # Assuming index 1 is Attack, 0 is Benign
        label = "Attack" if bin_pred == 1 else "Benign"
        bin_conf = bin_probs[1] if bin_pred == 1 else bin_probs[0]

        attack_type = None
        multi_conf = None
        combined_score = None

        # --- STAGE 2: Deep Inspection (Always run if Attack detected) ---
        if label == "Attack":
            multi_pred = self.multiclass_model.predict(features.reshape(1, -1))[0]
            multi_probs = self.multiclass_model.predict_proba(features.reshape(1, -1))[0]
            
            attack_type = multi_pred
            multi_conf = np.max(multi_probs)
            
            # Weighted Scoring Formula
            # 70% Binary Confidence + 30% Multiclass Confidence
            try:
                combined_score = (0.7 * float(bin_conf)) + (0.3 * float(multi_conf))
                combined_score = round(combined_score, 4)
            except Exception as e:
                logging.error(f"Scoring error: {e}")
                combined_score = bin_conf # Fallback

        return TrafficSample(
            sample_id=sample_id,
            timestamp=datetime.now(),
            features=features,
            binary_prediction=label,
            binary_probability=bin_conf,
            attack_type=attack_type,
            multiclass_probability=multi_conf,
            combined_score=combined_score
        )
    
    def detect_and_route(self, sample_id: str, raw_features: np.ndarray) -> RoutingDecision:
        """Public interface."""
        sample = self.predict_sample(sample_id, raw_features)
        decision = self.router.route_decision(sample)
        
        # Cache for reporting
        self.predictions_cache.append({'sample': sample, 'decision': decision})
        
        return decision