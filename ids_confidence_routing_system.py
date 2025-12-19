"""
Dual-Model IDS with Deep Inspection & Weighted Scoring Logic

This module implements a production-ready intrusion detection system architecture.
Key Changes:
- Renamed 'ActionPriority' to 'ResponseAction' to reflect actual system behavior.
- Benign traffic now maps to 'ALLOW' action, not a weird priority level.
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
# 1. TERMINOLOGY (DÜZELTİLMİŞ)
# ============================================================================

class ConfidenceLevel(Enum):
    """
    Defines how much we trust the detection (Fidelity).
    """
    CONFIRMED = "CONFIRMED"     # Score >= 0.85 (Both models agree strongly)
    SUSPICIOUS = "SUSPICIOUS"   # Score 0.70 - 0.84 (Strong suspicion)
    UNCERTAIN = "UNCERTAIN"     # Score < 0.70 (Binary said Attack, but Score is low)
    SAFE = "SAFE"               # Binary model predicted Benign

class ResponseAction(Enum):
    """
    Defines the TYPE of action the system is suggesting.
    This replaces 'ActionPriority'.
    """
    BLOCK = "BLOCK"           # Stop the threat (Critical)
    MITIGATE = "MITIGATE"     # Reduce the risk / Containment (Warning)
    MONITOR = "MONITOR"       # Watch and Learn (Info)
    ALLOW = "ALLOW"           # Let it pass (Benign)

@dataclass
class TrafficSample:
    sample_id: str
    timestamp: datetime
    features: np.ndarray
    binary_prediction: str
    binary_probability: float
    attack_type: Optional[str] = None
    multiclass_probability: Optional[float] = None
    combined_score: Optional[float] = None

@dataclass
class RoutingDecision:
    sample_id: str
    timestamp: datetime
    confidence_level: ConfidenceLevel
    suggested_response: ResponseAction  # İSİM DEĞİŞTİ: priority -> suggested_response
    classification: str
    final_score: float
    actions: List[str]

    def __str__(self):
        return (f"Decision(ID={self.sample_id}, "
                f"Response={self.suggested_response.value}, "
                f"Level={self.confidence_level.value}, "
                f"Score={self.final_score:.4f})")

# ============================================================================
# 2. ROUTING LOGIC (ACTION ORIENTED)
# ============================================================================

class ConfidenceBasedRouter:
    def __init__(self, critical_threshold: float = 0.85, warning_threshold: float = 0.70):
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.decisions_log = []

    def get_specific_actions(self, response: ResponseAction, sample_id: str) -> List[str]:
        """Returns concrete actions based on the Response Category."""
        
        if response == ResponseAction.BLOCK:
            return [
                "FIREWALL: Add source IP to Blocklist",
                "NETWORK: Send TCP Reset (RST)",
                f"SIEM: Critical Alert (ID: {sample_id})"
            ]
        
        elif response == ResponseAction.MITIGATE:
            return [
                "TRAFFIC: Rate Limit / Throttle bandwidth",
                "ROUTING: Redirect to Honeypot/Sandbox",
                "FORENSICS: Trigger Full Packet Capture"
            ]
            
        elif response == ResponseAction.MONITOR:
            return [
                "LOGGING: Log event with 'Low Confidence' tag",
                "DATASET: Mark sample for retraining",
                "STATUS: No active blocking performed"
            ]
            
        else: # ALLOW
            return [
                "FIREWALL: Allow traffic flow",
                "LOGGING: Update normal traffic metrics"
            ]

    def route_decision(self, sample: TrafficSample) -> RoutingDecision:
        
        # --- PATH 1: BENIGN TRAFFIC ---
        if sample.binary_prediction == "Benign":
            return RoutingDecision(
                sample_id=sample.sample_id,
                timestamp=sample.timestamp,
                confidence_level=ConfidenceLevel.SAFE,
                suggested_response=ResponseAction.ALLOW,  # Bening -> ALLOW (Mantıklı olan bu)
                classification="NORMAL_TRAFFIC",
                final_score=sample.binary_probability,
                actions=self.get_specific_actions(ResponseAction.ALLOW, sample.sample_id)
            )

        # --- PATH 2: ATTACK TRAFFIC (Weighted Scoring) ---
        score = sample.combined_score if sample.combined_score is not None else 0.0
        attack_str = sample.attack_type if sample.attack_type else "Unknown"
        
        # Determine Response Action based on Score
        if score >= self.critical_threshold:
            # CONFIRMED -> BLOCK
            conf_level = ConfidenceLevel.CONFIRMED
            response = ResponseAction.BLOCK
            cls_text = f"THREAT_CONFIRMED: {attack_str}"
            
        elif score >= self.warning_threshold:
            # SUSPICIOUS -> MITIGATE
            conf_level = ConfidenceLevel.SUSPICIOUS
            response = ResponseAction.MITIGATE
            cls_text = f"THREAT_SUSPECTED: {attack_str}"
            
        else:
            # UNCERTAIN -> MONITOR (Binary said attack, but score is low)
            conf_level = ConfidenceLevel.UNCERTAIN
            response = ResponseAction.MONITOR
            cls_text = "THREAT_UNCERTAIN: False Positive Likely"

        # Create Decision
        decision = RoutingDecision(
            sample_id=sample.sample_id,
            timestamp=sample.timestamp,
            confidence_level=conf_level,
            suggested_response=response,
            classification=cls_text,
            final_score=score,
            actions=self.get_specific_actions(response, sample.sample_id)
        )
        
        self.decisions_log.append(decision)
        return decision

# ============================================================================
# 3. DUAL-MODEL IDS SYSTEM
# ============================================================================

class DualModelIDS:
    def __init__(self, binary_model_path, multiclass_model_path, scaler_path=None):
        self.binary_model = joblib.load(binary_model_path)
        self.multiclass_model = joblib.load(multiclass_model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        
        # Initialize Router
        self.router = ConfidenceBasedRouter(critical_threshold=0.85, warning_threshold=0.70)
    
    def preprocess(self, raw_features):
        features = raw_features.reshape(1, -1)
        if self.scaler:
            features = self.scaler.transform(features)
        return features[0]
    
    def predict_sample(self, sample_id, raw_features) -> TrafficSample:
        features = self.preprocess(raw_features)
        
        # Stage 1: Binary
        bin_pred = self.binary_model.predict([features])[0] # 0=Benign, 1=Attack
        bin_proba = self.binary_model.predict_proba([features])[0]
        
        label = "Attack" if bin_pred == 1 else "Benign"
        # Assuming index 1 is Attack
        bin_conf = bin_proba[1] if bin_pred == 1 else bin_proba[0]

        attack_type = None
        multi_conf = None
        combined_score = None

        # Stage 2: Always check Multiclass if Attack detected
        if label == "Attack":
            attack_type = self.multiclass_model.predict([features])[0]
            multi_probs = self.multiclass_model.predict_proba([features])[0]
            multi_conf = np.max(multi_probs)
            
            # Weighted Score Calculation
            combined_score = (0.7 * bin_conf) + (0.3 * multi_conf)
            combined_score = round(combined_score, 4)

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

    def detect_and_route(self, sample_id, raw_features):
        sample = self.predict_sample(sample_id, raw_features)
        return self.router.route_decision(sample)