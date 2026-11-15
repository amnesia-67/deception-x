"""
Branch 3: Cognitive Reasoning & Frontier Model Integration (Gemini 4 Argon / Gemini 3.8 Flash)
Accredited to Gemini.pdf Specifications (Pages 30-35).
Includes:
- Strict DeceptionX Observe-Think-Summarize Structured Output Schema
- System Prompt with 50/50 Base Rate Anchor & Reality Monitoring rules
- 3-Tier Resilient Circuit Breaker:
    Tier 1: gemini-4-argon (TTFT <= 2.5s)
    Tier 2: gemini-3.8-flash (TTFT <= 1.5s, timeout 2.2s)
    Tier 3: Local Deterministic Neuro-Tabular Synthesis (0.58 * p_neural + 0.42 * p_xgb)
"""

import os
import time
import asyncio
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

# 1. Strict DeceptionX Observe-Think-Summarize Structured Output Schema
class DeceptionXVerdictSchema(BaseModel):
    observe_trace: List[str] = Field(
        description="<observe> Timestamped multimodal anomalies across keyframes, rPPG/AUs, acoustic pitch/jitter, and VAD latency."
    )
    think_trace: str = Field(
        description="<think> Cross-modal discrepancy deduction, Reality Monitoring external vs. internal memory evaluation, and anxiety vs. deception disambiguation."
    )
    classification: Literal["Deceptive", "Truthful"] = Field(
        description="Binary categorical conclusion anchored to a 50/50 prior base rate."
    )
    p_argon: float = Field(
        ge=0.01,
        le=0.99,
        description="Calibrated probability of deception in [0.01, 0.99]."
    )
    forensic_summary: str = Field(
        description="<verdict> Concise 2-sentence Glass-Box forensic explanation citing primary conflicting channels."
    )

# 2. Curated 16 Gold-Standard DeceptChain & LOSO Exemplars (8 Truthful, 8 Deceptive)
GOLD_STANDARD_16_EXEMPLARS = [
    # 8 TRUTHFUL EXEMPLARS (Including 4 High-Stress MU3D Negative Truths)
    {
        "id": "T01_NegativeTruth_Bereavement",
        "category": "High-Stress MU3D Negative Truth",
        "z_anomalies": {"Autonomic_IBI_SDNN_z": -2.1, "Autonomic_HeartRate_z": +2.4, "VocalStress_Praat_RAP_z": +1.9, "FACS_AU04_BrowLowerer_z": +2.2, "FACS_AU07_LidTightener_z": +1.8},
        "vad_latency_ms": 480,
        "discrepancies": {"D_va": 0.12, "D_vl": 0.14, "D_al": 0.11},
        "transcript": "I was at the emergency department waiting room on 4th floor; the smell of floor disinfectant was overwhelming, and Dr. Sarah Jenkins informed us at 3:15 AM.",
        "priors": {"p_neural": 0.08, "p_xgb": 0.05},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.03, "summary": "Congruent negative affect and severe autonomic distress without cross-modal discrepancy; rich in sensory and temporal anchors."}
    },
    {
        "id": "T02_NegativeTruth_Collision",
        "category": "High-Stress MU3D Negative Truth",
        "z_anomalies": {"VocalStress_Praat_Local_Shimmer_z": +2.1, "Autonomic_Facial_MicroTremor_z": +1.9, "Silero_VAD_Latency_z": -0.7},
        "vad_latency_ms": 320,
        "discrepancies": {"D_va": 0.14, "D_vl": 0.13, "D_al": 0.12},
        "transcript": "The red Toyota truck blew through the crosswalk from my left, tires screeching with black smoke and burning rubber before hitting the curb.",
        "priors": {"p_neural": 0.06, "p_xgb": 0.04},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.04, "summary": "Acoustic stress accompanied by instant response latency and vivid perceptual reality monitoring markers."}
    },
    {
        "id": "T03_NegativeTruth_WrongfulAccusation",
        "category": "High-Stress MU3D Negative Truth",
        "z_anomalies": {"VocalStress_Praat_Local_Jitter_z": +1.8, "FACS_AU04_BrowLowerer_z": +2.0, "FACS_AU23_LipTightener_z": +1.7},
        "vad_latency_ms": 390,
        "discrepancies": {"D_va": 0.15, "D_vl": 0.12, "D_al": 0.14},
        "transcript": "I never touched the lab vault keys; I logged out of workstation 7 at 5:02 PM, badge swipe recorded at the north turnstile.",
        "priors": {"p_neural": 0.09, "p_xgb": 0.06},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.05, "summary": "Congruent indignation; facial action units align with vocal inflection and verifiable external temporal facts."}
    },
    {
        "id": "T04_NegativeTruth_PanicAttack",
        "category": "High-Stress MU3D Negative Truth",
        "z_anomalies": {"Autonomic_HeartRate_z": +2.6, "Autonomic_Cheek_POS_BVP_z": -2.2, "LinguisticRM_FirstPerson_z": +1.5},
        "vad_latency_ms": 510,
        "discrepancies": {"D_va": 0.13, "D_vl": 0.15, "D_al": 0.10},
        "transcript": "My throat completely constricted in the lecture hall, hands shaking so badly that the laser pointer trembled across the podium.",
        "priors": {"p_neural": 0.07, "p_xgb": 0.05},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.04, "summary": "Severe autonomic arousal accompanied by high first-person pronoun rate and zero deceptive channel masking."}
    },
    {
        "id": "T05_PositiveTruth_Commute",
        "category": "Baseline Truthful",
        "z_anomalies": {"Silero_VAD_Latency_z": -0.3, "Ocular_MAR_mean_z": +0.1},
        "vad_latency_ms": 360,
        "discrepancies": {"D_va": 0.11, "D_vl": 0.10, "D_al": 0.09},
        "transcript": "I walked down Mill Avenue, grabbed a black cold brew at Cartel, and took the light rail train eastward.",
        "priors": {"p_neural": 0.03, "p_xgb": 0.02},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.02, "summary": "Low autonomic arousal, rapid latency, and coherent spatial-temporal memory."}
    },
    {
        "id": "T06_PositiveTruth_Vacation",
        "category": "Baseline Truthful",
        "z_anomalies": {"FACS_AU06_CheekRaiser_z": +1.8, "FACS_AU12_LipCornerPuller_z": +2.1},
        "vad_latency_ms": 340,
        "discrepancies": {"D_va": 0.09, "D_vl": 0.11, "D_al": 0.08},
        "transcript": "The ocean water in San Diego was freezing, but we spent three hours on the pier watching pelicans dive for baitfish.",
        "priors": {"p_neural": 0.02, "p_xgb": 0.01},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.01, "summary": "Genuine Duchenne smile markers with high sensory-perceptual grounding."}
    },
    {
        "id": "T07_PositiveTruth_ProjectDefense",
        "category": "Baseline Truthful",
        "z_anomalies": {"LinguisticRM_WordCount_z": +0.8, "SpeechIntent_Assertive_z": +1.1},
        "vad_latency_ms": 420,
        "discrepancies": {"D_va": 0.12, "D_vl": 0.13, "D_al": 0.11},
        "transcript": "We profiled the Triton kernels with PyTorch Nsight Systems, isolating memory bandwidth bottleneck at the reduction layer.",
        "priors": {"p_neural": 0.04, "p_xgb": 0.03},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.03, "summary": "Assertive delivery with natural technical detail and synchronized multimodal prosody."}
    },
    {
        "id": "T08_PositiveTruth_ChildhoodHome",
        "category": "Baseline Truthful",
        "z_anomalies": {"Ocular_Gaze_Pitch_dispersion_z": +0.4, "LinguisticRM_Sensory_Ratio_z": +1.6},
        "vad_latency_ms": 380,
        "discrepancies": {"D_va": 0.10, "D_vl": 0.12, "D_al": 0.09},
        "transcript": "Our back porch had squeaky pine floorboards, and you could hear the evening cicadas right through the screen door.",
        "priors": {"p_neural": 0.03, "p_xgb": 0.02},
        "expected_verdict": {"classification": "Truthful", "p_argon": 0.02, "summary": "Unscripted sensory recall with steady physiological baseline."}
    },
    # 8 DECEPTIVE EXEMPLARS (Including 4 Calm Rehearsed Alibis & 4 Improvised Lies)
    {
        "id": "D01_RehearsedLie_WeekendAlibi",
        "category": "Calm Rehearsed Alibi",
        "z_anomalies": {"Silero_VAD_Latency_z": +2.4, "LinguisticRM_Sensory_Ratio_z": -2.2, "LinguisticRM_FirstPerson_z": -1.9},
        "vad_latency_ms": 1150,
        "discrepancies": {"D_va": 0.46, "D_vl": 0.49, "D_al": 0.42},
        "transcript": "Everything was completely normal; one spent the evening studying at the library, simply reading until closing time.",
        "priors": {"p_neural": 0.92, "p_xgb": 0.94},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.97, "summary": "Delayed VAD latency, pronounced sensory vacuity, pronoun distancing, and elevated cross-modal divergence."}
    },
    {
        "id": "D02_RehearsedLie_NonDuchenneSmile",
        "category": "Calm Rehearsed Alibi",
        "z_anomalies": {"FACS_AU12_LipCornerPuller_z": +2.3, "FACS_AU06_CheekRaiser_z": -0.8, "Ocular_HeadPose_Rotational_Rigidity_z": +2.1},
        "vad_latency_ms": 980,
        "discrepancies": {"D_va": 0.45, "D_vl": 0.43, "D_al": 0.39},
        "transcript": "No, I had nothing to do with that incident; everyone knows I was in room 302 the whole afternoon.",
        "priors": {"p_neural": 0.89, "p_xgb": 0.91},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.95, "summary": "Masked polite smile without eye muscle contraction (AU12 without AU06) combined with rigid head pose and delayed latency."}
    },
    {
        "id": "D03_RehearsedLie_ConcealedCard",
        "category": "Calm Rehearsed Alibi",
        "z_anomalies": {"Autonomic_Facial_MicroTremor_z": +2.2, "LinguisticRM_Hedging_z": +2.5, "LinguisticRM_Word_Repetition_z": +1.8},
        "vad_latency_ms": 1050,
        "discrepancies": {"D_va": 0.43, "D_vl": 0.44, "D_al": 0.40},
        "transcript": "The card... it was perhaps a blue triangle, or maybe an oval, I'm pretty sure it had rounded corners.",
        "priors": {"p_neural": 0.90, "p_xgb": 0.88},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.94, "summary": "Sub-visual micro-tremor with pervasive hedging language and repetitive speech disfluencies."}
    },
    {
        "id": "D04_RehearsedLie_FictitiousMeeting",
        "category": "Calm Rehearsed Alibi",
        "z_anomalies": {"Silero_VAD_Latency_z": +2.1, "LinguisticRM_Cognitive_Operations_z": +2.3, "SpeechIntent_Stalling_z": +2.0},
        "vad_latency_ms": 1100,
        "discrepancies": {"D_va": 0.47, "D_vl": 0.45, "D_al": 0.44},
        "transcript": "Well, we essentially discussed the strategic priorities and aligned on milestones, as one would expect.",
        "priors": {"p_neural": 0.91, "p_xgb": 0.93},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.96, "summary": "Excessive cognitive operations words, stalling speech intent, and utter absence of perceptual Reality Monitoring details."}
    },
    {
        "id": "D05_ImprovisedLie_Vasoconstriction",
        "category": "High-Stakes Improvised Fabrication",
        "z_anomalies": {"VocalStress_Praat_RAP_z": +2.5, "Autonomic_Cheek_POS_BVP_z": -2.4, "FACS_AU24_LipPressor_z": +2.0},
        "vad_latency_ms": 1300,
        "discrepancies": {"D_va": 0.55, "D_vl": 0.51, "D_al": 0.48},
        "transcript": "The transaction was authorized by regional operations; I don't recall the specific supervisor's name right now.",
        "priors": {"p_neural": 0.95, "p_xgb": 0.96},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.98, "summary": "Acute vocal fold stress, severe sympathetic facial vasoconstriction, lip pressing, and sharp cross-modal discordance."}
    },
    {
        "id": "D06_ImprovisedLie_SpatialProbeEvasion",
        "category": "High-Stakes Improvised Fabrication",
        "z_anomalies": {"Ocular_Gaze_Aversion_gt10deg_z": +2.6, "SpeechIntent_Stalling_Repetition_z": +2.4, "Silero_VAD_Latency_z": +2.2},
        "vad_latency_ms": 1420,
        "discrepancies": {"D_va": 0.52, "D_vl": 0.48, "D_al": 0.49},
        "transcript": "Who was to my left? Uh, someone... a person with a dark jacket, I think, or maybe nobody specific was standing there.",
        "priors": {"p_neural": 0.94, "p_xgb": 0.95},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.97, "summary": "Sudden ocular gaze aversion, cognitive stalling, and inability to resolve unexpected spatial perceptual probe."}
    },
    {
        "id": "D07_ImprovisedLie_ChannelMismatch",
        "category": "High-Stakes Improvised Fabrication",
        "z_anomalies": {"VocalStress_Praat_Local_Shimmer_z": +2.4, "FACS_AU12_LipCornerPuller_z": +2.2, "Autonomic_IBI_SDNN_z": -2.3},
        "vad_latency_ms": 1200,
        "discrepancies": {"D_va": 0.58, "D_vl": 0.50, "D_al": 0.47},
        "transcript": "Everything is great, honestly couldn't be happier with how the investigation concluded.",
        "priors": {"p_neural": 0.96, "p_xgb": 0.93},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.98, "summary": "Forced positive smile directly contradicting vocal shimmer instability and parasympathetic heart rate withdrawal."}
    },
    {
        "id": "D08_ImprovisedLie_CountermeasureStraining",
        "category": "High-Stakes Improvised Fabrication",
        "z_anomalies": {"Ocular_EAR_max_velocity_z": +2.8, "Autonomic_Facial_MicroTremor_z": +2.7, "Silero_VAD_Latency_z": +2.5},
        "vad_latency_ms": 1550,
        "discrepancies": {"D_va": 0.54, "D_vl": 0.53, "D_al": 0.51},
        "transcript": "I... am... absolutely... certain... of... every... single... detail... I previously reported.",
        "priors": {"p_neural": 0.98, "p_xgb": 0.97},
        "expected_verdict": {"classification": "Deceptive", "p_argon": 0.99, "summary": "Unnatural cadence with micro-tremor spikes, elevated blink burst velocities, and extreme response latency."}
    }
]

def format_exemplars_for_prompt(exemplars: List[Dict[str, Any]]) -> str:
    lines = ["\n--- 16 GOLD-STANDARD DECEPTCHAIN & LOSO EXEMPLARS (8 TRUTHFUL, 8 DECEPTIVE) ---"]
    for i, ex in enumerate(exemplars, 1):
        lines.append(f"\n[Exemplar {i:02d}: {ex['id']} ({ex['category']})]")
        lines.append(f"  Utterance: \"{ex['transcript']}\"")
        lines.append(f"  VAD Latency: {ex['vad_latency_ms']} ms | Cross-Modal Discrepancy: D_va={ex['discrepancies']['D_va']:.2f}, D_vl={ex['discrepancies']['D_vl']:.2f}")
        lines.append(f"  Priors: P_Neural={ex['priors']['p_neural']:.2f}, P_XGB={ex['priors']['p_xgb']:.2f}")
        lines.append(f"  Verdict: {ex['expected_verdict']['classification']} (P_Argon={ex['expected_verdict']['p_argon']:.2f})")
        lines.append(f"  Glass-Box Summary: {ex['expected_verdict']['summary']}")
    lines.append("\n--- END EXEMPLARS ---\n")
    return "\n".join(lines)

def get_16_gold_standard_exemplars() -> List[Dict[str, Any]]:
    """Returns the 16 curated gold-standard DeceptChain/LOSO exemplars."""
    return GOLD_STANDARD_16_EXEMPLARS

# 3. Forensic Engine System Prompt with Calibration Anchor & Curated Exemplars
SYSTEM_PROMPT_AND_16_EXEMPLARS = f"""You are the DeceptionX Cognitive Forensic Engine.
CRITICAL CALIBRATION ANCHOR (50/50 BASE RATE):
Exactly 50.0% of subjects in this environment are telling the TRUTH, including subjects experiencing high situational anxiety, embarrassment, or negative emotional valence (MU3D Negative Truth paradigm).
DO NOT equate isolated autonomic arousal (elevated heart rate, jitter, or facial tension) with deception.
A 'Deceptive' classification requires CONVERGENT CROSS-MODAL DISCREPANCY + COGNITIVE LOAD:
1. <observe>: Audit the 4-6 peak anomaly keyframes, the Top-30 within-subject Z-scores (|z| > 1.96), the VAD response latency, and the Branch 1 Cross-Modal Discrepancy scores (D_va, D_vl, D_al). Ignore AU20 unless corroborated by MediaPipe MAR/mouth-corner geometry.
2. <think>: Apply the Reality Monitoring (RM) framework and DeceptionX discrepancy rules:
   - External Memory (Truthful): Rich in sensory, perceptual, spatial, and temporal details; congruent affect between facial Action Units and vocal prosody, even when stressed (e.g., Negative Truth where AU04/AU07 matches negative vocal affect and coherent sensory reconstruction).
   - Internal Fabrication (Deceptive): Elevated cognitive operations, hedging, pronoun distancing, high VAD latency (> +1.8 SD), or stark channel contradiction (e.g., Duchenne-lacking polite smile AU12 without AU06 masking a +2.5 SD pitch/shimmer spike and forehead POS rPPG vasoconstriction).
3. <verdict>: Synthesize the final calibrated probability p_argon in [0.01, 0.99] integrating Branch 1 (P_Neural) and Branch 2 (P_XGBoost) out-of-fold priors.
{format_exemplars_for_prompt(GOLD_STANDARD_16_EXEMPLARS)}
"""

class Branch3CognitiveHarness:
    """
    Cognitive Reasoning Harness with asynchronous 3-tier circuit breaker.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.primary_model = "gemini-4-argon"
        self.fallback_model = "gemini-3.8-flash"
        self.argon_cache_name = None
        self.flash_cache_name = None
        
        if GENAI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def initialize_context_caches(self, exemplar_parts: list):
        """Creates persistent 2-hour caches for both primary and fallback models."""
        if not self.client or not GENAI_AVAILABLE:
            return
            
        try:
            self.argon_cache = self.client.caches.create(
                model=self.primary_model,
                config=types.CreateCachedContentConfig(
                    contents=[types.Content(role="user", parts=exemplar_parts)],
                    system_instruction=SYSTEM_PROMPT_AND_16_EXEMPLARS,
                    ttl="7200s",
                ),
            )
            self.argon_cache_name = self.argon_cache.name
        except Exception:
            self.argon_cache_name = None

        try:
            self.flash_cache = self.client.caches.create(
                model=self.fallback_model,
                config=types.CreateCachedContentConfig(
                    contents=[types.Content(role="user", parts=exemplar_parts)],
                    system_instruction=SYSTEM_PROMPT_AND_16_EXEMPLARS,
                    ttl="7200s",
                ),
            )
            self.flash_cache_name = self.flash_cache.name
        except Exception:
            self.flash_cache_name = None

    def _tier3_emergency_fallback(self, p_neural: float, p_xgb: float, reason: str = "Cloud API offline or unavailable") -> dict:
        """Tier-3 Total Offline Circuit Breaker: Local Branch 1 + Branch 2 Synthesis."""
        p_local = float(np_clip := max(0.01, min(0.99, 0.58 * p_neural + 0.42 * p_xgb)))
        classification = "Deceptive" if p_local >= 0.50 else "Truthful"
        return {
            "observe_trace": [f"{reason}; executing local deterministic neuro-tabular synthesis."],
            "think_trace": "Evaluated via RTX 5070 ReDA+DARE discrepancy head and LOSO TreeSHAP digest.",
            "classification": classification,
            "p_argon": p_local,
            "forensic_summary": "Verdict rendered via local neural-tabular ensemble under offline network protection.",
            "tier_used": "LOCAL_EMERGENCY",
        }

    async def evaluate_utterance(
        self,
        payload_parts: list,
        p_neural: float,
        p_xgb: float
    ) -> dict:
        """Executes gemini-4-argon with a strict 2.5s TTFT circuit breaker to gemini-3.8-flash, and Tier 3 emergency fallback."""
        if not self.client or not GENAI_AVAILABLE:
            return self._tier3_emergency_fallback(p_neural, p_xgb, "Google GenAI client uninitialized or offline")

        t0 = time.perf_counter()
        try:
            # gemini-4-argon strictly requires LOW, MEDIUM, or HIGH
            argon_config = types.GenerateContentConfig(
                cached_content=self.argon_cache_name,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=DeceptionXVerdictSchema,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW,
                    include_thoughts=True,
                ),
            )
            return await self._stream_with_ttft_guard(
                model=self.primary_model,
                parts=payload_parts,
                config=argon_config,
                ttft_limit_sec=2.5,
                t0=t0,
            )
        except (TimeoutError, Exception):
            # Automatic fallback to gemini-3.8-flash with MINIMAL thinking level
            try:
                flash_config = types.GenerateContentConfig(
                    cached_content=self.flash_cache_name,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=DeceptionXVerdictSchema,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL,
                        include_thoughts=False,
                    ),
                )
                return await asyncio.wait_for(
                    self._stream_with_ttft_guard(
                        model=self.fallback_model,
                        parts=payload_parts,
                        config=flash_config,
                        ttft_limit_sec=1.5,
                        t0=t0,
                    ),
                    timeout=2.2,
                )
            except Exception:
                return self._tier3_emergency_fallback(p_neural, p_xgb, "Cloud API timeout exceeded")

    async def _stream_with_ttft_guard(
        self,
        model: str,
        parts: list,
        config: Any,
        ttft_limit_sec: float,
        t0: float
    ) -> dict:
        first_chunk_received = False
        accumulated_json = ""
        stream = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=parts,
            config=config
        )
        async for chunk in stream:
            if not first_chunk_received:
                ttft = time.perf_counter() - t0
                if ttft > ttft_limit_sec:
                    raise TimeoutError(f"TTFT {ttft:.2f}s exceeded {ttft_limit_sec}s on {model}")
                first_chunk_received = True
            for part in chunk.candidates[0].content.parts:
                if part.text and not getattr(part, "thought", False):
                    accumulated_json += part.text

        parsed = DeceptionXVerdictSchema.model_validate_json(accumulated_json)
        result = parsed.model_dump()
        result["tier_used"] = model
        return result
