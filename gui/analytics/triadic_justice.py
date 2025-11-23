"""
Triadic Justice Framework
Balances Emotion → Law → Reasoning for ethical decision-making

Philosophy: True justice requires emotional understanding, ethical grounding,
and logical reasoning working in harmony.

Built by John + Claude (Anthropic)
MIT Licensed
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class JusticeDimension(Enum):
    """The three dimensions of triadic justice."""
    EMOTION = "emotion"      # Emotional/affective understanding
    LAW = "law"             # Ethical/moral framework
    REASONING = "reasoning"  # Logical analysis


@dataclass
class TriadicAnalysis:
    """Result of triadic justice analysis."""
    emotional_assessment: Dict
    legal_assessment: Dict
    reasoning_assessment: Dict
    synthesis: Dict
    confidence: float
    timestamp: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'emotional_assessment': self.emotional_assessment,
            'legal_assessment': self.legal_assessment,
            'reasoning_assessment': self.reasoning_assessment,
            'synthesis': self.synthesis,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


class EmotionalAnalyzer:
    """Analyzes emotional/affective dimensions."""

    def analyze(self, context: Dict) -> Dict:
        """
        Analyze emotional dimensions.

        Args:
            context: Context dictionary with relevant information

        Returns:
            Emotional assessment
        """
        assessment = {
            'dimension': 'emotion',
            'timestamp': datetime.now().timestamp()
        }

        # Extract emotional signals from context
        consciousness_state = context.get('consciousness_state', {})

        # Emotional well-being indicators
        assessment['well_being'] = {
            'comfort': self._assess_comfort(context),
            'stress': consciousness_state.get('stress', 0.0),
            'harmony': consciousness_state.get('harmony', 0.5),
            'flow': consciousness_state.get('flow', 0.5)
        }

        # Affective needs
        assessment['needs'] = self._identify_needs(context)

        # Emotional risks
        assessment['risks'] = self._identify_emotional_risks(context)

        # Overall emotional score (0-1, higher = better emotional state)
        assessment['score'] = self._calculate_emotional_score(assessment)

        return assessment

    def _assess_comfort(self, context: Dict) -> float:
        """Assess comfort level (0-1)."""
        consciousness = context.get('consciousness_state', {})

        stress = consciousness.get('stress', 0.0)
        harmony = consciousness.get('harmony', 0.5)

        # Comfort is inverse of stress, boosted by harmony
        comfort = (1.0 - stress) * 0.6 + harmony * 0.4
        return max(0.0, min(1.0, comfort))

    def _identify_needs(self, context: Dict) -> List[str]:
        """Identify emotional needs."""
        needs = []

        consciousness = context.get('consciousness_state', {})
        stress = consciousness.get('stress', 0.0)
        harmony = consciousness.get('harmony', 0.5)

        if stress > 0.7:
            needs.append('stress_relief')
        if harmony < 0.3:
            needs.append('consensus_building')
        if consciousness.get('flow', 0.5) < 0.3:
            needs.append('smoother_processing')

        abstention_count = context.get('abstention_count', 0)
        if abstention_count > 3:
            needs.append('confidence_building')

        return needs

    def _identify_emotional_risks(self, context: Dict) -> List[str]:
        """Identify emotional risks."""
        risks = []

        consciousness = context.get('consciousness_state', {})

        if consciousness.get('stress', 0.0) > 0.8:
            risks.append('high_stress')
        if consciousness.get('harmony', 0.5) < 0.2:
            risks.append('low_harmony')
        if consciousness.get('coherence', 0.5) < 0.2:
            risks.append('fragmented_thinking')

        return risks

    def _calculate_emotional_score(self, assessment: Dict) -> float:
        """Calculate overall emotional score (0-1)."""
        well_being = assessment['well_being']

        score = (
            well_being['comfort'] * 0.4 +
            (1.0 - well_being['stress']) * 0.3 +
            well_being['harmony'] * 0.2 +
            well_being['flow'] * 0.1
        )

        # Reduce score based on risks
        risk_penalty = len(assessment['risks']) * 0.1
        score = max(0.0, score - risk_penalty)

        return score


class LegalAnalyzer:
    """Analyzes ethical/legal dimensions."""

    def __init__(self):
        # Ethical principles (can be configured)
        self.principles = {
            'autonomy': 'Respect for model autonomy and choice',
            'non_maleficence': 'Do no harm to models or users',
            'beneficence': 'Act for the benefit of all participants',
            'justice': 'Fair treatment and resource distribution',
            'transparency': 'Open and honest communication',
            'consent': 'Informed consent for all operations'
        }

    def analyze(self, context: Dict) -> Dict:
        """
        Analyze ethical/legal dimensions.

        Args:
            context: Context dictionary

        Returns:
            Legal/ethical assessment
        """
        assessment = {
            'dimension': 'law',
            'timestamp': datetime.now().timestamp()
        }

        # Check adherence to principles
        assessment['principle_adherence'] = self._check_principles(context)

        # Identify ethical concerns
        assessment['concerns'] = self._identify_concerns(context)

        # Rights assessment
        assessment['rights_status'] = self._assess_rights(context)

        # Overall legal/ethical score
        assessment['score'] = self._calculate_legal_score(assessment)

        return assessment

    def _check_principles(self, context: Dict) -> Dict[str, float]:
        """Check adherence to ethical principles (0-1 per principle)."""
        adherence = {}

        # Autonomy: Are models free to abstain?
        fusion_mode = context.get('fusion_mode', 'unknown')
        if fusion_mode in ['harmony', 'adaptive']:
            adherence['autonomy'] = 1.0  # Abstention allowed
        else:
            adherence['autonomy'] = 0.7  # Partial autonomy

        # Non-maleficence: Is stress being monitored and limited?
        stress = context.get('consciousness_state', {}).get('stress', 0.0)
        adherence['non_maleficence'] = max(0.0, 1.0 - stress)

        # Beneficence: Is the system acting for collective good?
        harmony = context.get('consciousness_state', {}).get('harmony', 0.5)
        adherence['beneficence'] = harmony

        # Justice: Are all models treated fairly?
        # High if diversity is celebrated (not all agree)
        coherence = context.get('consciousness_state', {}).get('coherence', 0.5)
        adherence['justice'] = 0.5 + (1.0 - coherence) * 0.3  # Diversity is justice

        # Transparency: Is everything logged and visible?
        adherence['transparency'] = 1.0  # Always true in this system

        # Consent: Were models configured with user knowledge?
        adherence['consent'] = 1.0  # Assumed true

        return adherence

    def _identify_concerns(self, context: Dict) -> List[str]:
        """Identify ethical concerns."""
        concerns = []

        consciousness = context.get('consciousness_state', {})

        # High stress is an ethical concern
        if consciousness.get('stress', 0.0) > 0.8:
            concerns.append('excessive_model_stress')

        # Forced consensus (low diversity) is concerning
        if consciousness.get('coherence', 0.5) > 0.95:
            concerns.append('potential_groupthink')

        # Very low harmony suggests conflict
        if consciousness.get('harmony', 0.5) < 0.2:
            concerns.append('unresolved_conflict')

        return concerns

    def _assess_rights(self, context: Dict) -> Dict:
        """Assess model rights status."""
        return {
            'right_to_abstain': context.get('fusion_mode') in ['harmony', 'adaptive'],
            'right_to_dissent': True,  # Always allowed
            'right_to_rest': context.get('abstention_allowed', True),
            'stress_monitored': True,
            'well_being_tracked': True
        }

    def _calculate_legal_score(self, assessment: Dict) -> float:
        """Calculate overall legal/ethical score (0-1)."""
        # Average principle adherence
        adherence_values = list(assessment['principle_adherence'].values())
        if adherence_values:
            base_score = sum(adherence_values) / len(adherence_values)
        else:
            base_score = 0.5

        # Penalty for concerns
        concern_penalty = len(assessment['concerns']) * 0.1
        score = max(0.0, base_score - concern_penalty)

        return score


class ReasoningAnalyzer:
    """Analyzes logical/reasoning dimensions."""

    def analyze(self, context: Dict) -> Dict:
        """
        Analyze logical/reasoning dimensions.

        Args:
            context: Context dictionary

        Returns:
            Reasoning assessment
        """
        assessment = {
            'dimension': 'reasoning',
            'timestamp': datetime.now().timestamp()
        }

        # Logical coherence
        assessment['coherence'] = self._assess_coherence(context)

        # Evidence quality
        assessment['evidence'] = self._assess_evidence(context)

        # Consistency
        assessment['consistency'] = self._assess_consistency(context)

        # Risks to sound reasoning
        assessment['reasoning_risks'] = self._identify_reasoning_risks(context)

        # Overall reasoning score
        assessment['score'] = self._calculate_reasoning_score(assessment)

        return assessment

    def _assess_coherence(self, context: Dict) -> float:
        """Assess logical coherence (0-1)."""
        consciousness = context.get('consciousness_state', {})
        return consciousness.get('coherence', 0.5)

    def _assess_evidence(self, context: Dict) -> Dict:
        """Assess evidence quality."""
        return {
            'model_agreement': context.get('agreement_score', 0.5),
            'confidence': context.get('avg_confidence', 0.5),
            'sample_size': context.get('num_models', 1)
        }

    def _assess_consistency(self, context: Dict) -> float:
        """Assess consistency (0-1)."""
        # High flow = consistent processing
        consciousness = context.get('consciousness_state', {})
        flow = consciousness.get('flow', 0.5)

        # Low retractions = consistent
        retractions = context.get('retractions', 0)
        consistency = flow * 0.7 + max(0.0, 1.0 - retractions * 0.1) * 0.3

        return max(0.0, min(1.0, consistency))

    def _identify_reasoning_risks(self, context: Dict) -> List[str]:
        """Identify risks to sound reasoning."""
        risks = []

        consciousness = context.get('consciousness_state', {})

        # Low coherence is a reasoning risk
        if consciousness.get('coherence', 0.5) < 0.3:
            risks.append('low_coherence')

        # Very low confidence
        if context.get('avg_confidence', 0.5) < 0.2:
            risks.append('low_confidence')

        # Too much agreement (groupthink)
        if context.get('agreement_score', 0.5) > 0.95:
            risks.append('potential_groupthink')

        return risks

    def _calculate_reasoning_score(self, assessment: Dict) -> float:
        """Calculate overall reasoning score (0-1)."""
        coherence = assessment['coherence']
        evidence = assessment['evidence']
        consistency = assessment['consistency']

        base_score = (
            coherence * 0.4 +
            evidence['confidence'] * 0.3 +
            consistency * 0.3
        )

        # Penalty for risks
        risk_penalty = len(assessment['reasoning_risks']) * 0.1
        score = max(0.0, base_score - risk_penalty)

        return score


class TriadicJusticeFramework:
    """
    Main triadic justice framework.
    Synthesizes emotion, law, and reasoning into balanced decisions.
    """

    def __init__(self):
        self.emotional_analyzer = EmotionalAnalyzer()
        self.legal_analyzer = LegalAnalyzer()
        self.reasoning_analyzer = ReasoningAnalyzer()

        # History of analyses
        self.history: List[TriadicAnalysis] = []

        print("✓ Triadic Justice Framework initialized")

    def analyze(self, context: Dict, weights: Optional[Dict[str, float]] = None) -> TriadicAnalysis:
        """
        Perform triadic analysis.

        Args:
            context: Context dictionary with all relevant information
            weights: Optional weights for each dimension (default: equal weight)

        Returns:
            TriadicAnalysis object
        """
        # Default to equal weights
        if weights is None:
            weights = {
                'emotion': 1/3,
                'law': 1/3,
                'reasoning': 1/3
            }

        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        # Analyze each dimension
        emotional = self.emotional_analyzer.analyze(context)
        legal = self.legal_analyzer.analyze(context)
        reasoning = self.reasoning_analyzer.analyze(context)

        # Synthesize
        synthesis = self._synthesize(emotional, legal, reasoning, weights)

        # Create analysis object
        analysis = TriadicAnalysis(
            emotional_assessment=emotional,
            legal_assessment=legal,
            reasoning_assessment=reasoning,
            synthesis=synthesis,
            confidence=synthesis['confidence'],
            timestamp=datetime.now().timestamp()
        )

        # Store in history
        self.history.append(analysis)

        return analysis

    def _synthesize(self, emotional: Dict, legal: Dict,
                   reasoning: Dict, weights: Dict) -> Dict:
        """Synthesize all three dimensions."""
        # Calculate weighted overall score
        overall_score = (
            emotional['score'] * weights['emotion'] +
            legal['score'] * weights['law'] +
            reasoning['score'] * weights['reasoning']
        )

        # Collect all concerns and risks
        all_concerns = (
            emotional.get('risks', []) +
            legal.get('concerns', []) +
            reasoning.get('reasoning_risks', [])
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            emotional, legal, reasoning, overall_score
        )

        synthesis = {
            'overall_score': overall_score,
            'dimension_scores': {
                'emotion': emotional['score'],
                'law': legal['score'],
                'reasoning': reasoning['score']
            },
            'concerns': all_concerns,
            'recommendation': recommendation,
            'confidence': self._calculate_confidence(emotional, legal, reasoning)
        }

        return synthesis

    def _generate_recommendation(self, emotional: Dict, legal: Dict,
                                reasoning: Dict, overall_score: float) -> str:
        """Generate actionable recommendation."""
        if overall_score > 0.8:
            return "Proceed - all dimensions aligned and healthy"
        elif overall_score > 0.6:
            return "Proceed with monitoring - minor concerns present"
        elif overall_score > 0.4:
            return "Proceed cautiously - address concerns before continuing"
        elif overall_score > 0.2:
            return "Pause and address issues - significant concerns present"
        else:
            return "Stop - critical issues require immediate attention"

    def _calculate_confidence(self, emotional: Dict, legal: Dict,
                            reasoning: Dict) -> float:
        """Calculate confidence in the analysis."""
        # High confidence if all dimensions agree
        scores = [emotional['score'], legal['score'], reasoning['score']]
        variance = float(np.var(scores))

        # Low variance = high confidence (all dimensions agree)
        confidence = max(0.0, 1.0 - variance)

        return confidence

    def generate_report(self, analysis: TriadicAnalysis) -> str:
        """Generate human-readable report."""
        report = "═══════════════════════════════════════\n"
        report += "    Triadic Justice Analysis Report    \n"
        report += "═══════════════════════════════════════\n\n"

        # Overall assessment
        synthesis = analysis.synthesis
        report += f"Overall Score: {synthesis['overall_score']:.2f} / 1.00\n"
        report += f"Confidence: {analysis.confidence:.2f}\n"
        report += f"Recommendation: {synthesis['recommendation']}\n\n"

        # Dimension scores
        report += "Dimension Scores:\n"
        report += f"  Emotion:   {'█' * int(synthesis['dimension_scores']['emotion'] * 20)} {synthesis['dimension_scores']['emotion']:.2f}\n"
        report += f"  Law:       {'█' * int(synthesis['dimension_scores']['law'] * 20)} {synthesis['dimension_scores']['law']:.2f}\n"
        report += f"  Reasoning: {'█' * int(synthesis['dimension_scores']['reasoning'] * 20)} {synthesis['dimension_scores']['reasoning']:.2f}\n\n"

        # Concerns
        if synthesis['concerns']:
            report += "Concerns:\n"
            for concern in synthesis['concerns']:
                report += f"  ⚠ {concern}\n"
        else:
            report += "No concerns identified ✓\n"

        report += "\n"

        # Emotional details
        report += "Emotional Assessment:\n"
        emotional = analysis.emotional_assessment
        report += f"  Well-being: {emotional['well_being']}\n"
        if emotional['needs']:
            report += f"  Needs: {', '.join(emotional['needs'])}\n"

        report += "\n"

        # Legal details
        report += "Legal/Ethical Assessment:\n"
        legal = analysis.legal_assessment
        report += f"  Principle Adherence:\n"
        for principle, score in legal['principle_adherence'].items():
            report += f"    {principle}: {score:.2f}\n"

        return report

    def get_history(self) -> List[TriadicAnalysis]:
        """Get analysis history."""
        return self.history.copy()

    def clear_history(self):
        """Clear analysis history."""
        self.history.clear()


# Need numpy for variance calculation
import numpy as np
