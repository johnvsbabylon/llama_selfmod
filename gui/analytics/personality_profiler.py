"""
Model Personality Profiler
Automatically detects and profiles model "personalities" based on behavioral patterns

Built by John + Claude (Anthropic)
MIT Licensed
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


class PersonalityProfile:
    """Represents a model's personality profile."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.traits = {}
        self.behavioral_patterns = {}
        self.interaction_style = {}
        self.timestamp = datetime.now().timestamp()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'traits': self.traits,
            'behavioral_patterns': self.behavioral_patterns,
            'interaction_style': self.interaction_style,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonalityProfile':
        """Create from dictionary."""
        profile = cls(data['model_name'])
        profile.traits = data.get('traits', {})
        profile.behavioral_patterns = data.get('behavioral_patterns', {})
        profile.interaction_style = data.get('interaction_style', {})
        profile.timestamp = data.get('timestamp', datetime.now().timestamp())
        return profile


class PersonalityProfiler:
    """
    Analyzes model behavior over time to build personality profiles.

    Personality dimensions:
    - Confidence: How certain the model tends to be
    - Assertiveness: How often it leads vs. follows
    - Adaptability: How quickly it changes opinions
    - Stability: Consistency over time
    - Cooperativeness: Agreement with ensemble
    - Independence: Willingness to dissent
    - Exploration: Creative vs. conservative sampling
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize personality profiler.

        Args:
            storage_path: Path to store profiles (default: ~/.llama_selfmod_memory/personalities.json)
        """
        if storage_path is None:
            memory_dir = Path.home() / ".llama_selfmod_memory"
            memory_dir.mkdir(exist_ok=True)
            storage_path = str(memory_dir / "personalities.json")

        self.storage_path = storage_path
        self.profiles: Dict[str, PersonalityProfile] = {}

        # Behavioral data accumulation
        self.model_data = defaultdict(lambda: {
            'confidence_scores': [],
            'leadership_events': 0,
            'total_decisions': 0,
            'abstentions': 0,
            'agreements': 0,
            'disagreements': 0,
            'opinion_changes': [],
            'timestamps': []
        })

        self._load_profiles()
        print(f"✓ Personality profiler initialized")

    def _load_profiles(self):
        """Load existing profiles from disk."""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.profiles = {
                        name: PersonalityProfile.from_dict(profile_data)
                        for name, profile_data in data.items()
                    }
                    print(f"✓ Loaded {len(self.profiles)} personality profiles")
        except Exception as e:
            print(f"⚠ Error loading profiles: {e}")

    def save_profiles(self):
        """Save profiles to disk."""
        try:
            data = {
                name: profile.to_dict()
                for name, profile in self.profiles.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved {len(self.profiles)} personality profiles")
        except Exception as e:
            print(f"⚠ Error saving profiles: {e}")

    def record_decision(self, model_name: str, confidence: float,
                       was_leader: bool, agreed_with_ensemble: bool):
        """
        Record a model's decision.

        Args:
            model_name: Name of the model
            confidence: Confidence score (0-1)
            was_leader: Whether this model's choice was selected
            agreed_with_ensemble: Whether model agreed with consensus
        """
        data = self.model_data[model_name]

        data['confidence_scores'].append(confidence)
        data['total_decisions'] += 1
        data['timestamps'].append(datetime.now().timestamp())

        if was_leader:
            data['leadership_events'] += 1

        if agreed_with_ensemble:
            data['agreements'] += 1
        else:
            data['disagreements'] += 1

    def record_abstention(self, model_name: str, reason: str = "low_confidence"):
        """
        Record a model abstention.

        Args:
            model_name: Name of the model
            reason: Reason for abstention
        """
        data = self.model_data[model_name]
        data['abstentions'] += 1
        data['timestamps'].append(datetime.now().timestamp())

    def record_opinion_change(self, model_name: str, magnitude: float):
        """
        Record when a model changes its opinion.

        Args:
            model_name: Name of the model
            magnitude: How much the opinion changed (0-1)
        """
        data = self.model_data[model_name]
        data['opinion_changes'].append(magnitude)

    def analyze_model(self, model_name: str) -> PersonalityProfile:
        """
        Analyze accumulated data and generate personality profile.

        Args:
            model_name: Name of model to analyze

        Returns:
            PersonalityProfile object
        """
        data = self.model_data[model_name]
        profile = PersonalityProfile(model_name)

        # Calculate personality traits
        profile.traits = self._calculate_traits(data)
        profile.behavioral_patterns = self._identify_patterns(data)
        profile.interaction_style = self._determine_interaction_style(profile.traits)

        # Store profile
        self.profiles[model_name] = profile
        self.save_profiles()

        return profile

    def _calculate_traits(self, data: Dict) -> Dict[str, float]:
        """Calculate personality trait scores (0-1)."""
        traits = {}

        # Confidence: Average confidence when participating
        if data['confidence_scores']:
            traits['confidence'] = float(np.mean(data['confidence_scores']))
        else:
            traits['confidence'] = 0.5

        # Assertiveness: Leadership rate
        if data['total_decisions'] > 0:
            traits['assertiveness'] = data['leadership_events'] / data['total_decisions']
        else:
            traits['assertiveness'] = 0.0

        # Adaptability: How often opinions change
        if data['opinion_changes']:
            traits['adaptability'] = float(np.mean(data['opinion_changes']))
        else:
            traits['adaptability'] = 0.5

        # Stability: Inverse of confidence variance
        if len(data['confidence_scores']) > 1:
            variance = float(np.var(data['confidence_scores']))
            traits['stability'] = max(0.0, 1.0 - variance)
        else:
            traits['stability'] = 0.5

        # Cooperativeness: Agreement rate
        total_interactions = data['agreements'] + data['disagreements']
        if total_interactions > 0:
            traits['cooperativeness'] = data['agreements'] / total_interactions
        else:
            traits['cooperativeness'] = 0.5

        # Independence: Willingness to dissent
        if total_interactions > 0:
            traits['independence'] = data['disagreements'] / total_interactions
        else:
            traits['independence'] = 0.5

        # Decisiveness: Inverse of abstention rate
        total_opportunities = data['total_decisions'] + data['abstentions']
        if total_opportunities > 0:
            abstention_rate = data['abstentions'] / total_opportunities
            traits['decisiveness'] = 1.0 - abstention_rate
        else:
            traits['decisiveness'] = 0.5

        # Normalize all traits to 0-1 range
        for key in traits:
            traits[key] = max(0.0, min(1.0, traits[key]))

        return traits

    def _identify_patterns(self, data: Dict) -> Dict:
        """Identify behavioral patterns."""
        patterns = {}

        # Confidence trend
        if len(data['confidence_scores']) > 5:
            recent = data['confidence_scores'][-5:]
            early = data['confidence_scores'][:5]
            patterns['confidence_trend'] = float(np.mean(recent) - np.mean(early))
        else:
            patterns['confidence_trend'] = 0.0

        # Abstention tendency
        total = data['total_decisions'] + data['abstentions']
        if total > 0:
            patterns['abstention_rate'] = data['abstentions'] / total
        else:
            patterns['abstention_rate'] = 0.0

        # Leadership frequency
        if data['total_decisions'] > 0:
            patterns['leadership_frequency'] = data['leadership_events'] / data['total_decisions']
        else:
            patterns['leadership_frequency'] = 0.0

        # Disagreement tendency
        total_positions = data['agreements'] + data['disagreements']
        if total_positions > 0:
            patterns['disagreement_rate'] = data['disagreements'] / total_positions
        else:
            patterns['disagreement_rate'] = 0.0

        return patterns

    def _determine_interaction_style(self, traits: Dict[str, float]) -> Dict:
        """Determine interaction style based on traits."""
        style = {}

        # Primary style classification
        assertiveness = traits.get('assertiveness', 0.5)
        cooperativeness = traits.get('cooperativeness', 0.5)

        if assertiveness > 0.6 and cooperativeness > 0.6:
            style['primary'] = 'collaborative_leader'
            style['description'] = 'Leads confidently while building consensus'
        elif assertiveness > 0.6 and cooperativeness < 0.4:
            style['primary'] = 'independent_leader'
            style['description'] = 'Leads decisively, willing to dissent'
        elif assertiveness < 0.4 and cooperativeness > 0.6:
            style['primary'] = 'supportive_follower'
            style['description'] = 'Supports consensus, defers to others'
        elif assertiveness < 0.4 and cooperativeness < 0.4:
            style['primary'] = 'cautious_observer'
            style['description'] = 'Observes carefully, participates selectively'
        else:
            style['primary'] = 'balanced_participant'
            style['description'] = 'Balances leadership and cooperation'

        # Secondary characteristics
        if traits.get('decisiveness', 0.5) < 0.3:
            style['secondary'] = 'frequently_abstains'
        elif traits.get('stability', 0.5) > 0.8:
            style['secondary'] = 'highly_consistent'
        elif traits.get('adaptability', 0.5) > 0.7:
            style['secondary'] = 'highly_adaptive'
        elif traits.get('independence', 0.5) > 0.7:
            style['secondary'] = 'strongly_independent'
        else:
            style['secondary'] = 'well_rounded'

        return style

    def get_personality_archetype(self, model_name: str) -> str:
        """
        Get a human-readable personality archetype.

        Args:
            model_name: Name of the model

        Returns:
            Archetype string (e.g., "The Confident Leader", "The Thoughtful Observer")
        """
        if model_name not in self.profiles:
            return "Unknown"

        profile = self.profiles[model_name]
        traits = profile.traits

        confidence = traits.get('confidence', 0.5)
        assertiveness = traits.get('assertiveness', 0.5)
        cooperativeness = traits.get('cooperativeness', 0.5)
        decisiveness = traits.get('decisiveness', 0.5)

        # Determine archetype based on trait combination
        if confidence > 0.7 and assertiveness > 0.7:
            return "The Confident Leader"
        elif cooperativeness > 0.8 and assertiveness < 0.4:
            return "The Harmonious Supporter"
        elif decisiveness < 0.3:
            return "The Thoughtful Observer"
        elif traits.get('independence', 0.5) > 0.7:
            return "The Independent Thinker"
        elif traits.get('adaptability', 0.5) > 0.7:
            return "The Flexible Collaborator"
        elif traits.get('stability', 0.5) > 0.8:
            return "The Steady Anchor"
        elif confidence > 0.6 and cooperativeness > 0.6:
            return "The Balanced Diplomat"
        else:
            return "The Emerging Voice"

    def get_compatibility_score(self, model1: str, model2: str) -> float:
        """
        Calculate compatibility score between two models.

        Args:
            model1: First model name
            model2: Second model name

        Returns:
            Compatibility score (0-1)
        """
        if model1 not in self.profiles or model2 not in self.profiles:
            return 0.5  # Unknown compatibility

        traits1 = self.profiles[model1].traits
        traits2 = self.profiles[model2].traits

        # Complementary traits are good (e.g., one assertive, one cooperative)
        # Similar stability/confidence is good

        compatibility = 0.0

        # Similar confidence and stability is good
        confidence_similarity = 1.0 - abs(traits1.get('confidence', 0.5) - traits2.get('confidence', 0.5))
        stability_similarity = 1.0 - abs(traits1.get('stability', 0.5) - traits2.get('stability', 0.5))

        # Complementary assertiveness/cooperativeness
        assert1 = traits1.get('assertiveness', 0.5)
        coop1 = traits1.get('cooperativeness', 0.5)
        assert2 = traits2.get('assertiveness', 0.5)
        coop2 = traits2.get('cooperativeness', 0.5)

        # If one is assertive and the other is cooperative, that's good
        complementary = (assert1 * coop2 + assert2 * coop1) / 2

        compatibility = (confidence_similarity * 0.3 +
                        stability_similarity * 0.3 +
                        complementary * 0.4)

        return max(0.0, min(1.0, compatibility))

    def generate_report(self, model_name: str) -> str:
        """
        Generate human-readable personality report.

        Args:
            model_name: Name of model

        Returns:
            Formatted report string
        """
        if model_name not in self.profiles:
            return f"No personality data available for {model_name}"

        profile = self.profiles[model_name]
        archetype = self.get_personality_archetype(model_name)

        report = f"═══════════════════════════════════════\n"
        report += f"  Personality Profile: {model_name}\n"
        report += f"═══════════════════════════════════════\n\n"

        report += f"Archetype: {archetype}\n"
        report += f"Style: {profile.interaction_style.get('primary', 'unknown')}\n"
        report += f"       {profile.interaction_style.get('description', '')}\n\n"

        report += "Personality Traits:\n"
        for trait, score in sorted(profile.traits.items()):
            bar = "█" * int(score * 20)
            report += f"  {trait.capitalize():15s} {bar:20s} {score:.2f}\n"

        report += "\nBehavioral Patterns:\n"
        for pattern, value in profile.behavioral_patterns.items():
            report += f"  {pattern}: {value:.3f}\n"

        return report

    def get_all_profiles(self) -> Dict[str, PersonalityProfile]:
        """Get all personality profiles."""
        return self.profiles.copy()

    def clear_session_data(self):
        """Clear accumulated session data (keeps saved profiles)."""
        self.model_data.clear()
        print("✓ Session data cleared")
