"""
SIMA-2 Trajectory Generator.

Batch generation of demonstration trajectories for training.
"""

import os
import json
import numpy as np
from typing import List

from .co_agent import SIMACoAgent, SIMADataCollector


class TrajectoryGenerator:
    """
    High-throughput trajectory generation for SIMA-2.

    Generates diverse demonstrations for:
    - VLA transformer training
    - HRL imitation learning
    - Behavioral cloning
    """

    def __init__(self, env_factory=None):
        """
        Args:
            env_factory: Callable that creates environment instances
        """
        self.env_factory = env_factory
        self.generated_trajectories = []

    def generate_dataset(
        self,
        n_trajectories=1000,
        instruction_diversity=0.5,
        skill_diversity=0.3,
        param_noise=0.1,
        save_path=None
    ):
        """
        Generate complete training dataset.

        Args:
            n_trajectories: Number of trajectories to generate
            instruction_diversity: How much to vary instructions [0, 1]
            skill_diversity: How much to vary skill sequences [0, 1]
            param_noise: Noise added to skill parameters [0, 1]
            save_path: Optional path to save dataset

        Returns:
            dataset: List of trajectory samples
        """
        print(f"Generating {n_trajectories} SIMA trajectories...")

        if self.env_factory is None:
            # Generate synthetic data without environment
            return self._generate_synthetic_dataset(
                n_trajectories, instruction_diversity, skill_diversity, param_noise
            )

        # Generate with actual environment
        env = self.env_factory()
        collector = SIMADataCollector(env)

        # Generate diverse instructions
        instructions = self._generate_diverse_instructions(
            n_trajectories, instruction_diversity
        )

        # Collect trajectories
        collector.collect_batch(instructions, verbose=True)

        # Get training data
        dataset = collector.get_vla_training_data()

        if save_path:
            collector.save_trajectories(save_path)

        env.close()

        self.generated_trajectories = collector.trajectories
        return dataset

    def _generate_diverse_instructions(self, n, diversity):
        """
        Generate diverse instruction set.

        Args:
            n: Number of instructions
            diversity: Diversity level [0, 1]

        Returns:
            instructions: List of instruction strings
        """
        base_templates = [
            "open the drawer without hitting the vase",
            "carefully open the top drawer",
            "pull the drawer open while avoiding fragile objects",
            "open the cabinet drawer safely",
            "grasp the handle and open the drawer",
        ]

        modifiers = [
            ("carefully", "quickly", "slowly", "gently", "safely"),
            ("without hitting", "while avoiding", "keeping clear of", "not touching"),
            ("the vase", "the fragile vase", "the ceramic vase", "fragile objects"),
        ]

        instructions = []

        for i in range(n):
            if np.random.random() < diversity:
                # High diversity: combine templates and modifiers
                base = np.random.choice(base_templates)

                # Random modifier insertions
                if np.random.random() < diversity:
                    mod = np.random.choice(modifiers[0])
                    base = mod + " " + base

                # Synonym replacements
                if np.random.random() < diversity and "vase" in base:
                    replacement = np.random.choice(modifiers[2])
                    base = base.replace("vase", replacement)

                instructions.append(base)
            else:
                # Low diversity: use base templates
                instructions.append(np.random.choice(base_templates))

        return instructions

    def _generate_synthetic_dataset(self, n, inst_div, skill_div, param_noise):
        """
        Generate synthetic dataset without environment.

        Args:
            n: Number of samples
            inst_div: Instruction diversity
            skill_div: Skill sequence diversity
            param_noise: Parameter noise level

        Returns:
            dataset: List of training samples
        """
        from src.hrl.skills import SkillID, SkillParams

        dataset = []

        for i in range(n):
            # Generate instruction
            instruction = self._sample_instruction(inst_div)

            # Generate skill sequence
            skill_sequence = self._sample_skill_sequence(skill_div)

            # Generate parameters with noise
            skill_params = []
            for sid in skill_sequence:
                base_params = SkillParams.default_for_skill(sid).to_array()
                noisy_params = base_params + np.random.randn(5) * param_noise * 0.1
                noisy_params = np.clip(noisy_params, 0, 1)
                skill_params.append(noisy_params.astype(np.float32))

            sample = {
                'instruction': instruction,
                'skill_sequence': skill_sequence,
                'skill_params': skill_params,
            }

            # Add optional features
            if np.random.random() < 0.3:
                sample['z_v'] = np.random.randn(128).astype(np.float32) * 0.1
                sample['state'] = np.random.randn(13).astype(np.float32) * 0.1

            dataset.append(sample)

        return dataset

    def _sample_instruction(self, diversity):
        """Sample instruction with given diversity."""
        templates = [
            "open the drawer",
            "open the drawer without hitting the vase",
            "carefully open the top drawer",
            "pull open the drawer while maintaining clearance",
            "grasp handle and open drawer safely",
        ]

        if np.random.random() < diversity:
            # Add random modifiers
            base = np.random.choice(templates)
            if np.random.random() < 0.5:
                prefix = np.random.choice(["please", "robot,", "i need you to", ""])
                base = prefix + " " + base if prefix else base
            return base.strip()
        else:
            return np.random.choice(templates)

    def _sample_skill_sequence(self, diversity):
        """Sample skill sequence with given diversity."""
        from src.hrl.skills import SkillID

        standard_sequence = [
            SkillID.LOCATE_DRAWER,
            SkillID.LOCATE_VASE,
            SkillID.PLAN_SAFE_APPROACH,
            SkillID.GRASP_HANDLE,
            SkillID.OPEN_WITH_CLEARANCE,
            SkillID.RETRACT_SAFE,
        ]

        if np.random.random() < diversity:
            # Vary sequence
            sequence = standard_sequence.copy()

            # Maybe skip LOCATE_VASE
            if np.random.random() < 0.3:
                sequence.remove(SkillID.LOCATE_VASE)

            # Maybe skip RETRACT_SAFE
            if np.random.random() < 0.2:
                sequence.remove(SkillID.RETRACT_SAFE)

            return sequence
        else:
            return standard_sequence

    def generate_curriculum_dataset(self, levels=3, samples_per_level=100):
        """
        Generate curriculum learning dataset with increasing difficulty.

        Args:
            levels: Number of difficulty levels
            samples_per_level: Samples per level

        Returns:
            curriculum: List of (level, dataset) tuples
        """
        curriculum = []

        for level in range(levels):
            difficulty = level / (levels - 1)  # 0 to 1

            # Adjust parameters based on difficulty
            inst_div = 0.2 + difficulty * 0.6
            skill_div = 0.1 + difficulty * 0.4
            param_noise = 0.05 + difficulty * 0.15

            dataset = self._generate_synthetic_dataset(
                samples_per_level, inst_div, skill_div, param_noise
            )

            curriculum.append({
                'level': level,
                'difficulty': difficulty,
                'dataset': dataset,
                'instruction_diversity': inst_div,
                'skill_diversity': skill_div,
                'parameter_noise': param_noise,
            })

        return curriculum

    def export_for_vla_training(self, save_dir='data/vla_training'):
        """
        Export generated data in VLA training format.

        Args:
            save_dir: Directory to save data

        Returns:
            paths: dict with saved file paths
        """
        os.makedirs(save_dir, exist_ok=True)

        # Convert to serializable format
        train_data = []
        for traj in self.generated_trajectories:
            sample = traj.get_vla_training_sample()
            # Convert numpy to list
            sample['skill_params'] = [p.tolist() for p in sample['skill_params']]
            train_data.append(sample)

        # Save main dataset
        train_path = os.path.join(save_dir, 'train_data.json')
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)

        # Save narrations separately
        narrations_data = [
            {
                'instruction': traj.instruction,
                'narrations': traj.narrations,
                'skill_ids': traj.skill_ids,
            }
            for traj in self.generated_trajectories
        ]

        narrations_path = os.path.join(save_dir, 'narrations.json')
        with open(narrations_path, 'w') as f:
            json.dump(narrations_data, f, indent=2)

        # Save statistics
        stats = {
            'n_trajectories': len(self.generated_trajectories),
            'success_rate': sum(t.success for t in self.generated_trajectories) / max(1, len(self.generated_trajectories)),
            'mean_steps': np.mean([t.total_steps for t in self.generated_trajectories]) if self.generated_trajectories else 0,
            'unique_instructions': len(set(t.instruction for t in self.generated_trajectories)),
        }

        stats_path = os.path.join(save_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return {
            'train_data': train_path,
            'narrations': narrations_path,
            'statistics': stats_path,
        }

    def augment_with_language_variations(self, dataset):
        """
        Augment dataset with instruction paraphrases.

        Args:
            dataset: List of training samples

        Returns:
            augmented: Augmented dataset
        """
        narrator = SIMACoAgent().narrator
        augmented = []

        for sample in dataset:
            # Add original
            augmented.append(sample)

            # Generate variations
            variations = narrator.generate_instruction_variations(sample['instruction'])

            for var in variations[:3]:  # Add up to 3 variations
                if var != sample['instruction']:
                    new_sample = sample.copy()
                    new_sample['instruction'] = var
                    augmented.append(new_sample)

        return augmented


def create_sima_training_pipeline(env_factory, n_trajectories=1000, save_dir='data/sima'):
    """
    Complete pipeline for SIMA demonstration generation.

    Args:
        env_factory: Factory for creating environments
        n_trajectories: Number of demonstrations
        save_dir: Directory to save results

    Returns:
        results: dict with paths and statistics
    """
    print("=" * 70)
    print("SIMA-2 DEMONSTRATION GENERATION PIPELINE")
    print("=" * 70)

    # Create generator
    generator = TrajectoryGenerator(env_factory)

    # Generate dataset
    dataset = generator.generate_dataset(
        n_trajectories=n_trajectories,
        instruction_diversity=0.6,
        skill_diversity=0.3,
        param_noise=0.1,
        save_path=os.path.join(save_dir, 'raw_trajectories.json')
    )

    # Export for VLA training
    paths = generator.export_for_vla_training(save_dir)

    # Augment with language variations
    augmented = generator.augment_with_language_variations(dataset)

    # Save augmented
    aug_path = os.path.join(save_dir, 'augmented_data.json')
    with open(aug_path, 'w') as f:
        json.dump([
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()}
            for s in augmented
        ], f, indent=2)

    paths['augmented_data'] = aug_path

    print(f"\nGeneration complete!")
    print(f"Original samples: {len(dataset)}")
    print(f"Augmented samples: {len(augmented)}")

    return {
        'paths': paths,
        'n_original': len(dataset),
        'n_augmented': len(augmented),
    }
