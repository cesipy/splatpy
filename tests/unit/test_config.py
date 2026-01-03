"""Unit tests for configuration classes."""
import pytest
from splatpy.config import TrainingConfig, QualityPreset, get_quality_preset
from gsplat.strategy import DefaultStrategy, MCMCStrategy


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_initialization(self):
        """Test creating TrainingConfig with defaults."""
        config = TrainingConfig()
        assert config.data_factor == 1
        assert config.batch_size == 1
        assert config.sh_degree == 3
        assert isinstance(config.strategy, DefaultStrategy)
        assert config.colmap_data_dir == ".splatpy_dir"

    def test_custom_initialization(self):
        """Test creating TrainingConfig with custom values."""
        config = TrainingConfig(
            data_factor=2,
            sh_degree=2,
            means_lr=1e-3,
            results_dir="custom_results/"
        )
        assert config.data_factor == 2
        assert config.sh_degree == 2
        assert config.means_lr == 1e-3
        assert config.results_dir == "custom_results/"

    def test_learning_rates_are_positive(self):
        """Test that all learning rates are positive."""
        config = TrainingConfig()
        assert config.means_lr > 0
        assert config.scales_lr > 0
        assert config.opacities_lr > 0
        assert config.quats_lr > 0
        assert config.sh0_lr > 0
        assert config.shN_lr > 0

    def test_mcmc_strategy(self):
        """Test initialization with MCMC strategy."""
        config = TrainingConfig(strategy=MCMCStrategy())
        assert isinstance(config.strategy, MCMCStrategy)


class TestQualityPreset:
    """Test QualityPreset class."""

    @pytest.mark.parametrize("quality,expected_steps,expected_modulo,expected_sh", [
        ("low", 10_000, 30, 2),
        ("medium", 30_000, 20, 3),
        ("high", 50_000, 15, 3),
        ("ultra", 100_000, 10, 3),
    ])
    def test_preset_parameters(self, quality, expected_steps, expected_modulo, expected_sh):
        """Test that presets have correct parameters."""
        preset = QualityPreset(quality)
        assert preset.steps == expected_steps
        assert preset.frames_modulo == expected_modulo
        assert preset.sh_degree == expected_sh
        assert preset.name == quality
        assert preset.data_factor == 1

    def test_invalid_quality_raises_error(self):
        """Test that invalid quality raises ValueError."""
        with pytest.raises(ValueError, match="Quality must be one of"):
            QualityPreset("invalid")

    def test_preset_has_description(self):
        """Test that presets have descriptions."""
        preset = QualityPreset("medium")
        assert isinstance(preset.description, str)
        assert len(preset.description) > 0

    def test_preset_not_available(self):

        with pytest.raises(ValueError):
            preset = QualityPreset("asdf")


    def test_quality_case_sensitivity(self):
        """Test that quality strings are case-sensitive."""
        # Should work with lowercase
        preset = QualityPreset("low")
        assert preset.name == "low"

        with pytest.raises(ValueError):
            QualityPreset("LOW")


class TestGetQualityPreset:
    """Test get_quality_preset function."""

    def test_returns_quality_preset(self):
        """Test that function returns QualityPreset instance."""
        preset = get_quality_preset("medium")
        assert isinstance(preset, QualityPreset)
        assert preset.name == "medium"

    @pytest.mark.parametrize("quality", ["low", "medium", "high", "ultra"])
    def test_all_presets(self, quality):
        """Test that all quality presets can be retrieved."""
        preset = get_quality_preset(quality)
        assert preset.name == quality
