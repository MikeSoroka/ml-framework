using MLFramework.Optimizers.MixedPrecision;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for DynamicLossScaler class
/// </summary>
public class DynamicLossScalerTests
{
    [Fact]
    public void Constructor_WithOptions_CreatesScaler()
    {
        var options = MixedPrecisionOptions.ForFP16();
        var scaler = new DynamicLossScaler(options);

        Assert.NotNull(scaler);
        Assert.True(scaler.IsEnabled);
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new DynamicLossScaler(null!));
    }

    [Fact]
    public void Constructor_WithDefaultOptions_CreatesScaler()
    {
        var scaler = new DynamicLossScaler();

        Assert.NotNull(scaler);
        Assert.True(scaler.IsEnabled);
    }

    [Fact]
    public void CurrentScale_InitializesToInitialLossScale()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 1000.0f;
        var scaler = new DynamicLossScaler(options);

        Assert.Equal(1000.0f, scaler.CurrentScale);
    }

    [Fact]
    public void ConsecutiveOverflows_InitializesToZero()
    {
        var scaler = new DynamicLossScaler();

        Assert.Equal(0, scaler.ConsecutiveOverflows);
    }

    [Fact]
    public void StepsSinceLastOverflow_InitializesToZero()
    {
        var scaler = new DynamicLossScaler();

        Assert.Equal(0, scaler.StepsSinceLastOverflow);
    }

    [Fact]
    public void TotalOverflows_InitializesToZero()
    {
        var scaler = new DynamicLossScaler();

        Assert.Equal(0, scaler.TotalOverflows);
    }

    [Fact]
    public void IsEnabled_ReflectsOptions()
    {
        var options = new MixedPrecisionOptions
        {
            EnableDynamicLossScaling = true
        };
        var enabledScaler = new DynamicLossScaler(options);

        options.EnableDynamicLossScaling = false;
        var disabledScaler = new DynamicLossScaler(options);

        Assert.True(enabledScaler.IsEnabled);
        Assert.False(disabledScaler.IsEnabled);
    }

    [Fact]
    public void ScaleLoss_WithEnabled_ScalesLoss()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 2.0f;
        var scaler = new DynamicLossScaler(options);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var scaled = scaler.ScaleLoss(tensor);

        Assert.NotNull(scaled);
    }

    [Fact]
    public void ScaleLoss_WithDisabled_ReturnsOriginal()
    {
        var options = new MixedPrecisionOptions
        {
            EnableDynamicLossScaling = false
        };
        var scaler = new DynamicLossScaler(options);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var scaled = scaler.ScaleLoss(tensor);

        Assert.Equal(tensor, scaled);
    }

    [Fact]
    public void ScaleLoss_WithNullTensor_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.ScaleLoss(null!));
    }

    [Fact]
    public void UnscaleGradients_WithEnabled_UnscalesGradients()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 2.0f;
        var scaler = new DynamicLossScaler(options);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 10.0f;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        var unscaled = scaler.UnscaleGradients(gradients);

        Assert.NotNull(unscaled);
        Assert.Single(unscaled);
    }

    [Fact]
    public void UnscaleGradients_WithDisabled_ReturnsOriginal()
    {
        var options = new MixedPrecisionOptions
        {
            EnableDynamicLossScaling = false
        };
        var scaler = new DynamicLossScaler(options);
        var tensor = new Tensor(new[] { 1 });
        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        var unscaled = scaler.UnscaleGradients(gradients);

        Assert.Equal(gradients, unscaled);
    }

    [Fact]
    public void UnscaleGradients_WithNullGradients_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.UnscaleGradients(null!));
    }

    [Fact]
    public void CheckOverflow_WithInf_ReturnsTrue()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = float.PositiveInfinity;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        Assert.True(scaler.CheckOverflow(gradients));
    }

    [Fact]
    public void CheckOverflow_WithNaN_ReturnsTrue()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = float.NaN;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        Assert.True(scaler.CheckOverflow(gradients));
    }

    [Fact]
    public void CheckOverflow_WithNormalValues_ReturnsFalse()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 2 });
        tensor[0] = 1.0f;
        tensor[1] = 2.0f;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        Assert.False(scaler.CheckOverflow(gradients));
    }

    [Fact]
    public void CheckOverflow_WithNullGradients_ThrowsArgumentNullException()
    {
        var scaler = new DynamicLossScaler();

        Assert.Throws<ArgumentNullException>(() =>
            scaler.CheckOverflow(null!));
    }

    [Fact]
    public void CheckOverflow_WithDisabled_ReturnsFalse()
    {
        var options = new MixedPrecisionOptions
        {
            EnableDynamicLossScaling = false
        };
        var scaler = new DynamicLossScaler(options);
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = float.PositiveInfinity;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        Assert.False(scaler.CheckOverflow(gradients));
    }

    [Fact]
    public void UpdateScale_Overflow_DecreasesScale()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 1000.0f;
        options.BackoffFactor = 0.5f;
        var scaler = new DynamicLossScaler(options);

        var skipStep = scaler.UpdateScale(hadOverflow: true);

        Assert.True(skipStep);
        Assert.Equal(500.0f, scaler.CurrentScale);
        Assert.Equal(1, scaler.TotalOverflows);
        Assert.Equal(1, scaler.ConsecutiveOverflows);
    }

    [Fact]
    public void UpdateScale_NoOverflow_IncreasesCounter()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.GrowthInterval = 2000;
        var scaler = new DynamicLossScaler(options);
        var initialScale = scaler.CurrentScale;

        for (int i = 0; i < 1999; i++)
        {
            scaler.UpdateScale(hadOverflow: false);
        }

        Assert.Equal(initialScale, scaler.CurrentScale);
        Assert.Equal(1999, scaler.StepsSinceLastOverflow);
        Assert.Equal(0, scaler.ConsecutiveOverflows);
    }

    [Fact]
    public void UpdateScale_AfterGrowthInterval_IncreasesScale()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 1000.0f;
        options.GrowthInterval = 2;
        options.GrowthFactor = 2.0f;
        var scaler = new DynamicLossScaler(options);

        scaler.UpdateScale(hadOverflow: false);
        scaler.UpdateScale(hadOverflow: false);

        Assert.Equal(2000.0f, scaler.CurrentScale);
        Assert.Equal(0, scaler.StepsSinceLastOverflow);
    }

    [Fact]
    public void UpdateScale_RespectsMinScale()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 100.0f;
        options.MinLossScale = 10.0f;
        options.BackoffFactor = 0.5f;
        var scaler = new DynamicLossScaler(options);

        scaler.UpdateScale(hadOverflow: true);
        scaler.UpdateScale(hadOverflow: true);
        scaler.UpdateScale(hadOverflow: true);
        scaler.UpdateScale(hadOverflow: true);

        Assert.Equal(10.0f, scaler.CurrentScale);
    }

    [Fact]
    public void UpdateScale_RespectsMaxScale()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 100.0f;
        options.MaxLossScale = 200.0f;
        options.GrowthInterval = 1;
        options.GrowthFactor = 10.0f;
        var scaler = new DynamicLossScaler(options);

        scaler.UpdateScale(hadOverflow: false);

        Assert.Equal(200.0f, scaler.CurrentScale);
    }

    [Fact]
    public void GrowthCounter_IncreasesOnNoOverflow()
    {
        var scaler = new DynamicLossScaler();

        scaler.UpdateScale(hadOverflow: false);

        Assert.Equal(1, scaler.StepsSinceLastOverflow);
    }

    [Fact]
    public void GrowthCounter_ResetsOnOverflow()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.GrowthInterval = 10;
        var scaler = new DynamicLossScaler(options);

        for (int i = 0; i < 5; i++)
        {
            scaler.UpdateScale(hadOverflow: false);
        }
        Assert.Equal(5, scaler.StepsSinceLastOverflow);

        scaler.UpdateScale(hadOverflow: true);

        Assert.Equal(0, scaler.StepsSinceLastOverflow);
    }

    [Fact]
    public void ConsecutiveOverflows_ResetsOnNoOverflow()
    {
        var scaler = new DynamicLossScaler();

        scaler.UpdateScale(hadOverflow: true);
        scaler.UpdateScale(hadOverflow: true);
        Assert.Equal(2, scaler.ConsecutiveOverflows);

        scaler.UpdateScale(hadOverflow: false);

        Assert.Equal(0, scaler.ConsecutiveOverflows);
    }

    [Fact]
    public void CheckOverflowAndUpdate_WorksWithOverflow()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = float.PositiveInfinity;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        var shouldSkip = scaler.CheckOverflowAndUpdate(gradients);

        Assert.True(shouldSkip);
    }

    [Fact]
    public void CheckOverflowAndUpdate_WorksWithoutOverflow()
    {
        var scaler = new DynamicLossScaler();
        var tensor = new Tensor(new[] { 1 });
        tensor[0] = 1.0f;

        var gradients = new System.Collections.Generic.Dictionary<string, Tensor>
        {
            ["param1"] = tensor
        };

        var shouldSkip = scaler.CheckOverflowAndUpdate(gradients);

        Assert.False(shouldSkip);
    }

    [Fact]
    public void Reset_RestoresInitialState()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 1000.0f;
        var scaler = new DynamicLossScaler(options);

        scaler.UpdateScale(hadOverflow: true);
        scaler.UpdateScale(hadOverflow: false);
        scaler.UpdateScale(hadOverflow: false);

        Assert.Equal(1, scaler.TotalOverflows);
        Assert.Equal(2, scaler.StepsSinceLastOverflow);

        scaler.Reset();

        Assert.Equal(1000.0f, scaler.CurrentScale);
        Assert.Equal(0, scaler.TotalOverflows);
        Assert.Equal(0, scaler.StepsSinceLastOverflow);
        Assert.Equal(0, scaler.ConsecutiveOverflows);
    }

    [Fact]
    public void GetStats_ReturnsCorrectStatistics()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 1000.0f;
        options.MaxConsecutiveOverflows = 10;
        var scaler = new DynamicLossScaler(options);

        scaler.UpdateScale(hadOverflow: false);
        scaler.UpdateScale(hadOverflow: false);

        var stats = scaler.GetStats();

        Assert.Equal(1000.0f, stats.CurrentScale);
        Assert.Equal(2, stats.StepsSinceLastOverflow);
        Assert.Equal(0, stats.ConsecutiveOverflows);
        Assert.Equal(0, stats.TotalOverflows);
        Assert.Equal(options.GrowthInterval, stats.GrowthInterval);
        Assert.Equal(10, stats.MaxConsecutiveOverflows);
        Assert.True(stats.IsStable);
    }

    [Fact]
    public void GetStats_WithOverflow_ReportsUnstable()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.MaxConsecutiveOverflows = 3;
        var scaler = new DynamicLossScaler(options);

        scaler.UpdateScale(hadOverflow: true);
        scaler.UpdateScale(hadOverflow: true);
        scaler.UpdateScale(hadOverflow: true);

        var stats = scaler.GetStats();

        Assert.False(stats.IsStable);
        Assert.Equal(3, stats.ConsecutiveOverflows);
    }
}
