"""Unit tests for Wiener deconvolution.

Note on Wiener deconvolution limitations:
- Ringing artifacts occur when the source signal has limited bandwidth (e.g., Gaussian)
  because division by small |S(f)|^2 at high frequencies amplifies noise.
- Higher regularization reduces ringing but blurs the result.
- For best results, use a source with broad frequency content (e.g., impulse, chirp).

Run with visualization:
    pytest test_deconvolution.py -v -k "visual" --tb=no
    
Run without visualization (faster):
    pytest test_deconvolution.py -v -k "not visual"
"""

import numpy as np
import scipy.signal
import pytest
from fdtd import wiener_deconvolve


def convolve_same_length(source: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve source with IR, returning same length as inputs.
    
    Uses scipy.signal.convolve with mode='full', then trims to match
    the input length (keeping the causal part starting at t=0).
    """
    full = scipy.signal.convolve(source, ir, mode='full')
    # Keep first n samples (causal output)
    return full[:len(source)]


def plot_deconvolution_result(t, s, h_true, g, h_recovered, title="Deconvolution Result", 
                               filename=None):
    """Plot source, true IR, recorded signal, and recovered IR for visual inspection.
    
    Args:
        filename: If provided, save to this file instead of showing interactively.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    # Use non-interactive backend if no display available
    if filename or not _has_display():
        matplotlib.use('Agg')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)
    
    # Source
    axes[0, 0].plot(t, s, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Source Signal s(t)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # True IR
    axes[0, 1].plot(t, h_true, 'g-', linewidth=1.5)
    axes[0, 1].set_title('True Impulse Response h(t)')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Recorded signal
    axes[1, 0].plot(t, g, 'r-', linewidth=1.5)
    axes[1, 0].set_title('Recorded Signal g(t) = s * h')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comparison: True vs Recovered IR
    h_true_norm = h_true / np.max(np.abs(h_true)) if np.max(np.abs(h_true)) > 0 else h_true
    h_rec_norm = h_recovered / np.max(np.abs(h_recovered)) if np.max(np.abs(h_recovered)) > 0 else h_recovered
    
    axes[1, 1].plot(t, h_true_norm, 'g-', linewidth=2, label='True IR', alpha=0.7)
    axes[1, 1].plot(t, h_rec_norm, 'b--', linewidth=1.5, label='Recovered IR')
    axes[1, 1].set_title('True vs Recovered IR (normalized)')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close(fig)
    elif _has_display():
        plt.show()
    else:
        # No display, save to default file
        import os
        os.makedirs("output", exist_ok=True)
        default_name = "output/" + title.replace(' ', '_').replace('\n', '_')[:50] + '.png'
        plt.savefig(default_name, dpi=150)
        print(f"No display available. Saved: {default_name}")
        plt.close(fig)


def _has_display():
    """Check if a display is available for interactive plotting."""
    import os
    return os.environ.get('DISPLAY') is not None or os.environ.get('WAYLAND_DISPLAY') is not None


class TestWienerDeconvolve:
    """Tests for the wiener_deconvolve function."""
    
    def test_impulse_source_recovers_ir(self):
        """When source is an impulse, deconvolution should recover the IR directly."""
        n = 64
        
        # True impulse response: exponential decay
        h_true = np.zeros(n)
        h_true[0] = 1.0
        h_true[1] = 0.5
        h_true[2] = 0.25
        h_true[3] = 0.125
        
        # Source is a unit impulse at t=0
        s = np.zeros(n)
        s[0] = 1.0
        
        # Recorded signal is convolution of source and IR
        g = convolve_same_length(s, h_true)
        
        # Recover IR
        h_recovered = wiener_deconvolve(g, s, regularization=1e-10)
        
        # Should match closely
        np.testing.assert_allclose(h_recovered, h_true, atol=1e-10)
    
    def test_gaussian_source_recovers_ir(self):
        """Deconvolution should work with band-limited sources like Gaussians."""
        n = 256
        t = np.arange(n)
        
        # Gaussian pulse source starting near t=0 (causal)
        sigma = 3.0
        t0 = 4 * sigma  # Start at ~4 sigma for smooth onset
        s = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        
        # True IR: single delayed impulse
        h_true = np.zeros(n)
        delay = 20
        h_true[delay] = 1.0
        
        # Recorded = convolution
        g = convolve_same_length(s, h_true)
        
        # Recover IR - need low regularization for good accuracy
        h_recovered = wiener_deconvolve(g, s, regularization=1e-12)
        
        # Check the main peak is in the right place
        peak_true = np.argmax(np.abs(h_true))
        peak_recovered = np.argmax(np.abs(h_recovered))
        assert peak_recovered == peak_true, f"Peak at {peak_recovered}, expected {peak_true}"
        
        # Normalize and compare - the shape should match even if amplitude differs
        h_recovered_norm = h_recovered / np.max(np.abs(h_recovered))
        
        # Peak should be close to 1 after normalization
        np.testing.assert_allclose(h_recovered_norm[delay], 1.0, atol=0.05)
    
    def test_regularization_suppresses_noise(self):
        """Higher regularization should produce smoother results with noisy input."""
        n = 128
        
        # Simple IR
        h_true = np.zeros(n)
        h_true[0] = 1.0
        
        # Impulse source
        s = np.zeros(n)
        s[0] = 1.0
        
        # Clean recording
        g_clean = convolve_same_length(s, h_true)
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, n)
        g_noisy = g_clean + noise
        
        # Low regularization (more sensitive to noise)
        h_low_reg = wiener_deconvolve(g_noisy, s, regularization=1e-10)
        
        # High regularization (smoother)
        h_high_reg = wiener_deconvolve(g_noisy, s, regularization=1e-2)
        
        # High reg should have less high-frequency content (smoother)
        fft_low = np.abs(np.fft.rfft(h_low_reg))
        fft_high = np.abs(np.fft.rfft(h_high_reg))
        
        # High frequencies should be suppressed with high regularization
        high_freq_power_low = np.sum(fft_low[n//4:])
        high_freq_power_high = np.sum(fft_high[n//4:])
        
        assert high_freq_power_high < high_freq_power_low, \
            "Higher regularization should suppress high frequencies"
    
    def test_length_mismatch_raises_error(self):
        """Should raise ValueError if signal lengths don't match."""
        g = np.zeros(100)
        s = np.zeros(50)
        
        with pytest.raises(ValueError, match="lengths must match"):
            wiener_deconvolve(g, s)
    
    def test_identity_when_source_equals_recorded(self):
        """When g = s, the IR should be approximately a unit impulse."""
        n = 64
        
        # Source signal (any shape)
        t = np.arange(n)
        s = np.sin(2 * np.pi * 3 * t / n) * np.exp(-t / 20)
        
        # If recorded equals source, IR is identity (delta at t=0)
        g = s.copy()
        
        h = wiener_deconvolve(g, s, regularization=1e-10)
        
        # Should be ~1 at t=0, ~0 elsewhere
        assert np.abs(h[0] - 1.0) < 0.01, f"Expected h[0] ≈ 1, got {h[0]}"
        assert np.max(np.abs(h[1:])) < 0.01, f"Expected h[1:] ≈ 0, got max {np.max(np.abs(h[1:]))}"
    
    def test_delayed_impulse_response(self):
        """Should correctly identify a delayed impulse response."""
        n = 128
        delay = 15
        
        # Impulse source
        s = np.zeros(n)
        s[0] = 1.0
        
        # True IR: single impulse at delay
        h_true = np.zeros(n)
        h_true[delay] = 1.0
        
        # Recorded signal
        g = convolve_same_length(s, h_true)
        
        # Recover
        h_recovered = wiener_deconvolve(g, s, regularization=1e-10)
        
        # Peak should be at the correct delay
        peak_idx = np.argmax(np.abs(h_recovered))
        assert peak_idx == delay, f"Peak at {peak_idx}, expected {delay}"
        
        # Value at peak should be ~1
        np.testing.assert_allclose(h_recovered[delay], 1.0, atol=1e-10)
    
    def test_gaussian_source_gaussian_ir(self):
        """Gaussian source convolved with Gaussian IR (realistic case)."""
        n = 512
        t = np.arange(n)
        
        # Gaussian source pulse
        sigma_s = 5.0
        t0_s = 20.0
        s = np.exp(-0.5 * ((t - t0_s) / sigma_s) ** 2)
        
        # Gaussian impulse response (like a smoothed/dispersed propagation)
        sigma_h = 8.0
        t0_h = 30.0  # Delay representing propagation time
        h_true = np.exp(-0.5 * ((t - t0_h) / sigma_h) ** 2)
        # Normalize so peak = 1
        h_true = h_true / np.max(h_true)
        
        # Recorded = source * IR
        g = convolve_same_length(s, h_true)
        
        # Recover IR
        h_recovered = wiener_deconvolve(g, s, regularization=1e-10)
        
        # Normalize both for comparison
        h_true_norm = h_true / np.max(h_true)
        h_recovered_norm = h_recovered / np.max(h_recovered)
        
        # Peak should be at the right location
        peak_true = np.argmax(h_true)
        peak_recovered = np.argmax(h_recovered)
        assert abs(peak_recovered - peak_true) <= 1, \
            f"Peak at {peak_recovered}, expected {peak_true}"
        
        # Shape should match well in the main lobe region
        # Check a window around the peak
        window = slice(peak_true - 15, peak_true + 15)
        np.testing.assert_allclose(
            h_recovered_norm[window], 
            h_true_norm[window], 
            atol=0.05
        )
    
    def test_decaying_exponential_ir(self):
        """Gaussian source with smooth exponentially decaying IR (room-like response)."""
        n = 512
        t = np.arange(n)
        
        # Gaussian source pulse
        sigma_s = 4.0
        t0_s = 15.0
        s = np.exp(-0.5 * ((t - t0_s) / sigma_s) ** 2)
        
        # Smooth exponentially decaying IR: Gaussian onset with exponential decay
        # This is more realistic than a sharp step and better conditioned for deconvolution
        delay = 30
        onset_sigma = 3.0
        decay_rate = 0.015
        
        # Smooth onset (half Gaussian) transitioning to exponential decay
        h_true = np.zeros(n)
        for i in range(delay, n):
            t_rel = i - delay
            # Gaussian-windowed exponential decay
            onset = 1 - np.exp(-0.5 * (t_rel / onset_sigma) ** 2)
            decay = np.exp(-decay_rate * t_rel)
            h_true[i] = onset * decay
        
        # Normalize
        h_true = h_true / np.max(h_true)
        
        # Recorded = source * IR
        g = convolve_same_length(s, h_true)
        
        # Recover IR
        h_recovered = wiener_deconvolve(g, s, regularization=1e-8)
        
        # Normalize both
        h_recovered_norm = h_recovered / np.max(np.abs(h_recovered))
        
        # Peak should be at approximately the right location
        peak_true = np.argmax(h_true)
        peak_recovered = np.argmax(h_recovered)
        assert abs(peak_recovered - peak_true) <= 3, \
            f"Peak at {peak_recovered}, expected {peak_true}"
        
        # Check decay shape matches after the peak (where both are smooth)
        check_start = peak_true + 5
        check_range = slice(check_start, check_start + 40)
        np.testing.assert_allclose(
            h_recovered_norm[check_range],
            h_true[check_range],
            atol=0.1
        )


class TestWienerDeconvolveEdgeCases:
    """Edge case tests."""
    
    def test_zero_source_with_regularization(self):
        """Zero source should not cause division by zero due to regularization."""
        n = 32
        g = np.random.randn(n)
        s = np.zeros(n)  # Zero source
        
        # Should not raise, regularization prevents division by zero
        h = wiener_deconvolve(g, s, regularization=1e-3)
        
        # Result should be finite
        assert np.all(np.isfinite(h)), "Result should be finite"
    
    def test_single_sample(self):
        """Should work with single-sample signals."""
        g = np.array([2.0])
        s = np.array([1.0])
        
        h = wiener_deconvolve(g, s, regularization=1e-10)
        
        np.testing.assert_allclose(h, [2.0], atol=1e-10)
    
    def test_output_is_real(self):
        """Output should be real-valued (not complex)."""
        g = np.array([1.0, 2.0, 3.0, 4.0])
        s = np.array([1.0, 0.0, 0.0, 0.0])
        
        h = wiener_deconvolve(g, s)
        
        # Should be real (float type)
        assert np.issubdtype(h.dtype, np.floating), f"Expected float dtype, got {h.dtype}"


class TestWienerDeconvolveVisual:
    """Visual tests for manual inspection. Run with: pytest -k visual"""
    
    @pytest.mark.visual
    def test_visual_gaussian_gaussian(self):
        """Visual: Gaussian source × Gaussian IR."""
        n = 512
        t = np.arange(n)
        
        # Gaussian source
        sigma_s = 5.0
        t0_s = 20.0
        s = np.exp(-0.5 * ((t - t0_s) / sigma_s) ** 2)
        
        # Gaussian IR
        sigma_h = 8.0
        t0_h = 30.0
        h_true = np.exp(-0.5 * ((t - t0_h) / sigma_h) ** 2)
        h_true = h_true / np.max(h_true)
        
        g = convolve_same_length(s, h_true)
        h_recovered = wiener_deconvolve(g, s, regularization=1e-10)
        
        plot_deconvolution_result(t, s, h_true, g, h_recovered,
                                   "Gaussian Source × Gaussian IR",
                                   filename="output/test_deconv_gaussian_gaussian.png")
    
    @pytest.mark.visual
    def test_visual_decaying_exponential(self):
        """Visual: Gaussian source × decaying exponential IR.
        
        Note: Ringing before the IR onset is expected due to limited source bandwidth.
        The Gaussian's weak high-frequency content causes noise amplification.
        """
        n = 512
        t = np.arange(n)
        
        # Gaussian source
        sigma_s = 4.0
        t0_s = 15.0
        s = np.exp(-0.5 * ((t - t0_s) / sigma_s) ** 2)
        
        # Smooth exponentially decaying IR
        delay = 30
        onset_sigma = 3.0
        decay_rate = 0.015
        
        h_true = np.zeros(n)
        for i in range(delay, n):
            t_rel = i - delay
            onset = 1 - np.exp(-0.5 * (t_rel / onset_sigma) ** 2)
            decay = np.exp(-decay_rate * t_rel)
            h_true[i] = onset * decay
        h_true = h_true / np.max(h_true)
        
        g = convolve_same_length(s, h_true)
        h_recovered = wiener_deconvolve(g, s, regularization=1e-8)
        
        plot_deconvolution_result(t, s, h_true, g, h_recovered,
                                   "Gaussian Source × Decaying Exponential IR\n"
                                   "(Ringing before onset is expected - limited source bandwidth)",
                                   filename="output/test_deconv_exponential.png")
    
    @pytest.mark.visual
    def test_visual_regularization_comparison(self):
        """Visual: Effect of regularization on noisy signal."""
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Use non-interactive backend if no display
        if not _has_display():
            matplotlib.use('Agg')
        
        n = 256
        t = np.arange(n)
        
        # Gaussian source
        sigma_s = 5.0
        t0_s = 20.0
        s = np.exp(-0.5 * ((t - t0_s) / sigma_s) ** 2)
        
        # Simple delayed impulse IR
        h_true = np.zeros(n)
        delay = 25
        h_true[delay] = 1.0
        h_true[delay + 1] = 0.5
        h_true[delay + 2] = 0.25
        
        g_clean = convolve_same_length(s, h_true)
        
        # Add noise
        np.random.seed(42)
        noise_level = 0.02
        g_noisy = g_clean + np.random.normal(0, noise_level, n)
        
        # Different regularization values
        regs = [1e-12, 1e-6, 1e-3, 1e-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f'Effect of Regularization (noise level = {noise_level})', fontsize=14)
        
        axes[0, 0].plot(t, s, 'b-', linewidth=1.5)
        axes[0, 0].set_title('Source Signal')
        axes[0, 0].set_xlim(0, 80)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, g_clean, 'g-', linewidth=1.5, alpha=0.5, label='Clean')
        axes[0, 1].plot(t, g_noisy, 'r-', linewidth=1, label='Noisy')
        axes[0, 1].set_title('Recorded Signal')
        axes[0, 1].set_xlim(0, 120)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].stem(t[:50], h_true[:50], linefmt='g-', markerfmt='go', basefmt='k-')
        axes[0, 2].set_title('True IR')
        axes[0, 2].set_xlim(0, 50)
        axes[0, 2].grid(True, alpha=0.3)
        
        for idx, reg in enumerate(regs[:3]):
            ax = axes[1, idx]
            h_recovered = wiener_deconvolve(g_noisy, s, regularization=reg)
            h_rec_norm = h_recovered / np.max(np.abs(h_recovered)) if np.max(np.abs(h_recovered)) > 0 else h_recovered
            
            ax.plot(t, h_true / np.max(h_true), 'g-', linewidth=2, alpha=0.5, label='True')
            ax.plot(t, h_rec_norm, 'b-', linewidth=1, label='Recovered')
            ax.set_title(f'λ = {reg:.0e}')
            ax.set_xlim(0, 80)
            ax.set_ylim(-0.5, 1.2)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = "output/test_deconv_regularization.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--visual" in sys.argv:
        # Run only visual tests
        pytest.main([__file__, "-v", "-k", "visual"])
    else:
        # Run all non-visual tests by default
        pytest.main([__file__, "-v", "-k", "not visual"])
