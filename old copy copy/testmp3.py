import numpy as np
import os
from pydub import AudioSegment

# Settings
SAMPLE_RATE = 44100  # CD Quality
DURATION_SEC = 15
OUTPUT_DIR = "rhythmic_math_tests"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def numpy_to_audiosegment(samples):
    """
    Converts a numpy array (float between -1 and 1) 
    into a Pydub AudioSegment (16-bit PCM).
    """
    # Normalize to 16-bit integer range (-32768 to 32767)
    # This prevents clipping distortion
    samples = np.clip(samples, -1, 1)
    samples_int16 = (samples * 32767).astype(np.int16)
    
    # Convert to bytes
    byte_data = samples_int16.tobytes()
    
    # Create AudioSegment
    return AudioSegment(
        data=byte_data, 
        sample_width=2,  # 16-bit = 2 bytes
        frame_rate=SAMPLE_RATE, 
        channels=1 # Mono
    )

# Create a time array for math calculations
# This represents time in seconds for every sample in the 15s file
t = np.linspace(0, DURATION_SEC, int(SAMPLE_RATE * DURATION_SEC), endpoint=False)

print("Generating rhythmic math-based MP3 files...")

# --- Test 1: The Techno Pulse (Gated Sine) ---
# Math: A 220Hz sine wave is multiplied by a square wave created with modulo.
# Formula: sin(f * t) * (1 if mod(t, 0.25) > 0.125 else 0)
print("1. Generating Techno Pulse...")
frequency = 220.0
beat_freq = 4.0 # 4 beats per second (240 BPM)

# Generate the base tone
tone = np.sin(2 * np.pi * frequency * t)
# Generate a rhythm gate (switches between 0 and 1)
gate = (np.mod(t * beat_freq, 1) > 0.5).astype(float)
# Apply gate
signal_1 = tone * gate * 0.8

seg1 = numpy_to_audiosegment(signal_1)
seg1.export(os.path.join(OUTPUT_DIR, "test_1_techno_pulse.mp3"), format="mp3")


# --- Test 2: The Frequency Sweep (Chirp) ---
# Math: The frequency increases exponentially over time using f(t) = f0 * 2^(t).
# This creates a "zoom" sound.
print("2. Generating Frequency Sweep (Chirp)...")
f0 = 100.0 # Start frequency
f1 = 2000.0 # End frequency

# Exponential frequency growth
instantaneous_phase = 2 * np.pi * f0 * (np.power(f1/f0, t/DURATION_SEC) - 1) / (np.log(f1/f0) / DURATION_SEC)
signal_2 = np.sin(instantaneous_phase) * 0.7

seg2 = numpy_to_audiosegment(signal_2)
seg2.export(os.path.join(OUTPUT_DIR, "test_2_sine_chirp.mp3"), format="mp3")


# --- Test 3: Amplitude Tremolo (Vibrating Volume) ---
# Math: A steady tone whose volume is controlled by sin(t).
# Formula: sin(carrier) * (0.5 + 0.5 * sin(modulator))
print("3. Generating Tremolo Effect...")
carrier_freq = 440.0
modulator_freq = 6.0 # Volume oscillates 6 times per second

# Base tone
carrier = np.sin(2 * np.pi * carrier_freq * t)
# Volume LFO (Low Frequency Oscillator)
lfo = 0.5 + (0.5 * np.sin(2 * np.pi * modulator_freq * t))
signal_3 = carrier * lfo * 0.8

seg3 = numpy_to_audiosegment(signal_3)
seg3.export(os.path.join(OUTPUT_DIR, "test_3_tremolo.mp3"), format="mp3")


# --- Test 4: Additive Synthesis (Pulse Wave) ---
# Math: Adding multiple sine waves with diminishing amplitudes (1/n).
# This creates a rich, "buzzy" synthesizer sound.
print("4. Generating Additive Synthesis (Pulse Wave)...")
fundamental = 110.0
signal_4 = np.zeros_like(t)

# Add harmonics (1st, 3rd, 5th, etc. for a square-ish wave)
for n in range(1, 20, 2):
    signal_4 += (1.0/n) * np.sin(2 * np.pi * fundamental * n * t)

# Normalize to prevent clipping
signal_4 = signal_4 / np.max(np.abs(signal_4)) * 0.7

seg4 = numpy_to_audiosegment(signal_4)
seg4.export(os.path.join(OUTPUT_DIR, "test_4_additive_synth.mp3"), format="mp3")


# --- Test 5: Math Noise (Pseudo-Random) ---
# Math: We create noise using a polynomial function or random distribution.
# We then gate it rhythmically to make it a beat.
print("5. Generating Gated Math Noise...")
# Generate white noise using standard deviation math
noise = np.random.normal(0, 1, len(t))

# Create a rhythmic gate (1/16th notes)
# 8 beats per second
gate_speed = 8.0 
# Create a triangle wave for the gate shape
gate_shape = np.abs(2 * (np.mod(t * gate_speed, 1) - 0.5))

# Combine noise with gate
signal_5 = noise * gate_shape * 0.6

seg5 = numpy_to_audiosegment(signal_5)
seg5.export(os.path.join(OUTPUT_DIR, "test_5_gated_noise.mp3"), format="mp3")

print(f"\nSuccess! Files saved in folder: {OUTPUT_DIR}")