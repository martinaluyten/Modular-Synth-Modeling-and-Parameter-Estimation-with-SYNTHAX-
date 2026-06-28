import os
import jax
import jax.numpy as jnp
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import gc
import concurrent.futures
from PIL import Image
from synthax.config import SynthConfig
from synthax.modules.oscillators import SineVCO, SquareSawVCO, Noise
from synthax.modules.envelopes import ADSR
from synthax.modules.amplifiers import VCA
from synthax.modules.mixers import AudioMixer
from synthax.modules.control import ControlRateUpsample
from synthax.modules.keyboard import MonophonicKeyboard


try:
    from synthax.modules.filters import LPF
    HAS_FILTER = True
except ImportError:
    HAS_FILTER = False
    print("Notice: LowPassFilter not found in your SynthAX version. Filter stage will be bypassed.")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class PRNGKey():
    def __init__(self, seed=13):
        self.PRNG_key = jax.random.PRNGKey(seed)
    def split(self):
        self.PRNG_key, subkey = jax.random.split(self.PRNG_key)
        return subkey


def generate_batch(
    batch_offset: int,
    base_seed: int,
    batch_size: int,
    output_folder: str = "SynthDataset",
    append_metadata: bool = False
):
    """
    Generate a single batch of synthetic audio files.

    Args:
        batch_offset: Starting file index (e.g., 0, 1000, 2000 for sequential naming)
        base_seed: Seed for JAX random number generator (ensures unique streams per batch)
        batch_size: Number of files to generate in this batch
        output_folder: Output directory for audio, spectrograms, and metadata
        append_metadata: If True, append to existing metadata.csv; if False, create new
    """
    # Initialize PRNG with unique seed for this batch
    PRNG_key = PRNGKey(seed=base_seed)

    # =====================================================================
    # --- MASTER CONFIGURATION BLOCK ---
    # =====================================================================
    config = SynthConfig(
        batch_size=batch_size,
        sample_rate=44100,
        buffer_size_seconds=6.0
    )

    note_on_duration = jnp.full((batch_size,), 4.0)

    # 1. Note
    midi_min = 21
    midi_max = 108
    midi_f0 = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=midi_min, maxval=midi_max)

    # 2. ADSR Volume Envelope
    A_max = 2
    A_min = 0
    attack = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=A_min, maxval=A_max)
    D_max = 1
    D_min = 0
    decay = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=D_min, maxval=D_max)
    sustain = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=0.0, maxval=1.0)
    R_max = 2
    R_min = 0
    release = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=R_min, maxval=R_max)
    alpha_min = 0
    alpha_max = 6
    alpha = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=alpha_min, maxval=alpha_max)

    # --- 4. Filter (Log-Uniform with Dynamic f0 Floor) ---
    min_cutoff = 100.0
    max_cutoff = 20000.0

    f0_hz = 440.0 * (2.0 ** ((midi_f0 - 69.0) / 12.0))
    dynamic_min = jnp.maximum(f0_hz, min_cutoff)
    log_cutoff_min = jnp.log(dynamic_min)
    log_cutoff_max = jnp.log(max_cutoff)
    log_cutoff = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=log_cutoff_min, maxval=log_cutoff_max)
    filter_cutoff = jnp.exp(log_cutoff)

    # 3. Oscillators & Mixer (Continuous Volumes)
    noise_min = 0
    noise_max = 0.5
    noise_level = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=noise_min, maxval=noise_max)
    remaining_energy = 1.0 - noise_level
    sine_split = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=0.0, maxval=1.0)
    sine_level = remaining_energy * sine_split
    sqr_level = remaining_energy * (1.0 - sine_split)
    mixer_levels = jnp.column_stack([sine_level, sqr_level, noise_level])
    sqr_saw_shape = jax.random.uniform(PRNG_key.split(), shape=(batch_size,), minval=0.0, maxval=1.0)

    # =====================================================================
    # --- PARAMETER SANITY CHECKS ---
    # =====================================================================
    if jnp.any((A_max + D_max) >= note_on_duration):
        raise ValueError("ADSR Error...")
    if jnp.any((note_on_duration + R_max) > config.buffer_size_seconds):
        raise ValueError("Buffer Error...")

    # =====================================================================
    # --- Parameter Normalization BLOCK ---
    # =====================================================================
    global_log_min = jnp.log(min_cutoff)
    global_log_max = jnp.log(max_cutoff)
    norm_cutoff = (jnp.log(filter_cutoff) - global_log_min) / (global_log_max - global_log_min)
    norm_f0 = (midi_f0 - midi_min) / (midi_max - midi_min)
    norm_attack = (attack - A_min) / (A_max - A_min)
    norm_decay = (decay - D_min) / (D_max - D_min)
    norm_release = (release - R_min) / (R_max - R_min)
    norm_sustain = sustain
    norm_alpha = (alpha - alpha_min) / (alpha_max - alpha_min)
    norm_noise_level = (noise_level - noise_min) / (noise_max - noise_min)
    norm_sine_level = sine_level
    norm_sqr_level = sqr_level
    norm_sqr_saw_shape = sqr_saw_shape

    # --- PIPELINE INITIALIZATION ---
    keyboard = MonophonicKeyboard(config=config, midi_f0=midi_f0, duration=note_on_duration)
    adsr = ADSR(config=config, attack=attack, decay=decay, sustain=sustain, release=release, alpha=alpha)
    upsample = ControlRateUpsample(config=config)

    pitch_mod_env = jnp.zeros((batch_size, config.buffer_size))
    sine_vco = SineVCO(config=config, tuning=jnp.array([0.0]*batch_size), mod_depth=jnp.array([0.0]*batch_size), initial_phase=jnp.array([0.0]*batch_size))
    square_saw = SquareSawVCO(config=config, tuning=jnp.array([0.0]*batch_size), mod_depth=jnp.array([0.0]*batch_size), initial_phase=jnp.array([0.0]*batch_size), shape=sqr_saw_shape)
    noise = Noise(config=config)

    mixer = AudioMixer(config=config, n_input=3, level=mixer_levels)

    if HAS_FILTER:
        synth_filter = LPF(config=config)

    vca = VCA(config=config)

    # --- EXECUTE SYNTHESIS GRAPH ---
    params_kb = keyboard.init(PRNG_key.split())
    out_midi, out_duration = jax.jit(keyboard.apply)(params_kb)

    params_adsr = adsr.init(PRNG_key.split(), out_duration)
    envelope = jax.jit(adsr.apply)(params_adsr, out_duration)

    params_up = upsample.init(PRNG_key.split(), envelope)
    envelope_audio = jax.jit(upsample.apply)(params_up, envelope)

    params_sine = sine_vco.init(PRNG_key.split(), out_midi, pitch_mod_env)
    sine_out = jax.jit(sine_vco.apply)(params_sine, out_midi, pitch_mod_env)

    params_sqr = square_saw.init(PRNG_key.split(), out_midi, pitch_mod_env)
    sqr_out = jax.jit(square_saw.apply)(params_sqr, out_midi, pitch_mod_env)

    params_noise = noise.init(PRNG_key.split())
    noise_out = jax.jit(noise.apply)(params_noise)

    params_mixer = mixer.init(PRNG_key.split(), sine_out, sqr_out, noise_out)
    mixed_audio = jax.jit(mixer.apply)(params_mixer, sine_out, sqr_out, noise_out)

    if HAS_FILTER:
        cutoff_signal = jnp.ones_like(mixed_audio) * filter_cutoff[:, None]
        params_filt = synth_filter.init(PRNG_key.split(), mixed_audio, cutoff_signal)
        processed_audio = jax.jit(synth_filter.apply)(params_filt, mixed_audio, cutoff_signal)
    else:
        processed_audio = mixed_audio

    params_vca = vca.init(PRNG_key.split(), envelope_audio, processed_audio)
    final_audio = jax.jit(vca.apply)(params_vca, envelope_audio, processed_audio)

    final_audio = jnp.nan_to_num(final_audio, nan=0.0, posinf=0.0, neginf=0.0)
    final_audio = jnp.clip(final_audio, -1.0, 1.0)

    # --- SAVE TO DISK ---
    print(f"Generated audio shape: {final_audio.shape} -> (Batch Size, Samples)")

    audio_folder = os.path.join(output_folder, "audio")
    spec_folder = os.path.join(output_folder, "spectrograms")

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(spec_folder, exist_ok=True)

    print(f"Saving files to disk inside '{audio_folder}', '{spec_folder}'...")

    final_audio_np = np.asarray(final_audio)

    metadata_list = []
    
    # Process sequentially (faster than ThreadPool for disk I/O)
    print(f"Processing {batch_size} files sequentially...")

    for i in range(batch_size):
        file_index = batch_offset + i + 1
        base_name = f"synth_sound_{file_index:05d}"
        # Audio generation skipped - spectrograms only (saves 50% disk space, 2x faster)
        # wav_path = os.path.join(audio_folder, f"{base_name}.wav")
        spec_path = os.path.join(spec_folder, f"{base_name}.jpg")  # JPEG for 5-10x faster save

        audio_data = final_audio_np[i]

        if not np.isfinite(audio_data).all():
            print(f"Skipping {base_name}: Non-finite values detected.")
            continue

        try:
            # Compute & Save Spectrogram (skip audio - not needed for training)
            S = librosa.feature.melspectrogram(y=audio_data, sr=config.sample_rate, n_mels=128, fmax=20000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_norm = ((S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8) * 255).astype(np.uint8)
            S_norm = np.flipud(S_norm)

            img = Image.fromarray(S_norm, mode='L')
            img = img.resize((256, 128), Image.Resampling.BILINEAR)
            # JPEG quality=85: 5-10x faster than PNG, minimal quality loss
            img.save(spec_path, 'JPEG', quality=85)

            metadata_list.append({
                "filename": f"{base_name}.wav",  # Keep for compatibility
                "spec_path": f"spectrograms/{base_name}.jpg",  # JPEG format
                "raw_midi_f0": float(midi_f0[i]),
                "raw_cutoff": float(filter_cutoff[i]),
                "raw_attack": float(attack[i]),
                "raw_decay": float(decay[i]),
                "raw_sustain": float(sustain[i]),
                "raw_release": float(release[i]),
                "raw_alpha": float(alpha[i]),
                "raw_noise": float(noise_level[i]),
                "raw_sine": float(sine_level[i]),
                "raw_sqr": float(sqr_level[i]),
                "raw_shape": float(sqr_saw_shape[i]),
                "target_f0": float(norm_f0[i]),
                "target_cutoff": float(norm_cutoff[i]),
                "target_attack": float(norm_attack[i]),
                "target_decay": float(norm_decay[i]),
                "target_sustain": float(norm_sustain[i]),
                "target_release": float(norm_release[i]),
                "target_alpha": float(norm_alpha[i]),
                "target_noise": float(norm_noise_level[i]),
                "target_sine": float(norm_sine_level[i]),
                "target_sqr": float(norm_sqr_level[i]),
                "target_shape": float(norm_sqr_saw_shape[i])
            })
            
            # Progress indicator every 100 files
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{batch_size} files...")
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue

    print(f"Successfully processed {len(metadata_list)} / {batch_size} files!")

    # Save metadata (append or create)
    df = pd.DataFrame(metadata_list)
    metadata_path = os.path.join(output_folder, "metadata.csv")

    if append_metadata and os.path.exists(metadata_path):
        df.to_csv(metadata_path, mode='a', header=False, index=False)
        print(f"Appended {len(metadata_list)} rows to {metadata_path}")
    else:
        df.to_csv(metadata_path, index=False)
        print(f"Created {metadata_path} with {len(metadata_list)} rows")

    # Clean up memory
    del final_audio, final_audio_np, mixed_audio, sine_out, sqr_out, noise_out
    del envelope, envelope_audio, processed_audio
    gc.collect()

    print(f"Batch complete! Files {batch_offset + 1} to {batch_offset + batch_size}")


# Keep original standalone functionality
if __name__ == "__main__":
    generate_batch(batch_offset=0, base_seed=13, batch_size=100, append_metadata=False)
