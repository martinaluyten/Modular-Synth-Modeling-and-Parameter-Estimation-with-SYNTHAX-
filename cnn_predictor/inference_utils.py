import io
import math
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
import jax
import jax.numpy as jnp
from synthax.config import SynthConfig
from synthax.modules.oscillators import SineVCO, SquareSawVCO, Noise
from synthax.modules.envelopes import ADSR
from synthax.modules.amplifiers import VCA
from synthax.modules.mixers import AudioMixer
from synthax.modules.control import ControlRateUpsample
from synthax.modules.keyboard import MonophonicKeyboard
import torch
from torchvision import transforms
import pandas as pd

try:
    from synthax.modules.filters import LPF
    HAS_FILTER = True
except ImportError:
    HAS_FILTER = False


def decode_normalized_params(prediction, selected_params):
    """Reverse the training normalization to raw SynthAX parameter values."""
    values = np.asarray(prediction).flatten()

    midi_min = 21.0
    midi_max = 108.0
    cutoff_min = 100.0
    cutoff_max = 20000.0
    alpha_min = 0.0
    alpha_max = 6.0
    noise_max = 0.5

    raw_params = {}
    for idx, param in enumerate(selected_params):
        norm_value = float(values[idx])
        norm_value = max(0.0, min(1.0, norm_value))

        if param == 'midi_f0':
            raw_params[param] = norm_value * (midi_max - midi_min) + midi_min
        elif param == 'cutoff':
            log_min = math.log(cutoff_min)
            log_max = math.log(cutoff_max)
            raw_params[param] = math.exp(norm_value * (log_max - log_min) + log_min)
        elif param == 'attack':
            raw_params[param] = norm_value * 2.0
        elif param == 'decay':
            raw_params[param] = norm_value * 1.0
        elif param == 'sustain':
            raw_params[param] = norm_value
        elif param == 'release':
            raw_params[param] = norm_value * 2.0
        elif param == 'alpha':
            raw_params[param] = norm_value * 6.0
        elif param == 'noise':
            raw_params[param] = norm_value * noise_max
        elif param in ('sine', 'sqr', 'shape'):
            raw_params[param] = norm_value
        else:
            raw_params[param] = norm_value

    return raw_params


def synthesize_waveform(raw_params, sample_rate=44100, duration=6.0):
    """Synthesize one waveform from raw SynthAX parameter values."""
    config = SynthConfig(batch_size=1, sample_rate=sample_rate, buffer_size_seconds=duration)

    midi_f0 = jnp.array([raw_params.get('midi_f0', 69.0)])
    note_on_duration = jnp.array([4.0])
    attack = jnp.array([raw_params.get('attack', 0.0)])
    decay = jnp.array([raw_params.get('decay', 0.0)])
    sustain = jnp.array([raw_params.get('sustain', 0.0)])
    release = jnp.array([raw_params.get('release', 0.0)])
    alpha = jnp.array([raw_params.get('alpha', 0.0)])
    noise_level = jnp.array([raw_params.get('noise', 0.0)])
    sine_level = jnp.array([raw_params.get('sine', 0.0)])
    sqr_level = jnp.array([raw_params.get('sqr', 0.0)])
    sqr_saw_shape = jnp.array([raw_params.get('shape', 0.0)])

    # Ensure oscillator levels are numerically stable
    noise_level = jnp.clip(noise_level, 0.0, 0.5)
    sine_level = jnp.clip(sine_level, 0.0, 1.0)
    sqr_level = jnp.clip(sqr_level, 0.0, 1.0)
    mixer_levels = jnp.column_stack([sine_level, sqr_level, noise_level])

    min_cutoff = 100.0
    max_cutoff = 20000.0
    f0_hz = 440.0 * (2.0 ** ((midi_f0 - 69.0) / 12.0))
    dynamic_min = jnp.maximum(f0_hz, min_cutoff)
    cutoff_hz = raw_params.get('cutoff', 1000.0)
    cutoff_hz = jnp.clip(cutoff_hz, dynamic_min, max_cutoff)
    filter_cutoff = cutoff_hz

    keyboard = MonophonicKeyboard(config=config, midi_f0=midi_f0, duration=note_on_duration)
    adsr = ADSR(config=config, attack=attack, decay=decay, sustain=sustain, release=release, alpha=alpha)
    upsample = ControlRateUpsample(config=config)

    pitch_mod_env = jnp.zeros((1, config.buffer_size))
    sine_vco = SineVCO(config=config, tuning=jnp.array([0.0]), mod_depth=jnp.array([0.0]), initial_phase=jnp.array([0.0]))
    square_saw = SquareSawVCO(config=config, tuning=jnp.array([0.0]), mod_depth=jnp.array([0.0]), initial_phase=jnp.array([0.0]), shape=sqr_saw_shape)
    noise = Noise(config=config)
    mixer = AudioMixer(config=config, n_input=3, level=mixer_levels)

    if HAS_FILTER:
        synth_filter = LPF(config=config)

    vca = VCA(config=config)

    params_kb = keyboard.init(jax.random.PRNGKey(0))
    out_midi, out_duration = keyboard.apply(params_kb)

    params_adsr = adsr.init(jax.random.PRNGKey(1), out_duration)
    envelope = adsr.apply(params_adsr, out_duration)

    params_up = upsample.init(jax.random.PRNGKey(2), envelope)
    envelope_audio = upsample.apply(params_up, envelope)

    params_sine = sine_vco.init(jax.random.PRNGKey(3), out_midi, pitch_mod_env)
    sine_out = sine_vco.apply(params_sine, out_midi, pitch_mod_env)

    params_sqr = square_saw.init(jax.random.PRNGKey(4), out_midi, pitch_mod_env)
    sqr_out = square_saw.apply(params_sqr, out_midi, pitch_mod_env)

    params_noise = noise.init(jax.random.PRNGKey(5))
    noise_out = noise.apply(params_noise)

    params_mixer = mixer.init(jax.random.PRNGKey(6), sine_out, sqr_out, noise_out)
    mixed_audio = mixer.apply(params_mixer, sine_out, sqr_out, noise_out)

    if HAS_FILTER:
        cutoff_signal = jnp.ones_like(mixed_audio) * filter_cutoff[:, None]
        params_filt = synth_filter.init(jax.random.PRNGKey(7), mixed_audio, cutoff_signal)
        processed_audio = synth_filter.apply(params_filt, mixed_audio, cutoff_signal)
    else:
        processed_audio = mixed_audio

    params_vca = vca.init(jax.random.PRNGKey(8), envelope_audio, processed_audio)
    final_audio = vca.apply(params_vca, envelope_audio, processed_audio)

    final_audio = jnp.nan_to_num(final_audio, nan=0.0, posinf=0.0, neginf=0.0)
    final_audio = jnp.clip(final_audio, -1.0, 1.0)

    return np.asarray(final_audio)[0].astype(np.float32)


def waveform_to_wav_bytes(waveform, sample_rate=44100):
    """Encode a waveform to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.getvalue()


def waveform_to_spectrogram_image(waveform, sample_rate=44100, n_mels=128, fmax=20000):
    """Render a waveform as a mel spectrogram image."""
    S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = ((S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8) * 255).astype(np.uint8)
    S_norm = np.flipud(S_norm)
    img = Image.fromarray(S_norm, mode='L')
    img = img.resize((256, 128), Image.Resampling.BILINEAR)
    return img


def _normalize_image_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 255).astype(np.uint8)


def spectrogram_diff_image(img_a, img_b):
    """Compute an absolute difference spectrogram image between two spectrograms."""
    arr_a = np.asarray(img_a.convert('L'), dtype=np.float32)
    arr_b = np.asarray(img_b.convert('L'), dtype=np.float32)

    if arr_a.shape != arr_b.shape:
        img_a = Image.fromarray(arr_a.astype(np.uint8)).resize(img_b.size, Image.Resampling.BILINEAR)
        arr_a = np.asarray(img_a, dtype=np.float32)

    diff = np.abs(arr_a - arr_b)
    diff_norm = _normalize_image_array(diff)
    return Image.fromarray(diff_norm, mode='L')


def waveform_diff_wav_bytes(orig_waveform, pred_waveform, sample_rate=44100):
    """Return WAV bytes for the difference audio between original and predicted waveforms."""
    min_len = min(len(orig_waveform), len(pred_waveform))
    diff_waveform = orig_waveform[:min_len] - pred_waveform[:min_len]
    diff_waveform = np.clip(diff_waveform, -1.0, 1.0)
    return waveform_to_wav_bytes(diff_waveform, sample_rate=sample_rate)


def prepare_image_tensor(image, resize=(128, 256), mean=0.5, std=0.5, device='cpu'):
    """Convert a PIL Image to a batched torch tensor suitable for the trainer.

    Returns a tensor of shape (1, 1, H, W) on the requested device.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    tensor = transform(image)  # (C, H, W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def extract_raw_params_from_row(row):
    """Extract raw_... columns from a pandas Series (metadata row) into a param dict."""
    raw_cols = [c for c in row.index if str(c).startswith('raw_')]
    params = {}
    for c in raw_cols:
        pname = c.replace('raw_', '')
        try:
            params[pname] = float(row[c])
        except Exception:
            params[pname] = row[c]
    return params


def params_dict_to_dataframe(params_dict):
    """Convert a params dict to a pandas DataFrame for display."""
    return pd.DataFrame([{"Parameter": k, "Value": v} for k, v in params_dict.items()])


def predict_and_synthesize(trainer, image_tensor):
    """Run model prediction on a preprocessed image tensor and synthesize audio+spectrogram.

    Returns a dict with keys: 'pred_norm' (1D numpy), 'pred_raw' (dict), 'waveform', 'wav_bytes', 'spec_img'
    """
    # Ensure batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Move tensor to CPU first; trainer.predict handles device transfer
    image_tensor = image_tensor.cpu()

    pred = trainer.predict(image_tensor)
    # pred is numpy array (B, P)
    pred_np = np.asarray(pred)
    if pred_np.ndim == 2 and pred_np.shape[0] >= 1:
        pred_vec = pred_np[0]
    else:
        pred_vec = pred_np.flatten()

    pred_raw = decode_normalized_params(pred_vec, trainer.selected_params)

    # Synthesize predicted waveform and assets
    waveform = synthesize_waveform(pred_raw)
    wav_bytes = waveform_to_wav_bytes(waveform)
    spec_img = waveform_to_spectrogram_image(waveform)

    return {
        'pred_norm': pred_vec,
        'pred_raw': pred_raw,
        'waveform': waveform,
        'wav_bytes': wav_bytes,
        'spec_img': spec_img
    }


def synthesize_from_raw_params(raw_params):
    """Synthesize waveform, WAV bytes and spectrogram image from raw params."""
    waveform = synthesize_waveform(raw_params)
    wav_bytes = waveform_to_wav_bytes(waveform)
    spec_img = waveform_to_spectrogram_image(waveform)
    return {
        'waveform': waveform,
        'wav_bytes': wav_bytes,
        'spec_img': spec_img
    }


def predict_from_image(image, trainer, device='cpu'):
    """Prepare an image, run the trainer prediction and synthesize predicted assets.

    Args:
        image: PIL.Image or array-like spectrogram image (grayscale accepted).
        trainer: Trainer object with `predict()` and `selected_params`.
        device: device for tensor preparation (default 'cpu').

    Returns:
        dict: same structure as `predict_and_synthesize` (pred_norm, pred_raw, waveform, wav_bytes, spec_img)
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    img_tensor = prepare_image_tensor(image, device=device)
    return predict_and_synthesize(trainer, img_tensor)


def generate_and_predict(raw_params, trainer, device='cpu'):
    """Synthesize an original waveform from raw_params, create its spectrogram, then run prediction on that spectrogram.

    Returns a tuple: (orig_assets, pred_result)
    - orig_assets: dict with keys 'waveform','wav_bytes','spec_img'
    - pred_result: dict from `predict_and_synthesize` for the generated spectrogram
    """
    orig_assets = synthesize_from_raw_params(raw_params)
    spec_img = orig_assets['spec_img']
    pred_result = predict_from_image(spec_img, trainer, device=device)
    return orig_assets, pred_result
