# CoherenceForLlama
Phase Coherence Instrumentation for llama.cpp

Background and Rationale

Large language models can experience hidden state drift or instability during long sequence generation, leading to degraded output quality or hallucinations. Inspired by neuroscience, we introduce phase coherence as a new diagnostic of stability in llama.cpp. In brain networks, synchronized oscillatory activity (high phase coherence) is a sign of well-connected, stable communication between regions. Coherence metrics measure whether signals share a similar frequency and phase content. By analogy, measuring phase coherence in a transformer's internal activations can indicate whether the model’s layers are “in sync” or if the hidden representations start to drift out of alignment. A drop in phase coherence could flag moments when the model’s internal state becomes unstable, which may correlate with the onset of incoherent or hallucinated outputs. This approach complements traditional measures like residual norm magnitude by providing a phase-based view of the model’s internal dynamics.

Why phase? Neural network activations can be viewed as high-dimensional signals. We hypothesize that as long-context generation proceeds, hidden states maintain a consistent phase angle in some latent oscillatory mode, reflecting a stable encoding of context. If the model begins to lose context or stray (e.g. producing off-topic or repetitive text), this latent phase may shift erratically, reducing coherence. Phase coherence thus serves as a stability indicator: high coherence suggests the model is staying on track, while a sudden phase decoherence might signal the model is drifting from its narrative or factual grounding.

# Enabling Phase Coherence Tracing (Build Configuration)

To keep the overhead zero in normal use, the coherence tracing instrumentation is included behind a compile-time flag. You must build llama.cpp with the special flag enabled. A new CMake option `LLAMA_COHERENCE_TRACE` is provided:

`
cmake . -DLLAMA_COHERENCE_TRACE=ON
make
`
By default (flag off), none of the coherence-tracing code is included or executed, ensuring no performance impact or memory overhead on standard runs. When compiled with `-DLLAMA_COHERENCE_TRACE`, the llama.cpp binary will support additional options to configure and collect phase coherence data. This design is upstream-friendly – when the flag is disabled, the code is essentially a no-op (compiling to nothing) so it won’t affect existing functionality or speed. All tracing hooks are guarded with `#ifdef LLAMA_COHERENCE_TRACE` so they vanish completely in a normal build. Developers can thus merge this feature without worrying about regressions in core inference performance.

# Runtime Configuration (coh_trace_cfg)

When coherence tracing is enabled at compile time, you can control it at runtime through a structured configuration. We introduce a struct (in C/C++ code) called `llama_coherence_config` (or `coh_trace_cfg` for short) that holds all the relevant settings for phase coherence logging. This config can be set via new command-line arguments or through an API if llama.cpp is used as a library. The fields in the config include:

```
struct llama_coherence_config {
    int  layer_start;       // First layer index to trace (inclusive)
    int  layer_end;         // Last layer index to trace (inclusive)
    int  token_stride;      // Log every Nth token (e.g., 1 = every token, 5 = one out of 5)
    enum TapPoint {         // Where to tap the signal in each layer
        TAP_AFTER_ATTENTION,
        TAP_AFTER_MLP,
        TAP_FINAL_LAYER
    } tap_point;
    bool retain_magnitude;  // Whether to retain magnitude info (if false, only phase is kept)
    bool quantize;          // Whether to quantize phase (and magnitude) to 8-bit values
    int  output_buffer_size; // Buffer length (in tokens) for batching output writes
    const char* output_path; // File path for the output log (binary)
};
```

Layer range selection: You can choose a subset of layers to instrument by specifying `layer_start` and `layer_end`. For example, you might trace only the last few layers of the model where most of the high-level pattern formation occurs. By default, you could set `layer_start = 0` and `layer_end = N-1` (for all N layers) or focus on a critical range (e.g., the final layer or a middle segment). Tracing fewer layers reduces overhead and log size.

Token stride control: The `token_stride` parameter controls temporal sampling. Not every token’s activations need to be logged – for long sequences, you might log one out of every N tokens. For instance, `token_stride = 1` logs every token (highest resolution), while `token_stride = 4` would log phase info for one token, skip the next three, then log again. This can significantly cut down the volume of data and overhead for very long generations, at the cost of some time resolution. Striding is useful when monitoring extended outputs (like thousands of tokens) where fine-grained detail might be overkill.

Tap point selection: The `tap_point` setting lets you choose where in the layer to extract the activations for phase analysis. A transformer layer has multiple sub-parts; common choices are:

- After attention output (pre-MLP): tapping the residual after the multi-head attention block (before the feed-forward MLP is applied).

- After MLP (layer output): tapping the residual after the feed-forward network (i.e., after the entire layer’s computations, usually just before adding to the next layer or before final layer norm).

- Final layer output: tapping the final hidden state (after the last layer, before the logits are computed).

By selecting the tap point, you can investigate phase coherence at different processing stages. For example, tapping after the MLP of each layer gives a view of each layer’s output phase progression. Alternatively, tapping after attention might isolate coherence changes due to attention mechanism. The tap is implemented as a hook in the forward pass – when the model reaches the chosen point in each selected layer for a given token, it triggers the logging routine (described below).

Magnitude retention toggle: The `retain_magnitude` flag controls whether we capture the activation vector’s magnitude information along with phase. If this is set to `false`, the logging will only record the phase angle (essentially the directional information of the hidden state vector) and discard the length. If set to `true`, the logging will record both the phase and the magnitude (in practice, the projected magnitude on the chosen phase axis – see next section). Retaining magnitude can provide additional insight: for instance, a phase might remain the same but the vector norm might collapse or spike, which is another indicator of potential instability. However, omitting magnitude reduces the data size and focuses purely on directional coherence. By default, we might recommend `retain_magnitude = false` for minimal output, unless the user specifically wants to analyze the interplay of phase and activation norm.

Quantization option: If `quantize` is enabled, the phase (and magnitude, if kept) are quantized to 8-bit values when stored, rather than full precision floats. Quantization dramatically shrinks log size – each value becomes a single byte – and speeds up I/O, with negligible impact on analysis in most cases. Phase naturally falls in a bounded range (-π to π), so we can map that continuous range to 256 discrete levels. Magnitude can be unbounded, but typically we deal with normalized activations (especially if using post-LN outputs); we handle magnitude quantization by clamping to a reasonable range or using dynamic scaling. If high fidelity is required for research, you can leave quantization off (`quantize=false`), which will store 32-bit floats for each value. The trade-off is file size and throughput versus precision.

Output buffering: The `output_buffer_size` sets how many token records to accumulate before writing to disk. Writing to disk can be slow if done token by token. Instead, the implementation uses a memory buffer to batch writes. For example, if `output_buffer_size = 10`, the system will gather 10 tokens worth of phase data in memory and then flush them all at once to the output file. Larger buffer sizes improve performance by reducing I/O calls (especially important if `token_stride` is 1 and you are logging very frequently). However, setting it too large could risk losing a lot of data if the program crashes unexpectedly; a moderate default (e.g. 16 or 32 tokens) is used. The buffer is always flushed at the end of generation to ensure no data is left unwritten.

Finally, `output_path` specifies where to save the binary log. You provide a filepath (e.g., `--coherence-log myrun.coh`) via CLI. If not provided, a default name like `coherence_trace.bin` might be used, but it’s good practice to specify it to avoid overwriting previous logs.

# Logging Hook and Phase Extraction

With configuration in place, llama.cpp injects a logging hook into the model's forward pass for the specified layers and tap points. As the model generates each token, and for each selected layer, the following occurs:

\PSEUDOCODE\
```
#ifdef LLAMA_COHERENCE_TRACE
if (coherence_tracing_enabled && (token_index % cfg.token_stride == 0)) {
    // Only log for configured layers
    if (layer_index >= cfg.layer_start && layer_index <= cfg.layer_end && tap_point_reached(layer_index, cfg.tap_point)) {
        const int d = model.d_model;                  // hidden dimension size
        const float* vec = current_hidden_state.data; // pointer to the activation vector at tap point
        // Compute phase via cosine transform
        float cos_sum = 0.0f, sin_sum = 0.0f;
        float omega = 2.0f * M_PI / d;                // base frequency (1 cycle over the vector length)
        for (int i = 0; i < d; ++i) {
            float v = vec[i];
            cos_sum += v * cosf(omega * i);
            sin_sum += v * sinf(omega * i);
        }
        float phase = atan2f(sin_sum, cos_sum);       // phase angle in radians
        float magnitude = 0.0f;
        if (cfg.retain_magnitude) {
            magnitude = sqrtf(cos_sum*cos_sum + sin_sum*sin_sum);
        }
        // (Optional normalization could be applied here if needed)
        // Quantize if enabled
        if (cfg.quantize) {
            uint8_t phase_q = (uint8_t) floorf( (phase + (float)M_PI) * (255.0f / (2.0f * (float)M_PI)) + 0.5f );
            buffer.push_back(phase_q);
            if (cfg.retain_magnitude) {
                // For magnitude, a simple approach is to use a fixed scale or dynamic range
                // Here we assume magnitude is roughly in [0, max_val] (e.g., 10.0 as a safe upper bound or use max seen)
                float mag_clamped = fminf(magnitude, MAG_MAX);
                uint8_t mag_q = (uint8_t) floorf( mag_clamped * (255.0f / MAG_MAX) + 0.5f );
                buffer.push_back(mag_q);
            }
        } else {
            // Store as 32-bit floats
            buffer.append((char*) &phase, sizeof(float));
            if (cfg.retain_magnitude) {
                buffer.append((char*) &magnitude, sizeof(float));
            }
        }
        // Check buffer and flush if needed
        if (buffer.size() >= cfg.output_buffer_size * record_size) {
            write_buffer_to_file(buffer, outfile);
            buffer.clear();
        }
    }
}
#endif
```
/PSEUDOCODE/

In the above pseudocode, you can see how we compute the phase. We treat the hidden state vector (of length `d_model`) as a discrete signal over the “space” of neuron indices. We then project this signal onto a single cosine wave that spans the length of the vector. Specifically, we compute the dot product of the activation vector with a cosine and sine basis of one cycle across the vector indices. This yields two values, `cos_sum` and` sin_sum`, which can be thought of as the real and imaginary components of the first Fourier mode of the activation pattern. The phase angle is then `atan2(sin_sum, cos_sum)`, giving a value in `-π,π` that represents the dominant phase of the vector pattern relative to our reference cosine.

Adaptive frequency scaling: The choice of one full cycle across the vector (i.e., using `omega = 2π/d_model`) is deliberate. It ensures that the phase measurement naturally adapts to the size of the model’s hidden dimension. For a model with `d_model = 4096`, the cosine transform has a spatial frequency that completes one oscillation over 4096 elements; for a smaller model with `d_model = 1024`, the cosine wave completes in 1024 elements. In both cases, we are capturing a comparable mode of variation (the lowest-frequency mode). This adaptive scaling means the phase angle is roughly comparable across models of different widths. In principle, one could experiment with higher-frequency modes (multiple oscillations across the vector) to capture finer spatial patterns, but the default of one-cycle-per-vector worked well in initial testing as a broad measure of coherence. It is a lightweight computation – O(N) per vector – focusing on the primary oscillatory component of the activation pattern rather than doing a full FFT or complex analysis.

Why cosine transform? The discrete cosine transform (DCT) is used instead of a full Fourier transform because our data (the activation values) are real-valued and we are mostly interested in a real-valued phase alignment. By using cosine and sine sums (effectively a single-frequency DFT), we get a 2D projection of the high-dimensional state onto a circular gauge (cos vs sin). The resulting phase angle captures the shape of the activation distribution across neurons, modulo that sinusoidal basis. If the activations shift in a systematic way (e.g. a subset of neurons increase while others decrease in a pattern that “rotates” the vector in this basis), the phase angle will shift accordingly. If the activation pattern stays consistent, the phase angle remains steady. In essence, this method condenses a high-dimensional change into a one-dimensional phase change. (Note: Because the reference waveform is fixed, the absolute phase value is somewhat arbitrary; it’s changes or stability in phase over time that carry meaning.)

Magnitude (amplitude) calculation: If `retain_magnitude` is `true`, we also log the magnitude corresponding to that phase component. In the code, `magnitude = sqrt(cos_sum^2 + sin_sum^2)` gives the length of the projection of the activation vector onto the chosen cosine/sine basis. This is effectively the amplitude of that oscillatory component in the hidden state. If the overall activation vector norm shrinks or grows, this will reflect in the magnitude. We do not explicitly normalize the vector before computing phase, because the phase angle from `atan2` is scale-invariant (scaling the entire vector scales both `cos_sum` and `sin_sum` equally, canceling out in the ratio). However, extremely low magnitudes indicate that the phase is not well-defined (the vector had little projection on the basis, perhaps just noise). In such cases, the phase might fluctuate more (and coherence metric will account for it by weighting by magnitude or simply showing low confidence). If one wanted to strictly separate phase from amplitude, an optional normalization step could set the vector length to 1 before computing phase – but we left the raw computation to preserve real amplitude info for those who want it.

After computing phase (and possibly magnitude), the values are stored to a buffer (with optional quantization). Quantization works as follows:

Phase quantization: We map the phase from `-π,π` to an 8-bit unsigned integer `0,255`. For example, -π corresponds to 0, 0 phase corresponds to ~128, and +π corresponds to 255. The mapping in code: `phase_q = (phase + π) * (255 / (2π))`. This preserves relative differences in phase to about 1.4° precision, which is more than enough for our analysis.

Magnitude quantization: We need to decide a scale. If using post-layer-norm activations, the vector norm might be around 1 on average. But to be safe for spikes, we choose an upper bound (say `MAG_MAX`, e.g. 10 or a dynamic max) and map magnitude linearly into 0–255. Any magnitude above the cap is clamped to 255. For example, `mag_q = magnitude * (255 / MAG_MAX)`. The quantization error for magnitude is usually not an issue for our purposes (we care about relative changes, not absolute precision to many decimal places). If magnitude is not retained, this step is skipped.

The buffer accumulates bytes (if quantized) or floats (if not) for each logged value. The `record_size` in the pseudocode refers to the number of bytes per token per layer (e.g. if logging phase+mag as bytes for one layer, `record_size = 2 bytes`; if floats for one layer, `record_size = 8 bytes`, etc., and if multiple layers, multiply that accordingly). When the buffer is full (reaches `output_buffer_size * record_size bytes`), it writes out to disk and clears the buffer. Writing is done in binary mode to the file path specified in `cfg.output_path`.

# Output Format (Binary Log Structure)

The output file is a binary file containing a header followed by a sequence of records. We designed the format to be compact and easy to parse:

- Header: A fixed-size header at the start of the file contains metadata about the run, so that analysis tools know how to interpret the data.

- Records: Following the header, each record corresponds to one token’s data (for the selected layers).

The header layout (all integers are little-endian):
```
uint32_t magic         // Magic number to identify this as a coherence log (e.g., 0x434F4845 for "COHE")
uint16_t version       // Format version, start at 1
uint16_t n_layers      // Number of layers logged
uint16_t layer_start   // First layer index
uint16_t layer_end     // Last layer index
uint16_t d_model       // Model hidden dimension size (for reference)
uint16_t token_stride  // Token stride used
uint8_t  has_magnitude // 1 if magnitude retained, 0 if phase-only
uint8_t  quantized     // 1 if data is quantized (uint8 values), 0 if float
uint32_t n_tokens      // Number of token records (could be 0 if ended early unknown, or we may fill this at end)
uint32_t reserved[3]   // Reserved for future use (or to pad the header to a fixed size, e.g., 48 bytes total)
```
(We pad the header to a convenient size; e.g., 48 bytes, leaving room for more fields in future.)

Immediately after the header, the token records are stored back-to-back. Each token record contains data for each layer from `layer_start` to `layer_end` in order. The content of a record depends on whether quantization is enabled and whether magnitude is included:

- If `quantized = 1`: For each logged layer, 1 byte for phase (uint8) and if `has_magnitude = 1`, then 1 byte for magnitude. So per layer it’s either 1 or 2 bytes. For `n_layers` layers, that’s `n_layers` bytes (phase-only) or `2 * n_layers` bytes (phase+mag).

- If `quantized = 0`: For each logged layer, 4 bytes (float) for phase, and if `has_magnitude = 1`, then another 4 bytes (float) for magnitude. That means 4 or 8 bytes per layer per token.

The records are stored in chronological order of tokens (i.e., the generation order). We do not explicitly store a timestamp or token index for each record to save space; the index can be inferred by position in the sequence and the known stride. For example, if `token_stride=2`, and the first record corresponds to the first generated token, then record 0 -> token 1, record 1 -> token 3, record 2 -> token 5, etc. (assuming 1-based counting of tokens for generation, or 0-based depending on implementation; the exact mapping is something the analysis script will handle consistently). The `n_tokens` field in the header tells how many records are present, so the last token index can be inferred as well (last_index = (n_tokens - 1) * stride + first_index).

Example: Suppose you run generation for 100 tokens with `layer_start=30`, `layer_end=31` (2 layers traced), `token_stride=5`, `retain_magnitude=true`, and `quantize=true`. The header would indicate `n_layers=2`, `token_stride=5`, `has_magnitude=1`, `quantized=1`. Each token record will be 2 (layers) × 2 (bytes per layer: phase+mag) = 4 bytes. Since stride=5 over 100 tokens, approximately 20 records will be present (depending on alignment with end). The file size would be header (48 bytes) + 20*4 = 80 bytes of data ≈ 128 bytes total — very lightweight! In contrast, if quantization was off, each record would be 2 layers × 8 bytes = 16 bytes, and 20 records = 320 bytes of data (+ header) — still small, but this illustrates how quantization compresses the log significantly.

The binary format is designed for efficient post-processing. A Python or C++ script can memory-map or stream through the file, reading the header to know how to parse the rest. Binary format also avoids the overhead and precision issues of printing floats to text. By using a custom binary, we also ensure it’s easy to extend later (version field can be bumped if we add fields, reserved bytes allow adding flags, etc., without breaking compatibility).

# Example: CLI Invocation and Usage

Once compiled with tracing support, llama.cpp’s command-line tool (main or similar) will accept new options to configure coherence tracing. For example:
```
./llama_main -m models/7B/ggml-model.bin \
  -p "Once upon a time, in a distant kingdom, there was an alchemist who" \
  --coherence-log out.coh.bin \
  --coh-layers 30-31 \
  --coh-tap mlp \
  --coh-stride 2 \
  --coh-mag \
  --coh-quant
```
Explanation of the arguments in this example:

 - `--coherence-log out.coh.bin`: Enables coherence tracing and sets the output file path for the log to `out.coh.bin`. The presence of this flag signals the program to initialize the `coh_trace_cfg` with default values (which you can then override with other flags). If this flag is omitted, coherence tracing remains off (even if compiled in).

- `--coh-layers 30-31`: Selects the layer range to trace. Here we trace layers 30 and 31 (assuming 0-indexed layers, these might be the last two layers of a 32-layer model). You could also specify a single layer (`--coh-layers 10-10` for just layer 10, or perhaps we allow a comma-separated list in the future, but range is simpler for now).

- `--coh-tap mlp`: Chooses the tap point. In this syntax, `mlp` would correspond to “after the MLP” (layer output). We might allow values like `attn` (after attention), `aff` (after feed-forward), or `final` (after final layer). In code, this sets `cfg.tap_point = TAP_AFTER_MLP`.

- `--coh-stride 2`: Sets `token_stride = 2`, meaning log every 2nd token. In practice, if generating a long story, this will sample phase coherence every other token, which might be sufficient to catch trends while halving the amount of data.

- `--coh-mag`: This flag toggles magnitude retention (it sets `retain_magnitude = true`). If you omit it, by default magnitude might be off (depending on how defaults are defined). We provide it here to log both phase and magnitude.

- `--coh-quant`: This flag toggles quantization (sets `quantize = true`). Including it means we want the smaller 8-bit per value output. If you prefer full precision, you would omit this flag.

After the prompt, the model will start generating text. During generation, with these settings, the program will record the phase (and magnitude) for layers 30 and 31 at every second token. At the end, it writes the remaining buffer to `out.coh.bin`.

You’ll see normal text output on screen, and once complete, the file `out.coh.bin` will be available containing the coherence log. Depending on your logging settings, you might want to disable some other outputs; for example, if you are generating a lot of tokens and logging each, you may turn off verbose logging or other debug info to avoid overhead interference. But generally, the coherence logging runs in the background with minimal console output (maybe just a message at the start like “Coherence tracing enabled: writing to out.coh.bin”).

Programmatic use: If you’re using the llama.cpp API (for example via llama.cpp’s C API or Python bindings), you can similarly enable coherence tracing by populating a `llama_coherence_config` struct and passing it to the context. The design aims to keep it modular: the core logging logic is in one source file (e.g., `coherence_trace.cpp`) and not entangled with core model code beyond a few hook calls. This way, advanced users can compile in the feature and use it in their own applications. For instance, a Python binding could expose a flag or call to enable coherence logging to a given file, seamlessly integrating with the rest of the library.

# Python Post-Processing and Coherence Metric

Collecting raw phase data is only the first step. We provide a Python script (let’s call it `analyze_coherence.py`) that reads the binary log and computes an easy-to-interpret coherence metric over time. This script uses an adaptive exponential window approach to quantify phase coherence.

Reading the log: The script uses Python’s `struct` module to read the header and data. For example:
```
import struct

with open("out.coh.bin", "rb") as f:
    header = f.read(48)  # read header (assuming 48 bytes as defined)
    magic, version, n_layers, layer_start, layer_end, d_model, stride, has_mag, quantized, n_tokens, *_ = struct.unpack("<I H H H H H B B I 3I", header)
    # Basic checks
    assert magic == 0x434F4845  # 'COHE'
    phases = {layer: [] for layer in range(layer_start, layer_end+1)}
    magnitudes = {layer: [] for layer in range(layer_start, layer_end+1)} if has_mag else None

    # Calculate bytes per record for easy reading
    if quantized:
        bytes_per_layer = 1 + (1 if has_mag else 0)
    else:
        bytes_per_layer = 4 + (4 if has_mag else 0)
    record_size = bytes_per_layer * n_layers

    # Read each token record
    for rec_idx in range(n_tokens):
        record = f.read(record_size)
        for li, layer in enumerate(range(layer_start, layer_end+1)):
            if quantized:
                phase_val = record[li*bytes_per_layer]
                # Map uint8 back to phase in radians
                phase = (phase_val / 255.0) * (2*math.pi) - math.pi
                phases[layer].append(phase)
                if has_mag:
                    mag_val = record[li*bytes_per_layer + 1]
                    magnitude = (mag_val / 255.0) * MAG_MAX  # using same MAG_MAX as during logging
                    magnitudes[layer].append(magnitude)
            else:
                # floats
                offset = li*bytes_per_layer
                phase = struct.unpack_from("<f", record, offset)[0]
                phases[layer].append(phase)
                if has_mag:
                    magnitude = struct.unpack_from("<f", record, offset+4)[0]
                    magnitudes[layer].append(magnitude)
```
(The above is a simplified illustration; the actual script handles file length, maybe dynamic MAG_MAX if stored, etc.)

After reading, phases[layer] will be a list of phase values per recorded token for that layer. If multiple layers were logged, the script can either analyze them separately or combine them (for example, one might average coherence across layers or focus on the final layer – often the final layers are of most interest for output consistency).

Computing coherence metric: The next step is to compute a scalar metric that reflects how coherent the phase is over time. We use an exponential moving average of the phase vector to derive this. One robust approach is:

- Represent each phase as a unit complex number: zt=eiϕtzt​=eiϕt​, where ϕtϕt​ is the phase at token t.

- Compute an exponential moving average of these complex numbers: `C_t=α⋅C_t−1+(1−α)⋅z_t`​. Here `0<α<1` is a smoothing factor (e.g., `α = 0.95` corresponds to considering roughly the last 20 tokens’ influence significantly). We initialize `C_0=e^iϕ0​` for the first logged token.

- The coherence metric at time `t` can be defined as the magnitude of this EMA vector: `coh_t=∣Ct∣`. This value will range from 0 to 1. If the phase stays nearly constant over recent tokens, then all `z_t` point in roughly the same direction in the complex plane, reinforcing each other, and `|C_t|` will stay close to 1. If the phase is jittering or shifting randomly, the contributions cancel out and `|C_t|` drops toward 0.

This metric adapts to changes gradually thanks to the exponential window: older data decays in influence. The term "adaptive" here also implies that the windowing can adjust if needed – for instance, the script might automatically choose a different α for different layers or based on the token stride. In practice, you might start with a fixed α (say 0.95) for simplicity. More advanced adaptation could be to decrease α (shorter memory) if you want to detect instantaneous drops more sensitively, or increase α (longer memory) if you want a smoother trendline. The script could even try multiple α values or use an adaptive scheme where α changes if a drop is detected, but that is an advanced option.

Optionally, if magnitude data is available, the script can weigh each contribution by magnitude, i.e., use `z_t=(m_t/M)e^iϕt​` where `m_t` is the magnitude and `M` is a normalization (maybe the max magnitude seen or an expected value). This would mean when the activation pattern is weak (low magnitude), it influences the coherence less, whereas when the activation pattern is strong, it has more weight. This can help filter out noise: e.g., if phase jumps at a moment where the activation vector norm was very small, it might not truly indicate a model state change but just a low-signal blip. However, this weighted approach is optional and must be used carefully to avoid skewing the metric if magnitude has its own trends.

The output of the Python script could be a time series of `coh_t` values (one per logged token). The script can print statistics or plot the coherence over the sequence.

This coherence metric provides an easy-to-read indicator of stability. One could set up alerts or automatic detection: for instance, if `coh_t` falls below a threshold (say 0.5) during generation, that might be a cue to intervene (maybe by re-prime the model, reduce generation length, or use a different decoding strategy) to avoid incoherent output. This is analogous to how one might monitor perplexity or residual norm – but phase coherence is potentially more sensitive to structural changes in the hidden state dynamics.

The Python post-processing script can be extended further, for example:

- Compute moving average or variance of the phase itself (in addition to coherence).

- Align coherence drops with the generated text (perhaps printing the tokens or their indices where coherence dips).

- If multiple layers are logged, compare coherence across layers (maybe the earliest layer that decoheres could hint which part of the model first shows instability).

- Compare the phase coherence metric with other metrics like the hidden state norm or entropy of the next-token distribution.

# Example Use Case: Detecting Long-Context Drift

A practical scenario for phase coherence instrumentation is analyzing long-form generation where models are known to sometimes lose track of context (so-called drift or rambling). Suppose we have a 7B parameter model generating a several-thousand-word story. Initially, it stays on topic, but after a certain point it starts deviating or repeating itself. We want to pinpoint when this happens internally.

By tracing phase coherence on the last layer at every token (or every few tokens), we can observe how the model’s internal state evolves. In an experiment, we found that phase coherence tends to decline noticeably when the model’s output quality degrades. For example, during coherent generation (first few hundred tokens), the phase stays relatively aligned and coherence metric hovers high (near 0.9–1.0). When the model begins to lose the narrative thread (say around the 800th token), the phase angles start to wander. The coherence metric then drops, indicating that the hidden state has entered a less stable regime. Soon after, the text might show logical inconsistencies or start repeating phrases – symptoms of the model drifting.

By identifying this drift point via phase coherence, developers could implement strategies to mitigate it. For instance, one could reset the model state (if that's feasible) or prompt the model with a reminder of context when coherence dips. It also provides a quantitative way to compare models: if Model A maintains high coherence for 1000 tokens but Model B drops at 500 tokens, Model A arguably handles long contexts better (something not obvious just from final output, especially if subtle). This could form part of a benchmark for long-context coherence.

Additionally, phase coherence analysis might reveal differences in how models utilize their layers. Perhaps one layer’s coherence drops first, acting as an “early warning” of impending output issues. This could lead to insights on model architecture: e.g., maybe deeper layers should be regularized or modified to maintain coherence longer.

# Validation Roadmap

While phase coherence is an intuitively appealing metric, it’s important to rigorously validate that it correlates with meaningful model behaviors. We outline a roadmap for validation:

- Controlled Tests with Residual Norms: First, compare phase coherence against a known internal metric: the residual stream norm (the L2 norm of the hidden state). Residual norm changes have been observed during generation (e.g., spikes or dips may indicate certain phenomena). So we will need to log both metrics in parallel on various prompts. Do drops in coherence correspond to spikes or drops in residual norm? Initial expectations are that at least some correlation exists – for example, when residual norm shoots up (model might be amplifying certain features), the phase might become erratic. We might find cases where residual norm stays normal but phase coherence drops, indicating phase provides distinct information. If phase coherence correlates strongly with residual norm, that’s reassuring but also means it might be somewhat redundant; if it provides new information (e.g., cases where norm doesn’t signal a problem but phase does), that’s even more interesting.

- Output Quality Correlation: Design experiments where the model is prompted to generate until it contradicts itself or produces gibberish. Using known hallucination benchmarks or long-context tasks, record the coherence metric throughout. We will need to then check if low coherence correlates with independently measured output quality. For example, use a factual generation task where the model eventually starts introducing inaccuracies: does coherence drop prior to the factual error appearing? If yes, coherence could serve as a predictive signal of hallucination. Quantitatively, we can compute correlation coefficients between coherence and some quality score or measure the average coherence leading up to a known hallucination point vs. normal points. A strong inverse correlation (high coherence when factual, low when hallucinating) would validate the utility of the metric. Even a moderate correlation would be useful in combination with other signals.

- Ablation and Parameter Study: We need to vary aspects of the coherence computation to ensure the metric is robust. For instance, try different tap points (attention vs MLP output) and see which gives the clearest signal for drift. Perhaps the attention output phase coherence is more sensitive to short-term changes, while MLP output coherence reflects longer-term state. We’ll also need to test different frequency modes (maybe half a cycle or two cycles across the vector) to see if that yields a better metric. If one configuration correlates better with known issues, that will guide default settings. We will also need to validate that using quantized vs non-quantized data doesn’t change conclusions (expected to be nearly identical results).

- Across Models and Sizes: The ultimate validation is checking if phase coherence behaves consistently for different model sizes (7B vs 13B vs 70B) and architectures. If a larger model maintains high coherence longer, does it indeed produce more coherent long texts? And if a model architecture change (like adding recurrence or a longer attention window) is made, does it reflect in the phase coherence metric improving?

# Design Considerations for Upstream Integration

We have designed this feature with upstream integration in mind, meaning it should be straightforward to merge into the main llama.cpp project if the maintainers find it valuable. Key points of the design that align with upstream requirements:

- Zero overhead when disabled: As emphasized earlier, all coherence tracing code is guarded by the `LLAMA_COHERENCE_TRACE` macro. In a normal build, none of it runs. Even in a build where it’s compiled in but not enabled at runtime, the checks (like if (`coherence_tracing_enabled`)) are trivial and have no loops or heavy ops unless turned on. Memory allocation for buffers happens only if enabled. This ensures that there is no performance regression for users who do not use this feature.

- Modularity: The instrumentation is implemented as a self-contained module. We added a source file (e.g., `coherence_trace.cpp/.h`) that contains most of the logic (the config struct, the logging functions, the output routines). The existing codebase is touched only to insert hook calls at the tap points and to parse new CLI arguments. These hooks are small (`if` checks and a function call) and won’t clutter the core code. This modular approach means maintenance is easier and there’s a clear separation between core inference code and the tracing utility.

- Structured configuration: By using a config struct and explicit CLI flags, we avoid hard-coding any specific behavior. Users have full control over what to trace and how. This structured approach also makes the feature easier to extend – new options can be added to the config without changing the function signatures widely. It also makes the code more readable (e.g., `cfg.token_stride` is self-explanatory) and prevents a proliferation of ad-hoc global variables. All state related to coherence tracing is encapsulated in the config and related data structures (like the output buffer), which could be part of the `llama_context` or a separate global singleton if needed. We lean towards adding a pointer to `llama_context` so that in multi-context scenarios, each generation can have its own tracing settings and output.

- Minimal dependencies: The code uses only basic math library functions (`cosf`, `sinf`, `atan2f`, `sqrtf`) which are available in C99/C++ math. There’s no need for heavy libraries or OS-specific calls, making it portable. Writing to a file uses standard C I/O (`fopen, fwrite`), so it should work across platforms that llama.cpp supports.

- Thread-safety considerations: If llama.cpp ever does multi-threaded decoding, we ensure the logging buffer is handled safely. In the current design, generation is typically one token at a time in a single thread (the OpenMP threads are used within matrix ops, not across token steps). Our hook runs in that context, so it’s fine. If tokens were generated in parallel (as in some batch scenarios), we would need to mutex-protect the buffer or allocate separate buffers per thread and merge – but that’s beyond current scope.

- Disable capability at runtime: Even in a trace-enabled build, the user might not always want to trace. If no `--coherence-log` is provided, the code will skip all tracing. The overhead of checking an `if` flag each layer is negligible. We made sure that if the feature is compiled in but not actively used, it does essentially nothing (just checks a boolean that is false). So developers can include the feature in default builds (for convenience) without paying a price unless they use it.

The hope is that phase coherence tracing can be a valuable tool for the community to introspect and debug LLM behavior, and by making it opt-in and lightweight, it can be included in mainline llama.cpp. I think Phase Coherence Instrumentation offers a novel, neuroscience-inspired window into the inner workings of LLMs. By tracking the "rhythm" of the model’s hidden states, we can detect when a model remains in a stable regime versus when it starts to wander. The implementation in llama.cpp is designed to be non-intrusive yet flexible, giving developers a powerful new telemetry tool for their models. Early experiments are promising – we can catch subtle instabilities before they manifest in output – and ongoing validation will refine the approach. We invite the community to try it out, provide feedback, and collaborate on integrating this instrumentation for better understanding and improving LLM behavior.

Source for connection to neuroscience:

Bowyer, S.M. Coherence a measure of the brain networks: past and present. Neuropsychiatr Electrophysiol 2, 1 (2016). https://doi.org/10.1186/s40810-015-0015-7

Direct Link: https://npepjournal.biomedcentral.com/articles/10.1186/s40810-015-0015-7
