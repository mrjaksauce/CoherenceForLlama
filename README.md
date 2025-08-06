# CoherenceForLlama
Phase Coherence Instrumentation for llama.cpp

Background and Rationale

Large language models can experience hidden state drift or instability during long sequence generation, leading to degraded output quality or hallucinations. Inspired by neuroscience, we introduce phase coherence as a new diagnostic of stability in llama.cpp. In brain networks, synchronized oscillatory activity (high phase coherence) is a sign of well-connected, stable communication between regions. Coherence metrics measure whether signals share a similar frequency and phase content. By analogy, measuring phase coherence in a transformer's internal activations can indicate whether the model’s layers are “in sync” or if the hidden representations start to drift out of alignment. A drop in phase coherence could flag moments when the model’s internal state becomes unstable, which may correlate with the onset of incoherent or hallucinated outputs. This approach complements traditional measures like residual norm magnitude by providing a phase-based view of the model’s internal dynamics.

Why phase? Neural network activations can be viewed as high-dimensional signals. We hypothesize that as long-context generation proceeds, hidden states maintain a consistent phase angle in some latent oscillatory mode, reflecting a stable encoding of context. If the model begins to lose context or stray (e.g. producing off-topic or repetitive text), this latent phase may shift erratically, reducing coherence. Phase coherence thus serves as a stability indicator: high coherence suggests the model is staying on track, while a sudden phase decoherence might signal the model is drifting from its narrative or factual grounding.

# Enabling Phase Coherence Tracing (Build Configuration)

To keep the overhead zero in normal use, the coherence tracing instrumentation is included behind a compile-time flag. You must build llama.cpp with the special flag enabled. A new CMake option ```LLAMA_COHERENCE_TRACE``` is provided:

```
cmake . -DLLAMA_COHERENCE_TRACE=ON
make
```
By default (flag off), none of the coherence-tracing code is included or executed, ensuring no performance impact or memory overhead on standard runs. When compiled with -```DLLAMA_COHERENCE_TRACE```, the llama.cpp binary will support additional options to configure and collect phase coherence data. This design is upstream-friendly – when the flag is disabled, the code is essentially a no-op (compiling to nothing) so it won’t affect existing functionality or speed. All tracing hooks are guarded with ```#ifdef LLAMA_COHERENCE_TRACE``` so they vanish completely in a normal build. Developers can thus merge this feature without worrying about regressions in core inference performance.

# Runtime Configuration (coh_trace_cfg)

When coherence tracing is enabled at compile time, you can control it at runtime through a structured configuration. We introduce a struct (in C/C++ code) called ```llama_coherence_config``` (or ```coh_trace_cfg``` for short) that holds all the relevant settings for phase coherence logging. This config can be set via new command-line arguments or through an API if llama.cpp is used as a library. The fields in the config include:
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
Layer range selection: You can choose a subset of layers to instrument by specifying ```layer_start``` and ```layer_end```. For example, you might trace only the last few layers of the model where most of the high-level pattern formation occurs. By default, you could set ```layer_start = 0``` and ```layer_end = N-1``` (for all N layers) or focus on a critical range (e.g., the final layer or a middle segment). Tracing fewer layers reduces overhead and log size.

Token stride control: The ```token_stride``` parameter controls temporal sampling. Not every token’s activations need to be logged – for long sequences, you might log one out of every N tokens. For instance, ```token_stride = 1``` logs every token (highest resolution), while ```token_stride = 4``` would log phase info for one token, skip the next three, then log again. This can significantly cut down the volume of data and overhead for very long generations, at the cost of some time resolution. Striding is useful when monitoring extended outputs (like thousands of tokens) where fine-grained detail might be overkill.

Tap point selection: The ```tap_point``` setting lets you choose where in the layer to extract the activations for phase analysis. A transformer layer has multiple sub-parts; common choices are:

- After attention output (pre-MLP): tapping the residual after the multi-head attention block (before the feed-forward MLP is applied).

- After MLP (layer output): tapping the residual after the feed-forward network (i.e., after the entire layer’s computations, usually just before adding to the next layer or before final layer norm).

- Final layer output: tapping the final hidden state (after the last layer, before the logits are computed).

By selecting the tap point, you can investigate phase coherence at different processing stages. For example, tapping after the MLP of each layer gives a view of each layer’s output phase progression. Alternatively, tapping after attention might isolate coherence changes due to attention mechanism. The tap is implemented as a hook in the forward pass – when the model reaches the chosen point in each selected layer for a given token, it triggers the logging routine (described below).

Magnitude retention toggle: The ```retain_magnitude``` flag controls whether we capture the activation vector’s magnitude information along with phase. If this is set to ```false```, the logging will only record the phase angle (essentially the directional information of the hidden state vector) and discard the length. If set to ```true```, the logging will record both the phase and the magnitude (in practice, the projected magnitude on the chosen phase axis – see next section). Retaining magnitude can provide additional insight: for instance, a phase might remain the same but the vector norm might collapse or spike, which is another indicator of potential instability. However, omitting magnitude reduces the data size and focuses purely on directional coherence. By default, we might recommend ```retain_magnitude = false``` for minimal output, unless the user specifically wants to analyze the interplay of phase and activation norm.

Quantization option: If ```quantize``` is enabled, the phase (and magnitude, if kept) are quantized to 8-bit values when stored, rather than full precision floats. Quantization dramatically shrinks log size – each value becomes a single byte – and speeds up I/O, with negligible impact on analysis in most cases. Phase naturally falls in a bounded range (-π to π), so we can map that continuous range to 256 discrete levels. Magnitude can be unbounded, but typically we deal with normalized activations (especially if using post-LN outputs); we handle magnitude quantization by clamping to a reasonable range or using dynamic scaling. If high fidelity is required for research, you can leave quantization off (```quantize=false```), which will store 32-bit floats for each value. The trade-off is file size and throughput versus precision.

Output buffering: The ```output_buffer_size``` sets how many token records to accumulate before writing to disk. Writing to disk can be slow if done token by token. Instead, the implementation uses a memory buffer to batch writes. For example, if ```output_buffer_size = 10```, the system will gather 10 tokens worth of phase data in memory and then flush them all at once to the output file. Larger buffer sizes improve performance by reducing I/O calls (especially important if ```token_stride``` is 1 and you are logging very frequently). However, setting it too large could risk losing a lot of data if the program crashes unexpectedly; a moderate default (e.g. 16 or 32 tokens) is used. The buffer is always flushed at the end of generation to ensure no data is left unwritten.

Finally, ```output_path``` specifies where to save the binary log. You provide a filepath (e.g., ```--coherence-log myrun.coh```) via CLI. If not provided, a default name like ```coherence_trace.bin``` might be used, but it’s good practice to specify it to avoid overwriting previous logs.

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

In the above pseudocode, you can see how we compute the phase. We treat the hidden state vector (of length ```d_model```) as a discrete signal over the “space” of neuron indices. We then project this signal onto a single cosine wave that spans the length of the vector. Specifically, we compute the dot product of the activation vector with a cosine and sine basis of one cycle across the vector indices. This yields two values, ```cos_sum``` and``` sin_sum```, which can be thought of as the real and imaginary components of the first Fourier mode of the activation pattern. The phase angle is then ```atan2(sin_sum, cos_sum)```, giving a value in ```-π,π``` that represents the dominant phase of the vector pattern relative to our reference cosine.

Adaptive frequency scaling: The choice of one full cycle across the vector (i.e., using ```omega = 2π/d_model```) is deliberate. It ensures that the phase measurement naturally adapts to the size of the model’s hidden dimension. For a model with ```d_model = 4096```, the cosine transform has a spatial frequency that completes one oscillation over 4096 elements; for a smaller model with ```d_model = 1024```, the cosine wave completes in 1024 elements. In both cases, we are capturing a comparable mode of variation (the lowest-frequency mode). This adaptive scaling means the phase angle is roughly comparable across models of different widths. In principle, one could experiment with higher-frequency modes (multiple oscillations across the vector) to capture finer spatial patterns, but the default of one-cycle-per-vector worked well in initial testing as a broad measure of coherence. It is a lightweight computation – O(N) per vector – focusing on the primary oscillatory component of the activation pattern rather than doing a full FFT or complex analysis.

Why cosine transform? The discrete cosine transform (DCT) is used instead of a full Fourier transform because our data (the activation values) are real-valued and we are mostly interested in a real-valued phase alignment. By using cosine and sine sums (effectively a single-frequency DFT), we get a 2D projection of the high-dimensional state onto a circular gauge (cos vs sin). The resulting phase angle captures the shape of the activation distribution across neurons, modulo that sinusoidal basis. If the activations shift in a systematic way (e.g. a subset of neurons increase while others decrease in a pattern that “rotates” the vector in this basis), the phase angle will shift accordingly. If the activation pattern stays consistent, the phase angle remains steady. In essence, this method condenses a high-dimensional change into a one-dimensional phase change. (Note: Because the reference waveform is fixed, the absolute phase value is somewhat arbitrary; it’s changes or stability in phase over time that carry meaning.)

Magnitude (amplitude) calculation: If ```retain_magnitude``` is ```true```, we also log the magnitude corresponding to that phase component. In the code, ```magnitude = sqrt(cos_sum^2 + sin_sum^2)``` gives the length of the projection of the activation vector onto the chosen cosine/sine basis. This is effectively the amplitude of that oscillatory component in the hidden state. If the overall activation vector norm shrinks or grows, this will reflect in the magnitude. We do not explicitly normalize the vector before computing phase, because the phase angle from ```atan2``` is scale-invariant (scaling the entire vector scales both ```cos_sum``` and ```sin_sum``` equally, canceling out in the ratio). However, extremely low magnitudes indicate that the phase is not well-defined (the vector had little projection on the basis, perhaps just noise). In such cases, the phase might fluctuate more (and coherence metric will account for it by weighting by magnitude or simply showing low confidence). If one wanted to strictly separate phase from amplitude, an optional normalization step could set the vector length to 1 before computing phase – but we left the raw computation to preserve real amplitude info for those who want it.

After computing phase (and possibly magnitude), the values are stored to a buffer (with optional quantization). Quantization works as follows:

Phase quantization: We map the phase from ```-π,π``` to an 8-bit unsigned integer ```0,255```. For example, -π corresponds to 0, 0 phase corresponds to ~128, and +π corresponds to 255. The mapping in code: ```phase_q = (phase + π) * (255 / (2π))```. This preserves relative differences in phase to about 1.4° precision, which is more than enough for our analysis.

Magnitude quantization: We need to decide a scale. If using post-layer-norm activations, the vector norm might be around 1 on average. But to be safe for spikes, we choose an upper bound (say ```MAG_MAX```, e.g. 10 or a dynamic max) and map magnitude linearly into 0–255. Any magnitude above the cap is clamped to 255. For example, ```mag_q = magnitude * (255 / MAG_MAX)```. The quantization error for magnitude is usually not an issue for our purposes (we care about relative changes, not absolute precision to many decimal places). If magnitude is not retained, this step is skipped.

The buffer accumulates bytes (if quantized) or floats (if not) for each logged value. The ```record_size``` in the pseudocode refers to the number of bytes per token per layer (e.g. if logging phase+mag as bytes for one layer, ```record_size = 2 bytes```; if floats for one layer, ```record_size = 8 bytes```, etc., and if multiple layers, multiply that accordingly). When the buffer is full (reaches ```output_buffer_size * record_size bytes```), it writes out to disk and clears the buffer. Writing is done in binary mode to the file path specified in ```cfg.output_path```.
