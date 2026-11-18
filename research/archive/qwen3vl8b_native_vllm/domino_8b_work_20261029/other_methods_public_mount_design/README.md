# Four remaining local methods: public-operator mount design

This is a source-only design for the trained **Domino, DFlash2, DFlare and DFly** parents, dated 2026-10-26. All four appear expressible with stock vLLM 0.22.1 layers/cache operators and public PyTorch tensor operations, but **none is implemented by the frozen B887 spatial adapter or by a config-only export**. Each needs a distinct, truthfully identified draft model; three also need predecessor-aware draft proposal sampling. The installed wheel has no inspected public proposer-registration API: a future mount needs an explicitly approved, source-bound draft integration hook as well as public operators. If only documented EngineArgs and model registration are allowed, that missing proposer/context hook blocks the complete mounts described here.

No Torch import, checkpoint-content read, model construction, forward, GPU query, training, export or production change was performed. Checkpoint hashes below are read from the small original retention records, not freshly recomputed. The earlier frozen inventory supplies the actual checkpoint tensor schema. This design does not authorize hardware execution or a benchmark identity substitution.

## Reference and feasibility order

Source aliases used below:

- **L** = local `/Users/Admin/grounding-parallel/a6000-v2-work/vlm_dflash/`.
- **R** = canonical remote `/home/lyh/VelaDraft-a6000-v2/vlm_dflash/`.
- **O** = installed `/home/lyh/miniforge3/envs/c1vllm/lib/python3.12/site-packages/vllm/`.
- **C** = `research/fair_official_backend_20261025/`.

The adjacent README.json records complete file paths, SHA256 values, source spans, configs/contracts/retention records and tensor schemas. All 21 compared method/factory/trainer/mask files match L and R byte for byte; shared `dflash/draft_model.py` differs (L `41f9f31404647fb82e1798469fd621ec6c116d91e33664a99dc1f8d8c12787d6`, R `6056a90681af091261813d70fa6b30ebb635f99901f6010e2dfd0edd9d64e8c7`). Six relevant helpers have equal normalized ASTs: context extraction, MRoPE merge, base context construction, GRU step, proposal sampler and block mask. The full `spec_generate` AST differs. A future parity oracle must choose the exact R driver; whole local/remote files are not interchangeable.

| Rank | Method | Retained tensors | Smallest additional draft work | Reason for order |
| --- | --- | ---: | --- | --- |
| 1 | Domino | 40 | GRU/projector modules; anchor-aware sequential sampler; shifted row selection | Leaves context, decoder, attention and KV math intact |
| 2 | DFlare | 42 | Raw-feature context transport; per-layer fusion/norm and separate target K/V precompute | Leaves query decoder and ordinary proposal sampling intact |
| 3 | DFly | 42 | Per-layer FC-plus-fusion context transport/precompute plus sequential hidden correction | Needs both a context path and a proposal path |
| 4 | DFlash2 | 51 | Six learned dynamic-convolution modules in decoder layers plus sequential top-16 selector | Changes decoder/residual execution as well as proposal sampling |

This is an implementation-feasibility ranking, not a performance, acceptance or numerical-parity prediction. DFlare and DFly share a useful context-transport design, but their equations and K/V weights differ.

All parents are step 37500 at `/home/lyh/a6000_3b_v2_L300/runs/<method>_L300_s42/train/ckpt_final.pt`. Their stat sizes match their retention records. No `export`, `export_hf` or `hf_export` directories were found at those four run roots.

| Method | Bytes | Recorded checkpoint SHA256 |
| --- | ---: | --- |
| Domino | 585879669 | `ea7aa9170725ce76648b8d0eb3ebf138dcb6c59f132f680791ab9eb5e874af91` |
| DFlash2 | 656958069 | `ad3629989034ed0fa7e58e0cbd5b57d4f64a69a9ae95843d44481e2fa54ba128` |
| DFlare | 468767669 | `66185a87886aad40dfb698e009f6d31c6a856f8a0a874cca7285419968e88937` |
| DFly | 529593845 | `8b8ea08e4da7e807dd7f9a9425e61ac8e6bf62afdf5771d0b71b2f6fef470bd8` |

## Common invariant and operator boundary

The actual contracts keep target Qwen2.5-VL-3B, hidden 2048, vocabulary 151936, three draft layers, 16 query/2 KV heads, head dimension 128, intermediate 11008, block 8 and mask 151669. Declared target layers [1,17,33] correspond to HF hidden-state indices [2,18,34], in order. The existing official aux capture supplies these same residual-added features; no extra target forward or final teacher-hidden capture is needed at serving time (`L/dflash/draft_model.py:832–840`; `O/v1/worker/gpu_model_runner.py:5241–5256`). Preserve the actual tied target embedding/output-head objects, full vocabulary, logit scale 1, no bias, BF16 and inference mode. A draft config's untied flag does not authorize inventing an independent head.

All four use spatial RoPE replacement, theta 1e6, sections [16,24,24], noninterleaved, and no learned visual query branch. Reuse the *math and invariants* of `C/spatial_adapter/official_spatial_adapter.py:60–112`: independent T/H/W coordinates, rejection-adjusted last-valid axes and separate absolute physical KV slots. Keep physical page16 distinct from runtime query B8. Context overwrite, rejected-prefix visibility and query-slot expansion must retain official metadata semantics. Additional target K/V matrices in DFlare project captured target **hidden states** into the draft cache; they are not target-model KV tensors.

These four attention masks remain **noncausal within the draft block**: each query sees the valid target prefix and all eight current query K/V rows. The causal GRU, convolution taps or predecessor walk do not make the attention JetSpec-causal. Actual configs omit the JetSpec flag; `L/dflash/draft_model.py:904,1337–1366` returns no extra mask. Training's packed/multi-anchor isolation is a training layout, not an instruction to connect unrelated serving requests (`L/dflash/mask.py:108–128,164–196`). Do not install JetSpec's three causal hooks for these methods.

The reusable stock pieces are `DFlashQwen3Attention`, QKVParallelLinear, output/MLP linears, RMSNorm, activation, ordinary Flex attention and the official per-layer KV writer. PyTorch embedding/indexing, linear, softmax, einsum, concatenation, elementwise gates, shifts/masks, topk and argmax express every added path; no authored CUDA/Triton kernel is indicated. Public operators do **not** imply a stable public integration ABI:

- `O/model_executor/models/registry.py:963–1004` exposes external model registration. Use a new architecture/algorithm marker; never overwrite ordinary DFlash's registry entry. The real EAGLEConfig wrapper prefixes non-DFlash architecture names under method='dflash' (`O/transformers_utils/configs/eagle.py:65–86`); registration must match the resolved name and validate the actual wrapper/type without model_type spoofing.
- `O/v1/worker/gpu_model_runner.py:584–586` directly constructs DFlashProposer. No analogous public proposer registry was found. Any subclass selection must be narrowly gated by the new method's exact architecture, marker, checkpoint and config, with source/origin checks and fresh-process installation.
- `O/v1/spec_decode/llm_base_proposer.py:450–461` requires a supported model type, calls combine_hidden_states and asserts width2048. A DFlashQwen3ForCausalLM subclass preserves the type relation, but per-layer fusion cannot pass untouched through its one-context ABI.
- `O/model_executor/models/qwen3_dflash.py:289–305,344–436` builds fused **noise/shared** K/V weight buffers and projects one normalized context through them. Its `attn.impl.do_kv_cache_update` call is an official, version-bound internal seam, not a documented stable plugin API. Flex implements it at `O/v1/attention/backends/flex_attention.py:1047–1068`.
- `O/v1/spec_decode/llm_base_proposer.py:504–515` gathers sample rows before calling a sampler that receives only hidden states and SamplingMetadata. It has no anchor-ID argument. The default greedy/local-argmax shortcut must not choose a token before a learned correction/selector executes.

A new bundle must independently bind these seams, retain B887's 3x(B+1) position-buffer stride, immutable real S, real requested=padded=B, initialized PIECEWISE dispatch, exception-safe dummy1..B scopes and shared-token-axis compile metadata. The current frozen installers are dependencies/evidence, not mutable extension points or authorization for extra heads.

## Domino

The actual config enables projector_type=domino, GRU hidden1024, bottleneck256, pure_draft_prefix_len=1 and shift_label=true. Besides all 36 backbone tensors, retain `prefix_gru.weight_ih_l0 [3072,2048]`, `weight_hh_l0 [3072,1024]`, `embed_proj.0.weight [256,3072]` and `embed_proj.2.weight [151936,256]`. All four extra tensors participate in inference (`L/dflash/draft_model.py:946–971`; factory `L/domino/factory.py:43–58`).

Use the exact existing bias-free GRU step: input/hidden linear projections split reset/update/new, r=sigmoid(ir+hr), z=sigmoid(iz+hz), n=tanh(in+r*hn), g=n+z*(g_previous−n). This is already expressed with public Torch operations and avoids a new cuDNN-packing convention (`L/dflash/draft_model.py:1115–1130`, R:1081–1096). Reset g to zero **each proposal round**, not each session and not once per engine.

For output position k=1..7, consume the actual previously selected token (the target bonus anchor for k=1), advance the GRU even for the pure-prefix step, and use hidden row k−1. Row0's proposal uses the ordinary target head only; rows1..6 add `W2*SiLU(W1*concat(h,g))` to base logits. Each selected token feeds the next GRU step. No teacher-forced token is available at inference (`L/dflash/draft_model.py:1315–1334`, R:1281–1300).

Minimal route: keep the official DFlash backbone/context/attention operators; register the four modules in a distinct model and strictly load them. A distinct draft proposer retains stable next_token_ids for the round and selects rows **0..6**, while retaining **all eight query rows** for noncausal attention. Merely calling stock sample_hidden_states loses row0; its normal indices are1..7. Compute target-head base logits before adding the full-vocabulary correction, preserving dtype/addition order; disable the pre-correction local-argmax shortcut. Only this draft sampler changes; target sampler/rejection/bonus verification remain stock. Reuse the source's explicit step rather than replacing the trained path with an unrelated GRU variant.

The trainer supervises eight shifted labels and mixes base/corrected CE through lambda_base; it sets the trainer's sample_from_anchor=true for label alignment (`L/domino/factory.py:74–80`; `L/domino/trainer.py:84–103,129–175,218–246`). That training flag does not make this saved draft model's serving sampler produce eight tokens. Its saved dflash_config omits sample_from_anchor; the model defaults false and its actual sampler returns seven. Lambda schedule, loss depth decay, anchor masks and optimizer state are training-only. Neither the GRU nor projector is training-only.

## DFlare

Retain 42 tensors: the base schema **without fc.weight**, plus learned `layer_fusion_weights [3,3]` and six `k_proj_target/v_proj_target [256,2048]` matrices. There is no missing FC to reconstruct or initialize as a substitute. For each draft layer l, let a_l=softmax(fusion_logits_l), and let H contain the three ordered captured target features. Its context is `c_l=hidden_norm(sum_t a_l,t*H_t)`. Use the saved live BF16 fusion tensor and the source's softmax/cast order, not optimizer FP32 masters or initial distance weights (`L/dflare/model.py:145–171,176–200`).

Context K/V are `Ktarget_l*c_l, Vtarget_l*c_l`; query/noise K/V retain the ordinary draft matrices. K normalization and 3-axis rotation follow projection, before writing the same layer's cache (`L/dflare/model.py:71–104`; shared attention `L/dflash/draft_model.py:393–420`). No target KV-cache extraction or extra target auxiliary layer is required.

Minimal route: retain raw [C,6144] features through a distinct draft context path, then per layer softmax/einsum, official RMSNorm, dedicated target K/V linears, K norm, unchanged spatial rotation and official KV-cache update. Stock combine_hidden_states irreversibly collapses them through one FC to2048; stock fused precompute uses the wrong shared/noise projection weights. Neither routine can be inherited unchanged. A narrow new proposer path must explicitly bypass that collapse/width assumption for its context payload; no global hidden-size lie or unverified model-side hidden-state side channel is proposed. Update dummy profiling to generate a correctly shaped three-feature context and execute every layer's real projection/norm path with no cache write when mapping=None.

Query decoder, QKV/noise projections, attention mask and ordinary rows1..7 sampling can remain stock. Strict loader/buffer construction must select dedicated context matrices instead of stock _fused_kv_weight, while retaining query K/V. Fusion initialization, trainable flags, initial-logit diagnostic buffer, FP32 optimizer master and entropy/drift reports are training/diagnostic concerns; the learned fusion logits and all six matrices are inference-essential (`L/dflare/trainer.py:29–67`).

## DFly

Retain all 42 tensors: all36 backbone tensors including FC, learned3x3 fusion, two correction-norm vectors and gate/up/down matrices [2048,4096], [2048,4096], [2048,2048]. This is not DFlare with six matrices dropped: the actual config explicitly uses **shared context/noise K/V**, and its context equation differs.

Compute base=FC(concat(H)); for layer l use `c_l=hidden_norm(base+sum_t softmax(a_l)_t*H_t)` (`L/dfly/model.py:101–156`). Apply each layer's ordinary K/V weights to its own context, then stock K norm/MRoPE/cache insertion. A single normalized FC context, a post-norm residual, or DFlare's fusion-only context changes this method. The raw-feature/per-layer context ABI work above is still necessary even though its K/V matrices are shared with queries.

For k=1..7, with p equal to the actual previously selected token, form u=concat(norm_h(h_k),norm_e(embed(p))) and `h'_k=h_k+down(SiLU(gate(u))*up(u))`; only then apply the unchanged target LM head and greedy selection. The next selected token feeds the next step (`L/dfly/model.py:46–95,167–209`). A stock independent head call cannot express this dependency.

Minimal route: distinct context-capable model/proposer plus a round-local anchor-aware correction sampler over ordinary rows1..7, using vLLM RMSNorm/linears/activation or the equivalent public Torch operations. Preserve separate norms, dtype conversions and residual order; no learned tensor is removable because down_proj started at zero. Precompute/dummy requirements mirror the per-layer context route; proposal head calls remain serial across seven slots unless a separate equivalent implementation is proved.

The actual run selects upstream_two_stage, with LK/D-PACE cold start then CE/TV stage, teacher final-hidden supervision and schedule metadata recorded in the inventory. These loss recipes, teacher-forced predecessor construction and diagnostic scalar reductions do not add serving weights (`L/dfly/factory.py:102–124`; `L/dfly/trainer.py:28–61`). The hidden correction used in training is still an essential inference module. Saved BF16 fusion/correction tensors, not diagnostic optimizer masters, define this trained parent.

## DFlash2

Retain all51 tensors: all36 backbone tensors; twelve convolution tensors (attention_conv and mlp_conv in each of three layers, each with base_kernel [2,2,2048] and kernel_projection [512,2048]); and selector predecessor/successor codebooks [151936,256] plus hidden_projection [256,2048]. Actual kernel2, group16, rank256 and top16 are inference parameters (`L/dflash2/model.py:28–157`; `L/dflash2/factory.py:44–72`).

Each attention and MLP sublayer is wrapped by prepare(input) and finish(output, saved_output_kernel). Prepare derives both dynamic kernels from the normalized **pre-convolution** input. For tap t, multiply shifted input x[i−t] by `base[t,channel]+delta[i,t,group]`, masking t>i%B so no tap crosses an anchor-block boundary. Finish reuses the output-side delta from prepare; it does not recompute it from sublayer output (`L/dflash2/conv.py:40–91,132–157`). This is token-conditioned grouped math, not a static Conv1d.

Minimal route: a distinct draft decoder layer keeps official RMSNorm, QKV attention, o_proj and SwiGLU/MLP operators, but inserts the six learned modules at the exact four per-layer locations. Preserve the official delayed residual convention: convolution finish must happen **before** residual addition/post-attention norm, and MLP finish before its residual addition. Stock layer forward cannot be used unchanged (`L/dflash2/model.py:54–96`; `O/model_executor/models/qwen3_dflash.py:194–213`). Construct each Attention/cache registration once; constructing stock layers and then replacing them with newly registered layers under the same prefix risks static-context duplication. Context FC/shared K/V precompute itself remains the ordinary path.

After one parallel B8 backbone, obtain unary target-head logits for h1..h7. At each slot, take the **unary** top16, then score candidate c by `unary(c)+sum((pred_code[p]*hidden_projection(h))*succ_code[c])`; select greedily and feed that candidate ID to the next slot (`L/dflash2/selector.py:47–86`; `L/dflash2/model.py:164–201`). The learned transition term must precede argmax; corrected full-vocabulary topk, beam/tree search or skipping the selector is a different policy. Use a distinct anchor-aware proposer sampler, ordinary full-vocabulary logits and public topk/gather/multiply/sum/argmax. No new target sampler is required.

Identity initialization is not a trained-value identity proof. The selector KL/CE, teacher final-hidden inputs, alpha/warmup/ramp/stop-gradient and chunk size are training-only controls, not reasons to omit any of its15 added tensors (`L/dflash2/trainer.py:47–102,166–198,237–249`). The actual strict profile does not inject a missing teacher target into serving candidates (`L/dflash2/loss.py:25–40`).

## Runtime budget and validation boundary

**S7/queryB8 is admitted by the inspected local inference sources for all four.** Preserve checkpoint/training block8. The ordinary public proposer geometry then supplies seven proposals and eight target-verifier positions, with the anchor retained. Domino maps hidden0..6 to the seven future proposals; DFlash2/DFlare/DFly use hidden1..7. Do not change query coordinates/KV addresses to implement Domino's label shift. In R, the shared serving sampler enforces block8 at1241–1300, round width is8 when sample_from_anchor=false at1404, query forward uses eight rows at1801–1829, and draft cache is cropped to start at1831. DFlash2/DFly independently enforce block8. The method-specific algorithm classes select their actual model classes despite generic architecture strings.

Domino trains eight shifted slots but the inspected serving policy walks seven; that narrower proposal policy is disclosed above. This supports a seven-proposal mount of the local method with every learned module retained; it does not authorize claiming an upstream eight-proposal runtime or silently removing a query row. An eight-proposal policy would require a separate query/verifier-geometry design. No S1/S3 reduction is proposed for these four. Reducing a noncausal query block alters hidden states; changing config.block_size or discarding a learned correction path is not a port.

Future implementation gates, after the root's active collection permits heavy work:

1. Rehash each canonical checkpoint and strictly export/reload **all**40/51/42/42 tensor values with an exact accepted-key set, original metadata and a distinct marker. Preserve all old source, adapters, exports, raw failures and benchmark identities.
2. Test each added computation against the exact local functional source, including adversarial predecessor IDs/GRU reset, Domino shift/pure-prefix boundary, dynamic tap/prepare-finish order, selector top16 transitions, per-layer fusion dtype/order and dedicated-vs-shared context K/V. Test train-only metadata cannot switch an inference path.
3. Exercise real config/type/registry resolution, custom proposer selection, target feature/head identity, raw-context/dummy shapes, actual official metadata and cache writer calls. Check target/no-marker engines remain unchanged. Test static-context uniqueness, graph metadata dictionary identity and capture/real/dummy guards on exceptions.
4. Only under a later root-controlled GPU gate, compare actual proposals/logits/masks/cache across rejection/page boundaries and graph capture/replay against the pinned reference, then fresh full-history target input/output/text/terminal compatibility. No small CPU mock proves those properties.
5. Timing or speed claims require a separately reviewed complete experiment with all learned paths included in measured endpoints. Source feasibility alone supports none.

The stock wheel inspected here is vLLM0.22.1, Torch distribution2.11.0 and Transformers5.11.0. This work read only text/small JSON and checkpoint file stats while formal GPU1 collection was active; no current weight bytes or image bodies were rehashed.
