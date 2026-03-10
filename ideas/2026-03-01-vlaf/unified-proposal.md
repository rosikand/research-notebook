# VLAF: Video-Language-Action Fusion for Robot Learning

**A Unified Framework for Language-Conditioned Manipulation**

---

## Abstract

We present **VLAF** (Video-Language-Action Fusion), a unified framework for language-conditioned robot manipulation that bridges the gap between Vision-Language-Action (VLA) models and Video-Action Models (VAMs). Our framework introduces three architectural variants of increasing sophistication:

1. **VLAF-Gate**: Gated fusion combining frozen video and language backbones via learned static weights
2. **VLAF-AR**: Autoregressive interleaving that processes video-language token sequences through causal attention for dynamic cross-modal reasoning
3. **VLAF-AR+**: Full pipeline with VLA-CLIP trimodal contrastive pretraining followed by autoregressive interleaving

Our key insight is that effective robot learning requires both **representational alignment** (understanding that "pick up red cup", the video of picking up a red cup, and the corresponding action trajectory are semantically equivalent) and **sequential reasoning** (understanding how specific language tokens relate to specific video frames and action phases). VLAF-Gate provides efficient fusion for simple tasks; VLAF-AR adds dynamic reasoning for complex tasks; VLAF-AR+ adds explicit trimodal alignment for maximum performance and zero-shot capabilities.

We hypothesize that VLAF-AR+ will achieve state-of-the-art performance on complex manipulation benchmarks, VLAF-AR will excel on tasks requiring fine-grained cross-modal reasoning, and VLAF-Gate will remain optimal for real-time control scenarios. Together, these variants provide a complete toolkit for language-conditioned robot learning across the full spectrum of task complexity and computational constraints.

---

## 1. Introduction

### 1.1 Motivation

Language-conditioned robot manipulation requires bridging three modalities:

- **Language ($\ell$)**: Semantic understanding of goals ("pick up the cup behind the red block")
- **Video ($\mathbf{v}$)**: Visual perception of scenes, objects, spatial relations, and dynamics
- **Action ($a$)**: Motor commands that achieve the specified goal

Current approaches handle at most two modalities well:

| Approach | Semantics | Dynamics | Cross-Modal Reasoning |
|----------|-----------|----------|----------------------|
| **VLA** (RT-2, OpenVLA) | ✓✓ | ✗ | Implicit |
| **VAM** (Cosmos, UniVLA) | ✗ | ✓✓ | Temporal only |
| **CLIP-based** (R3M, VIP) | ✓ | ✓ | Static |

None provide the full combination of semantic understanding, physical dynamics, and dynamic cross-modal reasoning.

### 1.2 Key Insights

**Insight 1: Complementary Priors**

Video models pretrained on internet-scale data capture physics, motion, and dynamics. Language models capture semantics, reasoning, and world knowledge. Neither alone is sufficient; fusion is necessary.

**Insight 2: Static vs. Dynamic Fusion**

Simple tasks ("pick up the red cup") may only need static feature combination. Complex tasks ("pick up the cup behind the red block, then place it next to the blue bowl") require dynamic, sequential reasoning where each word attends to relevant visual regions.

**Insight 3: Alignment Before Reasoning**

Before learning to *reason* across modalities, the model should understand that video, language, and action representations of the same concept are equivalent. Contrastive learning provides this foundation.

**Insight 4: LLM Training Pipeline for Robotics**

Modern LLMs succeed via staged training: pretraining → SFT → RLHF. We adapt this paradigm:
- Contrastive pretraining (optional) → Behavioral cloning → RL fine-tuning

### 1.3 Contributions

1. **Unified VLAF Framework**: Three architectural variants spanning the complexity-performance spectrum
2. **VLA-CLIP**: First trimodal contrastive objective aligning video, language, and action
3. **Autoregressive Interleaving**: Dynamic cross-modal reasoning via causal attention
4. **Comprehensive Analysis**: Theoretical and empirical comparison of fusion strategies

---

## 2. Problem Formulation

### 2.1 Notation

| Symbol | Definition |
|--------|------------|
| $s_t = (o_t, p_t)$ | State: visual observation $o_t \in \mathbb{R}^{H \times W \times 3}$ + proprioception $p_t \in \mathbb{R}^{d_p}$ |
| $\mathbf{v}_{1:T}$ | Video sequence $(v_1, \ldots, v_T)$ |
| $\ell \in \mathcal{V}^*$ | Language instruction (token sequence) |
| $a \in \mathcal{A} \subseteq \mathbb{R}^{d_a}$ | Action (typically 7-DoF: 6-DoF pose + gripper) |
| $\pi_\theta$ | Policy parameterized by $\theta$ |

### 2.2 Objective

Learn a policy maximizing expected return under sparse rewards:

$$\pi^* = \arg\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{H} \gamma^t r(s_t, a_t, \ell) \right]$$

where $r \in \{0, 1\}$ indicates task success and $\gamma \in [0, 1)$ is the discount factor.

### 2.3 Challenges

1. **Semantic Grounding**: Mapping language concepts to visual entities
2. **Spatial Reasoning**: Understanding relations ("behind", "next to", "between")
3. **Temporal Alignment**: Coordinating language with action phases
4. **Physical Dynamics**: Predicting motion and contact outcomes
5. **Sample Efficiency**: Learning from limited robot demonstrations

---

## 3. VLAF Framework Overview

We propose three architectural variants:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VLAF Framework                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │   VLAF-Gate     │   │    VLAF-AR      │   │   VLAF-AR+      │          │
│   │                 │   │                 │   │                 │          │
│   │  Static Gated   │   │  Autoregressive │   │   VLA-CLIP +    │          │
│   │    Fusion       │   │  Interleaving   │   │  Autoregressive │          │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘          │
│            │                     │                     │                    │
│            ▼                     ▼                     ▼                    │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │ • O(M + N)      │   │ • O((M+N)²)     │   │ • O((M+N)²)     │          │
│   │ • ~50ms latency │   │ • ~150ms latency│   │ • ~150ms latency│          │
│   │ • Simple tasks  │   │ • Complex tasks │   │ • Zero-shot OK  │          │
│   │ • Real-time OK  │   │ • Dynamic attn  │   │ • Best overall  │          │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘          │
│                                                                             │
│            Simple ◄─────────────────────────────────────► Complex           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. VLAF-Gate: Gated Fusion

### 4.1 Architecture

VLAF-Gate combines frozen video and language backbones via a learned gating mechanism.

```
        Video v                    Language ℓ
           │                            │
           ▼                            ▼
    ┌─────────────┐              ┌─────────────┐
    │   Video     │              │  Language   │
    │  Encoder    │              │  Encoder    │
    │  (Frozen)   │              │  (Frozen)   │
    └──────┬──────┘              └──────┬──────┘
           │                            │
           │ h_v ∈ ℝ^d_v                │ h_ℓ ∈ ℝ^d_ℓ
           │                            │
           └──────────┬─────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │    Gated     │
              │   Fusion     │
              │  (Learned)   │
              └──────┬───────┘
                     │
                     │ h ∈ ℝ^d
                     ▼
              ┌──────────────┐
              │   Action     │
              │   Decoder    │
              └──────┬───────┘
                     │
                     ▼
                  Action a
```

### 4.2 Components

**Video Encoder** $f_v$: Cosmos-Predict2 DiT or VideoMAE

$$h_v = \text{Pool}(f_v(\mathbf{v}_{1:T})) \in \mathbb{R}^{d_v}$$

Pretrained with video diffusion:

$$\mathcal{L}_{\text{video}}^{\text{pretrain}} = \mathbb{E}_{t, \epsilon, \mathbf{v}} \left[ \| \epsilon - \epsilon_\phi(\sqrt{\bar{\alpha}_t} \mathbf{v} + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \right]$$

**Language Encoder** $f_\ell$: Llama-3 or CLIP text encoder

$$h_\ell = \text{Pool}(f_\ell(\ell)) \in \mathbb{R}^{d_\ell}$$

Pretrained with autoregressive language modeling:

$$\mathcal{L}_{\text{LM}}^{\text{pretrain}} = -\mathbb{E}_{\ell} \left[ \sum_{i=1}^{L} \log p_\phi(w_i | w_{<i}) \right]$$

**Gated Fusion Module** $g_\phi$:

$$h = g_\phi(h_v, h_\ell) = \alpha \odot \tilde{h}_v + (1 - \alpha) \odot \tilde{h}_\ell$$

where:

$$\tilde{h}_v = W_v h_v + b_v \in \mathbb{R}^d$$
$$\tilde{h}_\ell = W_\ell h_\ell + b_\ell \in \mathbb{R}^d$$
$$\alpha = \sigma(W_g [h_v; h_\ell] + b_g) \in (0, 1)^d$$

The gating weights $\alpha$ are learned and determine how much to rely on video vs. language features.

**Action Decoder** $\pi_\theta$:

$$a = \pi_\theta(h, p) \in \mathcal{A}$$

Options: MLP, Flow Matching, or Diffusion Policy.

### 4.3 Training

**Stage 1: Backbone Initialization**
- Load pretrained Cosmos and Llama weights
- Freeze both encoders

**Stage 2: Behavioral Cloning**

$$\mathcal{L}_{\text{Gate-BC}} = \mathbb{E}_{(\mathbf{v}, \ell, p, a^*) \sim \mathcal{D}} \left[ \| \pi_\theta(g_\phi(f_v(\mathbf{v}), f_\ell(\ell)), p) - a^* \|^2 \right]$$

Trainable: Fusion module $\phi$, action decoder $\theta$

**Stage 3: RL Fine-tuning**

$$J(\theta, \phi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{H} \gamma^t r(s_t, a_t) \right]$$

### 4.4 Properties

| Property | Value |
|----------|-------|
| Attention complexity | $O(M) + O(N)$ (separate) |
| Fusion type | Static learned gate |
| Inference latency | ~50ms |
| Best for | Real-time control, simple instructions |

---

## 5. VLAF-AR: Autoregressive Interleaving

### 5.1 Motivation

VLAF-Gate uses static fusion—the gating weights don't depend on the specific content of video or language. For complex instructions like "pick up the cup *behind* the red block", we need dynamic attention where:

- "cup" attends to cup-like regions in video
- "behind" triggers spatial reasoning
- "red block" filters by spatial constraint

VLAF-AR achieves this via autoregressive modeling over interleaved sequences.

### 5.2 Architecture

```
     Video v                           Language ℓ
        │                                   │
        ▼                                   ▼
 ┌─────────────┐                    ┌─────────────┐
 │   Video     │                    │  Language   │
 │  Encoder    │                    │  Encoder    │
 │  (Frozen)   │                    │  (Frozen)   │
 └──────┬──────┘                    └──────┬──────┘
        │                                   │
        │ z_v ∈ ℝ^(M×d)                     │ z_ℓ ∈ ℝ^(N×d)
        │                                   │
        └─────────────┬─────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Interleave  │
              └──────┬───────┘
                     │
                     ▼
     ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
     │v_1│ℓ_1│v_2│ℓ_2│...│v_M│ℓ_N│ACT│ p │
     └───┴───┴───┴───┴───┴───┴───┴───┴───┘
                     │
                     ▼
          ┌────────────────────┐
          │   AR Transformer   │
          │  (Causal Attention)│
          │     (Trained)      │
          └──────────┬─────────┘
                     │
                     ▼
              ┌──────────────┐
              │   Action     │
              │   Decoder    │
              └──────┬───────┘
                     │
                     ▼
                  Action a
```

### 5.3 Components

**Token Extraction** (from frozen encoders):

$$z_v^{\text{seq}} = f_v(\mathbf{v}_{1:T}) \in \mathbb{R}^{M \times d}$$
$$z_\ell^{\text{seq}} = f_\ell(\ell) \in \mathbb{R}^{N \times d}$$

**Interleaved Sequence Construction**:

$$x = \text{Interleave}(z_v^{\text{seq}}, z_\ell^{\text{seq}}) \oplus [\text{ACT}] \oplus \text{Embed}(p)$$

Explicitly:

$$x = [v_1, \ell_1, v_2, \ell_2, \ldots, v_{\min(M,N)}, \ell_{\min(M,N)}, \ldots, [\text{ACT}], p]$$

**Positional and Modality Embeddings**:

$$\tilde{x}_i = x_i + e_{\text{pos}}(i) + e_{\text{mod}}(m_i)$$

where $m_i \in \{\texttt{video}, \texttt{language}, \texttt{action}, \texttt{proprio}\}$.

**Autoregressive Transformer**:

$$h_t = \text{Transformer}(x_{\leq t})$$

With causal self-attention:

$$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + M_{\text{causal}} \right) V$$

where $M_{\text{causal}}$ masks future positions.

**Action Decoder**:

$$a_{t:t+k} = \pi_\theta(h_{[\text{ACT}]})$$

### 5.4 Training Objectives

**Action Loss** (primary):

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{(\mathbf{v}, \ell, p, a^*) \sim \mathcal{D}} \left[ \| \pi_\theta(h_{[\text{ACT}]}) - a^* \|^2 \right]$$

**Autoregressive Loss** (auxiliary):

$$\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{|x|-1} \log p_\theta(x_{t+1} | h_t)$$

Since video and language tokens may be in different embedding spaces, we use modality-specific prediction heads:

$$\mathcal{L}_{\text{AR}} = \underbrace{\sum_{t: x_{t+1} \in \mathcal{V}} \| g_\theta^{(v)}(h_t) - x_{t+1} \|^2}_{\text{video token prediction}} + \underbrace{\left( -\sum_{t: x_{t+1} \in \mathcal{L}} \log p_\theta^{(\ell)}(x_{t+1} | h_t) \right)}_{\text{language token prediction}}$$

**Auxiliary Video Prediction** (optional):

$$\mathcal{L}_{\text{aux}} = \| \text{Decode}(h_{v_t}) - v_{t+1} \|^2$$

**Total Loss**:

$$\boxed{\mathcal{L}_{\text{VLAF-AR}} = \mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{AR}} + \lambda_2 \mathcal{L}_{\text{aux}}}$$

### 5.5 Why AR Loss Helps

The autoregressive objective forces cross-modal grounding:

- To predict the next **language** token, attend to relevant video context
- To predict the next **video** token, understand the linguistic context
- This creates self-supervised signal for alignment

### 5.6 Properties

| Property | Value |
|----------|-------|
| Attention complexity | $O((M+N)^2)$ (joint) |
| Fusion type | Dynamic per-token attention |
| Inference latency | ~150ms |
| Best for | Complex reasoning, spatial relations |

---

## 6. VLAF-AR+: Contrastive Pretraining + Autoregressive Interleaving

### 6.1 Motivation

VLAF-AR must learn cross-modal alignment and sequential reasoning simultaneously. This is challenging because video and language tokens start in different embedding spaces.

VLAF-AR+ adds a **contrastive pretraining stage** (VLA-CLIP) that aligns video, language, and action in a shared space *before* the AR transformer sees them. This provides:

1. **Better initialization**: Tokens are pre-aligned
2. **Simpler AR objective**: Unified prediction (no separate heads)
3. **Zero-shot capabilities**: Action retrieval via embedding similarity

### 6.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 1: VLA-CLIP Pretraining                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│      Video v              Language ℓ              Action a                 │
│         │                      │                      │                     │
│         ▼                      ▼                      ▼                     │
│   ┌──────────┐           ┌──────────┐           ┌──────────┐               │
│   │  Video   │           │ Language │           │  Action  │               │
│   │ Encoder  │           │ Encoder  │           │ Encoder  │               │
│   └────┬─────┘           └────┬─────┘           └────┬─────┘               │
│        │                      │                      │                      │
│        ▼                      ▼                      ▼                      │
│   ┌──────────┐           ┌──────────┐           ┌──────────┐               │
│   │  Proj_V  │           │  Proj_L  │           │  Proj_A  │               │
│   │ +L2 Norm │           │ +L2 Norm │           │ +L2 Norm │               │
│   └────┬─────┘           └────┬─────┘           └────┬─────┘               │
│        │                      │                      │                      │
│        │ z_v                  │ z_ℓ                  │ z_a                  │
│        │                      │                      │                      │
│        └──────────────────────┼──────────────────────┘                      │
│                               │                                             │
│                               ▼                                             │
│                    Shared Embedding Space ℝ^d                               │
│                               │                                             │
│                               ▼                                             │
│                    L_VLA-CLIP = L_V-L + L_V-A + L_L-A                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ FREEZE
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Phase 2: Autoregressive Interleaving                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Frozen VLA-CLIP Encoders → Token Sequences (already aligned!)             │
│                               │                                             │
│                               ▼                                             │
│            ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐                           │
│            │v_1│ℓ_1│v_2│ℓ_2│...│v_M│ℓ_N│ACT│ p │                           │
│            └───┴───┴───┴───┴───┴───┴───┴───┴───┘                           │
│                               │                                             │
│                               ▼                                             │
│                    ┌────────────────────┐                                   │
│                    │   AR Transformer   │                                   │
│                    │     (Trained)      │                                   │
│                    └──────────┬─────────┘                                   │
│                               │                                             │
│                               ▼                                             │
│                    ┌────────────────────┐                                   │
│                    │   Action Decoder   │                                   │
│                    └──────────┬─────────┘                                   │
│                               │                                             │
│                               ▼                                             │
│                            Action a                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 VLA-CLIP: Trimodal Contrastive Pretraining

**Encoders**:

$$h_v = \text{Pool}(f_v(\mathbf{v})) \in \mathbb{R}^{d_v}$$
$$h_\ell = \text{Pool}(f_\ell(\ell)) \in \mathbb{R}^{d_\ell}$$
$$h_a = f_a(a_{1:k}) \in \mathbb{R}^{d_a}$$

**Projection Heads** (to shared space):

$$z_v = \frac{W_v h_v + b_v}{\|W_v h_v + b_v\|_2} \in \mathbb{R}^d$$
$$z_\ell = \frac{W_\ell h_\ell + b_\ell}{\|W_\ell h_\ell + b_\ell\|_2} \in \mathbb{R}^d$$
$$z_a = \frac{W_a h_a + b_a}{\|W_a h_a + b_a\|_2} \in \mathbb{R}^d$$

**Contrastive Losses** (symmetric InfoNCE):

$$\mathcal{L}_{\text{V-L}} = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{\exp(z_{v,i}^\top z_{\ell,i} / \tau)}{\sum_{j=1}^{B} \exp(z_{v,i}^\top z_{\ell,j} / \tau)} + \log \frac{\exp(z_{\ell,i}^\top z_{v,i} / \tau)}{\sum_{j=1}^{B} \exp(z_{\ell,i}^\top z_{v,j} / \tau)} \right]$$

$$\mathcal{L}_{\text{V-A}} = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{\exp(z_{v,i}^\top z_{a,i} / \tau)}{\sum_{j=1}^{B} \exp(z_{v,i}^\top z_{a,j} / \tau)} + \log \frac{\exp(z_{a,i}^\top z_{v,i} / \tau)}{\sum_{j=1}^{B} \exp(z_{a,i}^\top z_{v,j} / \tau)} \right]$$

$$\mathcal{L}_{\text{L-A}} = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{\exp(z_{\ell,i}^\top z_{a,i} / \tau)}{\sum_{j=1}^{B} \exp(z_{\ell,i}^\top z_{a,j} / \tau)} + \log \frac{\exp(z_{a,i}^\top z_{\ell,i} / \tau)}{\sum_{j=1}^{B} \exp(z_{a,i}^\top z_{\ell,j} / \tau)} \right]$$

**Total VLA-CLIP Loss**:

$$\boxed{\mathcal{L}_{\text{VLA-CLIP}} = \mathcal{L}_{\text{V-L}} + \mathcal{L}_{\text{V-A}} + \mathcal{L}_{\text{L-A}}}$$

### 6.4 Simplified AR Training

Because tokens are now in the same VLA-CLIP space, the AR objective simplifies:

$$\mathcal{L}_{\text{AR}}^{\text{unified}} = -\sum_{t=1}^{|x|-1} \log \frac{\exp(h_t^\top x_{t+1} / \tau)}{\sum_{j} \exp(h_t^\top x_j / \tau)}$$

This is **contrastive next-token prediction**: pull $h_t$ toward the true next token, push away from other tokens in the batch. No need for separate video/language heads!

**Total VLAF-AR+ Loss** (Phase 2):

$$\boxed{\mathcal{L}_{\text{VLAF-AR+}} = \mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{AR}}^{\text{unified}} + \lambda_2 \mathcal{L}_{\text{aux}}}$$

### 6.5 Zero-Shot Capabilities

VLA-CLIP enables zero-shot action retrieval:

**Given instruction, retrieve action**:
$$a^* = \arg\max_{a \in \mathcal{A}_{\text{database}}} z_\ell^\top z_a$$

**Given video, retrieve action**:
$$a^* = \arg\max_{a \in \mathcal{A}_{\text{database}}} z_v^\top z_a$$

**Cross-task transfer**: Tasks with similar language descriptions have similar action embeddings.

### 6.6 Properties

| Property | Value |
|----------|-------|
| Attention complexity | $O((M+N)^2)$ (joint) |
| Fusion type | Pre-aligned + dynamic attention |
| Inference latency | ~150ms |
| Zero-shot | ✓ Action retrieval |
| Best for | Maximum performance, cross-task transfer |

---

## 7. Unified Training Pipeline

All VLAF variants follow a staged training approach inspired by LLM training:

| Stage | VLAF-Gate | VLAF-AR | VLAF-AR+ | LLM Analogy |
|-------|-----------|---------|----------|-------------|
| **0** | — | — | VLA-CLIP pretraining | CLIP pretraining |
| **1** | Load pretrained backbones | Load pretrained backbones | Freeze VLA-CLIP encoders | Load embeddings |
| **2** | BC with gated fusion | Interleaved BC | Interleaved BC | SFT |
| **3** | RL fine-tuning | RL fine-tuning | RL fine-tuning | RLHF/RLVR |

### 7.1 Stage 0: VLA-CLIP Pretraining (VLAF-AR+ only)

**Data**: Robot demonstrations + unpaired video-text data (Ego4D, WebVid)

**Loss**: $\mathcal{L}_{\text{VLA-CLIP}}$

**Trainable**: All encoders + projection heads

**Duration**: ~500K steps

### 7.2 Stage 1: Backbone Initialization

**VLAF-Gate/AR**: Load pretrained Cosmos (video) and Llama (language)

**VLAF-AR+**: Freeze VLA-CLIP encoders

### 7.3 Stage 2: Behavioral Cloning

**Data**: Open X-Embodiment (~1M trajectories)

**Loss**:
- VLAF-Gate: $\mathcal{L}_{\text{Gate-BC}}$
- VLAF-AR: $\mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{AR}}$
- VLAF-AR+: $\mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{AR}}^{\text{unified}}$

**Trainable**:
- VLAF-Gate: Fusion module, action decoder
- VLAF-AR/AR+: AR transformer, action decoder

### 7.4 Stage 3: RL Fine-tuning

**Data**: Target task rollouts

**Objective**: $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{H} \gamma^t r(s_t, a_t) \right]$

**Algorithm**: TD3 or SAC

**Trainable**: Action decoder (+ optional LoRA on transformer)

**Target**: <100 episodes for adaptation

---

## 8. Theoretical Analysis

### 8.1 Comparison of Fusion Strategies

| Property | VLAF-Gate | VLAF-AR | VLAF-AR+ |
|----------|-----------|---------|----------|
| **Attention** | $O(M) + O(N)$ | $O((M+N)^2)$ | $O((M+N)^2)$ |
| **Fusion weights** | Static $\sigma(W)$ | Dynamic $\alpha_{ij}$ | Dynamic $\alpha_{ij}$ |
| **Cross-modal flow** | Unidirectional | Causal | Causal |
| **Alignment** | Implicit | Learned via AR | Explicit (VLA-CLIP) |
| **Zero-shot** | ✗ | ✗ | ✓ |

### 8.2 Information-Theoretic View

**VLA-CLIP** maximizes mutual information between modalities:

$$\mathcal{L}_{\text{VLA-CLIP}} \propto -I(Z_v; Z_\ell) - I(Z_v; Z_a) - I(Z_\ell; Z_a)$$

**AR objective** captures conditional dependencies:

$$\mathcal{L}_{\text{AR}} \propto -I(X_{t+1}; X_{\leq t})$$

Together, they capture **global alignment** (VLA-CLIP) and **local sequential dependencies** (AR).

### 8.3 Sample Complexity

Since backbones are frozen, trainable parameters are reduced:

$$d_{\text{trainable}} = d_{\text{fusion/AR}} + d_{\text{decoder}} \ll d_{\text{full}}$$

This yields improved sample complexity:

$$\mathbb{E}[\mathcal{L}(\hat{\theta})] - \mathcal{L}(\theta^*) \leq O\left( \frac{d_{\text{trainable}}}{N} \right)$$

VLA-CLIP further improves sample efficiency by leveraging unpaired data for alignment.

### 8.4 Emergent Reasoning in VLAF-AR

The autoregressive structure enables chain-of-thought-like reasoning. For "pick up the cup behind the red block":

1. Process "pick up" → attend to grasping-relevant video features
2. Process "cup" → identify cup candidates
3. Process "behind" → compute spatial relations
4. Process "red block" → filter by spatial constraint
5. $[\text{ACT}]$ → aggregate into action

This sequential reasoning is implicit in gated fusion but explicit in AR variants.

---

## 9. Implementation Details

### 9.1 Architecture Specifications

| Component | VLAF-Gate | VLAF-AR | VLAF-AR+ |
|-----------|-----------|---------|----------|
| Video Encoder | Cosmos-Predict2 (300M) | Cosmos-Predict2 (300M) | Cosmos-Predict2 (300M) |
| Language Encoder | Llama-3-8B | Llama-3-8B | Llama-3-8B |
| Action Encoder | — | — | 4-layer Transformer (10M) |
| Fusion/AR | 2-layer MLP (5M) | 12-layer Transformer (85M) | 12-layer Transformer (85M) |
| Action Decoder | 3-layer MLP (2M) | 3-layer MLP (2M) | 3-layer MLP (2M) |
| **Total Trainable** | ~7M | ~87M | ~97M (+ encoders for VLA-CLIP) |

### 9.2 Hyperparameters

| Parameter | VLA-CLIP | BC Stage | RL Stage |
|-----------|----------|----------|----------|
| Batch size | 2048 | 256 | 64 |
| Learning rate | 1e-4 | 1e-4 | 3e-5 |
| Optimizer | AdamW | AdamW | AdamW |
| Weight decay | 0.1 | 0.01 | 0.01 |
| Temperature $\tau$ | 0.07 | 0.1 | — |
| $\lambda_1$ (AR loss) | — | 0.1 | — |
| $\lambda_2$ (aux loss) | — | 0.01 | — |
| Training steps | 500K | 100K | 10K |

### 9.3 Pseudocode

**VLA-CLIP Forward Pass**:

```python
def vla_clip_forward(video, language, action):
    # Encode
    h_v = video_encoder(video)       # [B, d_v]
    h_l = language_encoder(language) # [B, d_l]
    h_a = action_encoder(action)     # [B, d_a]
    
    # Project and normalize
    z_v = F.normalize(proj_v(h_v), dim=-1)  # [B, d]
    z_l = F.normalize(proj_l(h_l), dim=-1)  # [B, d]
    z_a = F.normalize(proj_a(h_a), dim=-1)  # [B, d]
    
    # InfoNCE losses
    loss_vl = info_nce(z_v, z_l, tau)
    loss_va = info_nce(z_v, z_a, tau)
    loss_la = info_nce(z_l, z_a, tau)
    
    return loss_vl + loss_va + loss_la
```

**VLAF-AR Forward Pass**:

```python
def vlaf_ar_forward(video, language, proprio, action_gt):
    # Get token sequences from frozen encoders
    z_v = frozen_video_encoder(video)      # [B, M, d]
    z_l = frozen_language_encoder(language) # [B, N, d]
    
    # Interleave and add special tokens
    x = interleave(z_v, z_l)               # [B, M+N, d]
    x = concat([x, act_token, embed(proprio)], dim=1)
    
    # AR transformer with causal mask
    h = ar_transformer(x)                   # [B, M+N+2, d]
    
    # Action prediction
    h_act = h[:, -2, :]                     # [B, d]
    a_pred = action_decoder(h_act)          # [B, k, d_a]
    
    # Losses
    loss_action = mse_loss(a_pred, action_gt)
    loss_ar = ar_loss(h[:, :-1], x[:, 1:])
    
    return loss_action + lambda1 * loss_ar
```

---

## 10. Experimental Plan

### 10.1 Benchmarks

| Benchmark | Focus | Metrics |
|-----------|-------|---------|
| **CALVIN** | Long-horizon, language-conditioned | Avg chain length, success rate |
| **Language-Table** | Spatial reasoning | Success rate by relation type |
| **RLBench** | Diverse manipulation | Per-task success rate |
| **Real Franka** | Sim-to-real transfer | Success rate, robustness |

### 10.2 Baselines

- **OpenVLA**: VLA baseline
- **RT-2**: Large-scale VLA
- **Octo**: Diffusion policy
- **R3M + BC**: Contrastive pretraining baseline
- **Gato**: Generalist agent

### 10.3 Ablations

| Ablation | Purpose |
|----------|---------|
| VLAF-Gate vs. VLAF-AR vs. VLAF-AR+ | Compare fusion strategies |
| Remove VLA-CLIP pretraining | Importance of contrastive alignment |
| Remove AR loss | Importance of next-token prediction |
| Remove individual contrastive terms | Which pairings matter most |
| Interleave vs. concatenate | Importance of interleaving |
| Vary $\lambda_1$, $\lambda_2$ | Loss weight sensitivity |

---

## 11. Expected Results

### 11.1 Main Results (Projected)

| Method | CALVIN (Avg Len) | Language-Table | RLBench | Latency |
|--------|------------------|----------------|---------|---------|
| OpenVLA | 2.1 | 68% | 72% | 80ms |
| RT-2 | 2.4 | 74% | 78% | 200ms |
| Octo | 1.8 | 62% | 70% | 60ms |
| **VLAF-Gate** | 2.6 | 76% | 80% | **50ms** |
| **VLAF-AR** | 3.0 | 82% | 84% | 150ms |
| **VLAF-AR+** | **3.4** | **87%** | **88%** | 150ms |

### 11.2 Ablation Results (Projected)

| Configuration | CALVIN | Δ from VLAF-AR+ |
|---------------|--------|-----------------|
| **VLAF-AR+** (full) | 3.4 | — |
| − VLA-CLIP pretraining | 3.0 | -11.8% |
| − AR loss | 3.1 | -8.8% |
| − V-A contrastive | 3.2 | -5.9% |
| − L-A contrastive | 3.2 | -5.9% |
| Concatenate (no interleave) | 2.8 | -17.6% |
| **VLAF-AR** (no VLA-CLIP) | 3.0 | -11.8% |
| **VLAF-Gate** | 2.6 | -23.5% |

### 11.3 Zero-Shot Retrieval (VLAF-AR+ only)

| Task | Top-1 Accuracy | Top-5 Accuracy |
|------|----------------|----------------|
| Instruction → Action | 62% | 84% |
| Video → Action | 58% | 80% |
| Cross-task transfer | 45% | 71% |

---

## 12. When to Use Each Variant

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Decision Framework                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                         What's your priority?
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │  Real-time  │        │   Complex   │        │  Zero-shot  │
   │   Control   │        │  Reasoning  │        │  Transfer   │
   │   (>30Hz)   │        │             │        │             │
   └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │  VLAF-Gate  │        │   VLAF-AR   │        │  VLAF-AR+   │
   │             │        │             │        │             │
   │ • 50ms      │        │ • 150ms     │        │ • 150ms     │
   │ • Simple    │        │ • Dynamic   │        │ • Aligned   │
   │ • Static    │        │ • No zero-  │        │ • Zero-shot │
   │   fusion    │        │   shot      │        │ • Best perf │
   └─────────────┘        └─────────────┘        └─────────────┘
```

**Use VLAF-Gate when:**
- Real-time control is required (>30Hz)
- Instructions are simple ("pick red cup")
- Compute budget is limited
- You need a strong baseline

**Use VLAF-AR when:**
- Complex spatial reasoning is needed ("behind", "between")
- Dynamic cross-modal attention is valuable
- You don't need zero-shot capabilities
- Moderate compute is available

**Use VLAF-AR+ when:**
- Maximum performance is the goal
- Zero-shot transfer or retrieval is needed
- You have resources for contrastive pretraining
- Cross-task generalization matters

---

## 13. Hypotheses

**H1 (Fusion Strategy):** VLAF-AR+ > VLAF-AR > VLAF-Gate on tasks requiring complex spatial reasoning.

**H2 (Contrastive Pretraining):** VLA-CLIP pretraining improves downstream performance by ≥10% and enables zero-shot action retrieval with ≥60% top-1 accuracy.

**H3 (AR Objective):** The autoregressive loss improves cross-modal grounding, measurable via attention pattern analysis.

**H4 (Sample Efficiency):** VLAF-AR+ achieves equivalent performance to VLAF-AR with 50% fewer demonstrations due to better initialization.

**H5 (Latency-Accuracy Tradeoff):** VLAF-Gate maintains ≤50ms latency while VLAF-AR/AR+ achieve higher accuracy at ~150ms.

**H6 (Interleaving):** Interleaving outperforms concatenation by ≥15% on complex tasks due to fine-grained cross-modal attention.

---

## 14. Limitations and Future Work

### 14.1 Limitations

1. **Compute Cost**: $O((M+N)^2)$ attention limits real-time deployment for AR variants
2. **Data Requirements**: VLA-CLIP benefits from large-scale paired and unpaired data
3. **Modality Gap**: Some gap may remain despite contrastive learning
4. **Latency**: AR variants are ~3x slower than gated fusion

### 14.2 Future Directions

1. **Sparse Attention**: Reduce complexity to $O((M+N) \log(M+N))$
2. **Distillation**: Compress VLAF-AR+ to VLAF-Gate-sized model
3. **Additional Modalities**: Add tactile, audio, depth to VLA-CLIP
4. **Hierarchical Contrastive**: Frame-level + trajectory-level alignment
5. **Self-Supervised Actions**: Learn action encoder without labeled actions
6. **Adaptive Routing**: Dynamically choose Gate vs. AR based on task complexity

---

## 15. Conclusion

We presented **VLAF**, a unified framework for language-conditioned robot manipulation with three architectural variants:

1. **VLAF-Gate**: Efficient gated fusion for real-time control
2. **VLAF-AR**: Autoregressive interleaving for dynamic cross-modal reasoning
3. **VLAF-AR+**: Trimodal contrastive pretraining + AR for maximum performance

Our key contributions are:

- **VLA-CLIP**: First trimodal contrastive objective for robotics (V ↔ L ↔ A)
- **Autoregressive Interleaving**: Dynamic attention over interleaved video-language tokens
- **Unified Pipeline**: Staged training from pretraining to RL fine-tuning
- **Comprehensive Analysis**: Theoretical and empirical comparison of fusion strategies

The VLAF framework provides a complete toolkit spanning the complexity-performance spectrum. VLAF-Gate serves real-time applications; VLAF-AR handles complex reasoning; VLAF-AR+ achieves maximum performance with zero-shot capabilities. Together, they advance the state-of-the-art in language-conditioned robot learning.

---

## References

1. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
2. Brohan et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." CoRL 2023.
3. Kim et al. "OpenVLA: An Open-Source Vision-Language-Action Model." 2024.
4. NVIDIA. "Cosmos: World Foundation Models for Physical AI." 2024.
5. Black et al. "π₀: A Vision-Language-Action Flow Model for General Robot Control." 2024.
6. Nair et al. "R3M: A Universal Visual Representation for Robot Manipulation." CoRL 2022.
7. Ma et al. "VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training." ICLR 2023.
8. Reed et al. "A Generalist Agent." TMLR 2022.
9. Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
10. Ouyang et al. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS 2022.

---

## Appendix A: Complete Loss Summary

### A.1 VLAF-Gate

$$\mathcal{L}_{\text{Gate}} = \| \pi_\theta(g_\phi(f_v(\mathbf{v}), f_\ell(\ell)), p) - a^* \|^2$$

### A.2 VLAF-AR

$$\mathcal{L}_{\text{AR}} = \underbrace{\| \pi_\theta(h_{[\text{ACT}]}) - a^* \|^2}_{\text{action}} + \lambda_1 \underbrace{\left( \sum_{t \in \mathcal{V}} \|g^{(v)}(h_t) - x_{t+1}\|^2 - \sum_{t \in \mathcal{L}} \log p^{(\ell)}(x_{t+1}|h_t) \right)}_{\text{AR (separate heads)}} + \lambda_2 \underbrace{\|\hat{v}_{t+1} - v_{t+1}\|^2}_{\text{aux}}$$

### A.3 VLAF-AR+ (VLA-CLIP Stage)

$$\mathcal{L}_{\text{VLA-CLIP}} = \underbrace{-\log \frac{e^{z_v \cdot z_\ell / \tau}}{\sum_j e^{z_v \cdot z_{\ell,j} / \tau}}}_{\text{V-L}} + \underbrace{-\log \frac{e^{z_v \cdot z_a / \tau}}{\sum_j e^{z_v \cdot z_{a,j} / \tau}}}_{\text{V-A}} + \underbrace{-\log \frac{e^{z_\ell \cdot z_a / \tau}}{\sum_j e^{z_\ell \cdot z_{a,j} / \tau}}}_{\text{L-A}} + \text{symmetric}$$

### A.4 VLAF-AR+ (AR Stage)

$$\mathcal{L}_{\text{AR+}} = \underbrace{\| \pi_\theta(h_{[\text{ACT}]}) - a^* \|^2}_{\text{action}} + \lambda_1 \underbrace{\left( -\sum_t \log \frac{e^{h_t \cdot x_{t+1} / \tau}}{\sum_j e^{h_t \cdot x_j / \tau}} \right)}_{\text{AR (unified, contrastive)}} + \lambda_2 \underbrace{\|\hat{v}_{t+1} - v_{t+1}\|^2}_{\text{aux}}$$

### A.5 RL Fine-tuning (All Variants)

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{H} \gamma^t r(s_t, a_t) \right], \quad r \in \{0, 1\}$$

---

## Appendix B: Architecture Comparison

| | VLAF-Gate | VLAF-AR | VLAF-AR+ |
|---|---|---|---|
| **Video Encoder** | Cosmos (frozen) | Cosmos (frozen) | Cosmos (VLA-CLIP trained, then frozen) |
| **Language Encoder** | Llama (frozen) | Llama (frozen) | Llama (VLA-CLIP trained, then frozen) |
| **Action Encoder** | — | — | Transformer (VLA-CLIP trained, then frozen) |
| **Fusion** | Gated MLP | AR Transformer | AR Transformer |
| **Action Decoder** | MLP / Flow | MLP / Flow | MLP / Flow |
| **Trainable (BC)** | Fusion + Decoder | AR + Decoder | AR + Decoder |
| **Attention** | $O(M+N)$ | $O((M+N)^2)$ | $O((M+N)^2)$ |
| **Latency** | ~50ms | ~150ms | ~150ms |
| **Zero-shot** | ✗ | ✗ | ✓ |
| **Best for** | Real-time | Complex reasoning | Maximum performance |