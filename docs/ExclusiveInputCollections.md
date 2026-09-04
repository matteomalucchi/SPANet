# Exclusive Input Collections

How the `assignment_source_exclusivity` option is realised inside the network, what it changes about
the learning problem, and when it is the right thing to do.

For the configuration syntax and a migration checklist, see
[`EventInfo.md`](EventInfo.md#exclusive-input-collections).

---

## 1. The starting point: one merged assignment space

SPANet is built around a single sequence of reconstructable vectors. Every `SEQUENTIAL` and `RELATIVE`
input is embedded independently and then **concatenated along the time axis** into one sequence
(`MultiInputVectorEmbedding.forward`, `spanet/network/layers/embedding/multi_input_vector_embedding.py:80`):

```
        JetHiggs (N_h vectors)          JetVBF (N_v vectors)
   ┌───────────────────────────┐  ┌───────────────────────┐
   │ 0   1   2   3   4   5     │  │ 6   7   8   9         │      merged index space, T = N_h + N_v
   └───────────────────────────┘  └───────────────────────┘
```

Everything downstream — the central transformer encoder, each particle's branch encoder, and the
symmetric attention that produces the assignment logits — operates on this one sequence and knows
nothing about which input a vector originally came from.

The `product: InputName` syntax in the event file was therefore only ever used to **offset the targets**
into the merged space when loading the dataset
(`spanet/dataset/jet_reconstruction_dataset.py:237`). A `JetVBF` target stored as `2` in your HDF5 file
becomes `N_h + 2` internally. Nothing constrained the *output*.

### What the assignment head produces

For an event particle with `k` decay products, `SymmetricAttentionSplit` runs `k` separate encoders over
the sequence, producing one embedding per product, and contracts them into a rank-`k` tensor
(`spanet/network/symmetric_attention/symmetric_attention_split.py:121`). For `k = 2`:

$$
O_{b,i_1,i_2} \;=\; \frac{1}{D}\sum_{a} y^{(1)}_{i_1,b,a}\, y^{(2)}_{i_2,b,a}
$$

so cell $(i_1, i_2)$ scores the hypothesis "product 1 is vector $i_1$ **and** product 2 is vector $i_2$".
The tensor is then symmetrised over the particle's permutation group
(`symmetric_attention_split.py:129`) and turned into a distribution by a **masked log-softmax over the
flattened $T^k$ cells** (`spanet/network/layers/branch_decoder.py:241`).

The key structural fact: the branch's output is one categorical distribution over *whole assignments*,
not $k$ independent per-product distributions. Constraints therefore live in the mask applied to that
distribution, and SPANet already uses this mechanism twice:

| Mask | Meaning | Built in |
| --- | --- | --- |
| Padding | do not assign a padded slot | `branch_decoder.py:123` |
| Diagonal | do not assign the same vector to two products of the same particle | `branch_decoder.py:123` |
| **Source** | **do not assign a product to a vector outside its declared input** | `branch_decoder.py:88` |

The source mask is the third member of that family. It is not a new architectural idea; it is the same
mechanism SPANet already uses to encode combinatorial facts that are known *a priori* rather than
learned.

---

## 2. How the source mask is built

**Step 1 — carry the provenance forward.** The embedding now also emits a vector
`input_index` of length `T` giving, for each position in the merged sequence, which input produced it
(`multi_input_vector_embedding.py:84`). This is a bookkeeping tensor of integers; it never enters any
linear layer and adds no parameters.

**Step 2 — build a per-product indicator.** Inside each branch decoder, for the product $d$ declared to
come from collection $c_d$:

$$
m^{(d)}_i \;=\; \mathbb{1}\!\left[\text{input}(i) = c_d\right],
\qquad
m^{(d)}_i \equiv 1 \ \text{ if product } d \text{ declares no input}
$$

**Step 3 — take the outer product over the products.** The per-product indicators are combined into the
same rank-`k` shape as the logits, with an einsum mirroring the existing diagonal-mask construction
(`branch_decoder.py:88`):

$$
S_{i_1 \ldots i_k} \;=\; \prod_{d=1}^{k} m^{(d)}_{i_d}
$$

**Step 4 — AND it into the output mask and normalise.** The final mask is
`padding ∧ diagonal ∧ source`, and the masked log-softmax fills every disallowed cell with $-\infty$
before normalising:

$$
\log P(i_1\ldots i_k) \;=\;
\begin{cases}
O_{i_1 \ldots i_k} - \log \displaystyle\sum_{(j_1 \ldots j_k)\,\in\,\mathcal{A}} e^{O_{j_1 \ldots j_k}}
  & (i_1 \ldots i_k) \in \mathcal{A} \\[2ex]
-\infty & \text{otherwise}
\end{cases}
$$

where $\mathcal{A}$ is the allowed set. **The denominator runs over the allowed cells only.** That single
detail is what makes this a change to the learning problem rather than a cosmetic filter, as explained
below.

The mask depends only on the sequence layout, so it is computed once and cached per
`(sequence length, device)`, exactly like the diagonal mask.

### Where it applies

Because the mask is applied inside `BranchDecoder.forward`, *before* the softmax, it is active in
**every** code path that touches the assignment distribution — the training loss, the validation
metrics, `predict.py`, `test.py`, the `interface.py` API, and a TorchScript/ONNX export. There is no way
for training and inference to disagree about the constraint, which is the reason for implementing it
here rather than as a post-processing filter on the predictions.

---

## 3. What it changes about the learning problem

The assignment loss is a plain cross-entropy on the masked distribution
(`spanet/network/utilities/divergence_losses.py:7`):

$$
\mathcal{L} \;=\; -\log P(\text{true assignment})
$$

Write $\mathcal{A}$ for the allowed set with the mask and $\mathcal{M}$ for the full merged set without
it ($\mathcal{A} \subset \mathcal{M}$). Then

$$
\mathcal{L}_{\text{merged}} = -O_{\text{true}} + \log\!\!\sum_{\mathcal{M}} e^{O}
\qquad\text{vs.}\qquad
\mathcal{L}_{\text{exclusive}} = -O_{\text{true}} + \log\!\!\sum_{\mathcal{A}} e^{O}
$$

Three consequences follow directly.

**(a) The network stops spending capacity learning something you already know.**
Without the mask, part of what the network learns from the data is simply *"a VBF quark is never a jet
from the `JetHiggs` collection"* — a fact that is true by construction of your dataset, in every event,
with no exceptions. Every gradient step spent pushing $O$ down on cells in $\mathcal{M}\setminus\mathcal{A}$
is a step not spent discriminating between the hypotheses you actually care about. Masking encodes the
fact exactly and for free, and the gradient on masked cells is identically zero
(`masked_softmax_no_gradient.py`), so no capacity leaks there.

**(b) Probability mass leaked onto impossible cells is mass taken away from the right answer.**
The softmax is normalised, so any probability the network assigns to an illegal combination is
subtracted from the legal ones — including the correct one. Learning $p \approx 0$ on those cells is
asymptotic: it is never exactly zero, and it is worst exactly where the network is least confident,
which is exactly where the assignment is hardest. With the mask, all mass is redistributed over the
legal hypotheses, so the score of the correct assignment is compared only against its real competitors.

**(c) The hypothesis space shrinks substantially.**
A particle with $k$ products restricted to a collection of $n$ vectors has $n!/(n-k)!$ ordered cells
instead of $T!/(T-k)!$. For a $HH \to 4b$ + VBF configuration with 6 `JetHiggs` and 4 `JetVBF` slots
($T = 10$), measured directly from the built masks:

| Branch | Allowed cells (merged) | Allowed cells (exclusive) | Reduction |
| --- | --- | --- | --- |
| `h1` (2 × JetHiggs) | 90 | 30 | 3.0× |
| `h2` (2 × JetHiggs) | 90 | 30 | 3.0× |
| `vbf` (2 × JetVBF) | 90 | 12 | 7.5× |

For the VBF branch, **87 % of the output cells the network had to suppress were physically impossible**.

---

## 4. Impact at inference

`extract_predictions` (`spanet/network/prediction_selection.py:169`) selects the final assignment
greedily: it takes the single highest-probability cell across all branches, fixes it, then **masks the
vectors it consumed out of every other branch**, and repeats.

This makes an unconstrained cross-collection error contagious. If the `vbf` branch's argmax lands on a
`JetHiggs` vector, that vector is not merely a wrong VBF answer — it is removed from the pool available
to `h1` and `h2`, so a single mistake in one collection can cascade into a wrong Higgs pairing as well.
With the source mask, the branches compete only within their own collection, and an error in the VBF
assignment cannot corrupt the Higgs assignment.

The output probabilities also become directly interpretable. `assignment_probability` in the prediction
file is now a probability over the physically meaningful hypothesis space, so it is usable as a per-event
confidence for the decision you are actually making, and a cut on it means the same thing in every event.

---

## 5. Why this is the right constraint for SPANet

**It restricts what may be *assigned*, not what may be *seen*.** This is the part worth stressing. The
central transformer encoder still attends over *all* vectors from *all* collections, and each branch
encoder still reads the whole sequence. The VBF assignment is still informed by the Higgs jets and vice
versa — which matters physically, since VBF jets are defined *relative* to the central system (large
$\Delta\eta_{jj}$, high $m_{jj}$, forward, away from the $b$-jets). The mask acts only on the final
output distribution.

That is what distinguishes this from the obvious alternative of training two separate networks, one per
collection. Separate networks would enforce the same exclusivity but destroy the cross-collection
context, and would also lose the global "each vector is used once" constraint across the event. Here you
get exclusivity **and** keep the joint reasoning.

**It is a hard constraint that is true by construction.** The collection a vector belongs to is a
property of how *you* built the input file, not something the network infers. If your `JetVBF` collection
is, say, the two most-forward jets outside the $b$-tagged selection, then a VBF quark's target index is
by definition an index into that collection. The truth conditional probability of a cross-collection
assignment is exactly zero — not small, zero. Masking a cell whose true probability is exactly zero
removes no representable solution and introduces no bias.

**It is consistent with the symmetry structure.** The assignment tensor is symmetrised over the
particle's permutation group before the mask is applied, so the mask must be invariant under that group
for the two to agree. This is why SPANet requires products related by a symmetry (and event particles
related by an event-level symmetry) to declare the same input, and validates it — `b1 ↔ b2` are
interchangeable only if they are drawn from the same pool.

**It costs nothing.** No parameters are added or removed, so existing checkpoints still load. The mask is
a cached boolean tensor and the runtime cost is a single elementwise AND per branch per forward pass.

---

## 6. Limitations and when *not* to enable it

- **It is hard, not soft.** If a VBF quark is genuinely best matched by a jet that your preprocessing put
  in the `JetHiggs` collection, the network can never recover it. The constraint is only sound if the
  collection split is a property of the *input construction* and your truth matching respects it. Check
  that no event in your training targets has an index that crosses collections before enabling it — a
  mismatch turns a recoverable error into an unlearnable one.

- **It does not help if the split is itself uncertain.** If deciding which jets are "the VBF jets" is
  part of the problem you want the network to solve, do not pre-split the input. Keep a single collection
  and let SPANet assign; the merged behaviour is the correct one in that case.

- **Retraining is required.** The option changes the normalisation of the loss. An existing checkpoint
  loads and will be constrained at inference, but it was trained to spread probability over cells that
  are now masked, so its logits are not calibrated for the smaller space. The benefit comes from
  training with the constraint in place.

- **ONNX export bakes the mask in at trace time**, as it already does for the diagonal mask. An exported
  multi-collection model must be fed the same per-collection padding it was traced with.

---

## 7. A one-paragraph summary

> Multiple input collections are concatenated into a single sequence internally, and SPANet's assignment
> head emits one categorical distribution over whole assignments of that sequence, constrained by masks
> for padding and for repeated jets. Declaring a decay product's input collection in the event file adds a
> third mask of the same kind: the outer product of per-product indicators over the collection each
> product belongs to, applied to the assignment logits before the log-softmax. The normalisation therefore
> runs over the legal combinations only, so the cross-entropy compares the correct assignment against just
> its physically possible competitors instead of also having to learn to suppress combinations that are
> impossible by construction — for the VBF branch of a $HH\to4b$ + VBF configuration this removes 87 % of
> the output cells. The transformer encoders still attend over every collection, so cross-collection
> context (which is what defines a VBF jet in the first place) is preserved; only the final assignment is
> restricted. The change adds no parameters.
