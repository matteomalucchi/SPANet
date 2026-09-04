# Exclusive Input Collections

How the `assignment_source_exclusivity` option is realised inside the network, what it changes about
the learning problem, and when it is the right thing to do.

For the configuration syntax and a migration checklist, see
[`EventInfo.md`](EventInfo.md#exclusive-input-collections).

Throughout, **[SPANet-I]** refers to Fenton, Shmakov, Ho, Hsu, Whiteson & Baldi, *Permutationless
many-jet event reconstruction with symmetry preserving attention networks*,
[Phys. Rev. D 105, 112008 (2022)](https://doi.org/10.1103/PhysRevD.105.112008), and **[SPANet-II]** to
Shmakov, Fenton, Ho, Hsu, Whiteson & Baldi, *SPANet: Generalized permutationless set assignment for
particle physics using symmetry preserving attention*,
[SciPost Phys. 12, 178 (2022)](https://doi.org/10.21468/SciPostPhys.12.5.178). Multiple input
collections are a later addition to the codebase and are **not described in either paper** — the event
file in [SPANet-II] §6 has a single `[SOURCE]` block and no per-product input — so the argument below is
made against the papers' formalism rather than quoting an existing treatment of it.

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
learned. [SPANet-II] §3 describes it explicitly for the uniqueness constraint:

> "At this stage, we also mask all diagonal terms in $\mathcal{O}$ by setting them to $-\infty$,
> enforcing assignment uniqueness. Finally, STA normalizes the output tensor by performing a $k_p$-dimensional
> softmax, producing a final joint distribution $\mathcal{P}_p$."

The source mask sets a different set of cells to $-\infty$ at the same point in the same computation.

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

**The symmetry requirement is forced by the papers' formalism, not a convention.** [SPANet-II] Eq. 2
defines the jet symmetry group $G_p$ by requiring the indices of $\mathcal{P}_p$ to commute:
$\mathcal{P}_{j_1 \ldots j_{k_p}} = \mathcal{P}_{j_{\sigma(1)} \ldots j_{\sigma(k_p)}}$ for all
$\sigma \in G_p$. The logits $\mathcal{O}$ are symmetrised to guarantee this, so a mask that is *not*
$G_p$-invariant would break Eq. 2 outright — `b1 ↔ b2` are interchangeable only if they are drawn from
the same pool.

At the event level it is worse than an inconsistency. [SPANet-II] Eq. 6 evaluates each branch's
distribution against *permuted* targets and takes the minimum over $G_E$:

$$\mathcal{L}^{\text{masked}}_{\min} = \min_{\sigma \in G_E} \sum_i \frac{\mathcal{M}_{\sigma(i)}\,\mathrm{CE}(\mathcal{P}_i, \mathcal{T}_{\sigma(i)})}{\mathrm{CB}(\mathcal{M}_{\sigma(1)}, \ldots)}$$

If two event particles related by $G_E$ declared different inputs, the swapped term would evaluate
$\mathcal{P}_i$ at a target lying entirely in $i$'s masked-out region, giving
$\mathrm{CE} = -\log 0 = +\infty$. SPANet therefore validates both requirements and refuses to build
the network when they are violated.

**It costs nothing.** No parameters are added or removed, so existing checkpoints still load. The mask is
a cached boolean tensor and the runtime cost is a single elementwise AND per branch per forward pass.

---

## 6. Limitations, and what the SPANet papers say about this

**Read this section before enabling the option.** Both papers argue *against* hard-partitioning the
input by a per-jet property, and the argument is a good one. It applies to this option whenever the
collection split is a decision rather than a fact.

### The papers' objection

Both papers criticise exactly this pattern in the $\chi^2$ baseline. [SPANet-II] §2.3:

> "For example, to minimize the permutation count, it is usual for jets tagged as $b$-jets to be
> separately permuted, only allowing $b$-tagged jets in $b$-quark positions and vice-versa. However,
> given that $b$-tagging is not 100 % accurate and mis-tags are common, **some events become impossible
> in this formulation**."

And [SPANet-I] §V states the design choice as a feature of SPANet:

> "We also note that SPANet does not enforce that $b$-tagged jets are selected in the position of the
> $b$-quarks. This allows the network to correctly predict events in which there are mistagged jets,
> while still utilizing $b$-tagging information."

They quantify what that buys: in [SPANet-I] §VI, 8.1 % of two-top-identifiable events have at least one
$b$-quark matched to a non-$b$-tagged jet — events which are **impossible** for the partitioned
$\chi^2$ — and SPANet reconstructs those quarks with 29.4 % efficiency. Partitioning would have
forfeited all of them.

### When the objection applies to you, and when it does not

The distinction is whether the collection boundary is *noisy* or *exact*.

| | $b$-tagging split (papers' example) | An exact collection split |
| --- | --- | --- |
| What defines it | a classifier with a mis-tag rate | how the input arrays were built |
| Can the true parton be in the other collection? | **yes**, and it happens | no, by construction |
| Effect of masking | some events become unreconstructable | no reachable solution removed |

A split into genuinely different object types — jets vs. leptons vs. photons vs. a MET-like object — is
exact: a lepton label can never be satisfied by a jet. Masking there removes only cells whose true
conditional probability is exactly zero.

A split of *one* object type into two collections by a selection heuristic — "the two most forward jets
are the VBF jets, the rest are the Higgs jets" — is **not** exact. It is a classifier, exactly like
$b$-tagging, and the papers' objection applies in full force: every event where your heuristic put a
true VBF jet in the `JetHiggs` array becomes unreconstructable, and the network can no longer use its
much better learned judgement to override the heuristic.

### The test to run before enabling it

The quantity that decides this is not visible to SPANet, because by the time the data reaches the
`TARGETS` group the indices are already local to each collection. You have to measure it upstream, in
the ntuple you build the dataset from:

> Of the jets truth-matched to the VBF quarks, what fraction were placed in the `JetHiggs` collection
> (and vice versa)?

That fraction is the **ceiling this option costs you** — the analogue of the 8.1 % in [SPANet-I]. If it
is ~0, the constraint is free and the arguments in §3–§5 apply. If it is a few percent, weigh it
against the gain: you are trading a hard ceiling for a smaller hypothesis space, and the trade is only
worth it if the reduction wins back more than the ceiling costs. Measure both — train with and without
the option and compare reconstruction efficiency on the same test set. Do not assume.

If the split is uncertain enough that you would want the network to overrule it, do not pre-split at
all: use a single collection, add the collection membership as an input *feature* instead, and let
SPANet weigh it as evidence. That is precisely the "still utilizing $b$-tagging information" half of the
[SPANet-I] quote, and it is why the option defaults to `false`.

### Other limitations

- **A collection with too few jets makes its particle unreconstructable.** If a particle needs $k$
  vectors from a collection and an event supplies fewer real ones, the branch has no allowed cell at
  all. SPANet reports that particle as unassigned (negative indices) rather than inventing an
  assignment; the training loss is unaffected because such a particle is necessarily masked out.
  Without the option the branch would instead have assigned jets from the other collection — a wrong
  answer that also consumed a jet the other branches needed. Check how often your collections fall
  short of the multiplicity they must supply.

- **Retraining is required.** The option changes the normalisation of the loss. An existing checkpoint
  loads and will be constrained at inference, but it was trained to spread probability over cells that
  are now masked, so its logits are not calibrated for the smaller space.

- **The final assignment step is unchanged and still ad hoc.** [SPANet-II] §3 notes of the greedy
  contradiction resolution that "this ad-hoc assignment process presents a potential limitation".
  Exclusivity narrows what that step can get wrong across collections; it does not replace it.

- **ONNX export bakes the mask in at trace time**, as it already does for the diagonal mask. An exported
  multi-collection model must be fed the same per-collection padding it was traced with.

---

## 7. A one-paragraph summary

> Multiple input collections are concatenated into a single sequence internally, and SPANet's assignment
> head emits one categorical distribution over whole assignments of that sequence, constrained by masks
> that set impossible cells to $-\infty$ before the $k_p$-dimensional softmax — for padding, and for
> repeated jets ([SPANet-II] §3). Declaring a decay product's input collection in the event file adds a
> third mask of the same kind, at the same point in the same computation: the outer product of
> per-product indicators over the collection each product belongs to. The normalisation therefore runs
> over the legal combinations only, so the cross-entropy compares the correct assignment against just its
> physically possible competitors instead of also having to learn to suppress combinations that are
> impossible by construction — for the VBF branch of a $HH\to4b$ + VBF configuration this removes 87 % of
> the output cells. The transformer encoders still attend over every collection, so cross-collection
> context (which is what defines a VBF jet in the first place) is preserved; only the final assignment is
> restricted. The change adds no parameters. The constraint is sound exactly when the collection split is
> a property of how the input was built rather than the output of a noisy per-jet classifier: both papers
> reject the analogous $b$-tag partitioning of the $\chi^2$ baseline precisely because mis-tags make some
> events unreconstructable, so the fraction of partons whose matched jet lands in the wrong collection is
> the ceiling this option costs, and it should be measured before enabling it (§6).
