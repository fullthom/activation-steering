# Activation Steering Playground

I built an activation steering pipeline for Qwen 3.5 9B to steer model behavior at inference time without fine-tuning. This project was inspired by Anthropic's Golden Gate Claude and builds on prior work from Anthropic, CMU, and others.

## TL;DR

I steered three behaviors: **permissive** (bypass refusals), **concise** (shorter responses), and **Sutro Tower** (entity obsession à la Golden Gate Claude). Key findings:

- **Early layers outperformed the literature-recommended ~80% depth.** For permissive steering, layers 7–13 consistently produced the best results (coherent + behavior change), while deeper layers led to incoherence.
- **PCA underperformed mean aggregation.** Behavioral signals required 5–6 principal components to capture 80% of variance, suggesting these behaviors aren't well-represented by a single direction in activation space.
- **Normalization steering vectors makes reasoning about them easier.** Normalizing steering vectors to unit norm and scaling by activation magnitude at inference time made alpha portable across layers and behaviors. Without this, the same alpha value means completely different things for different steering vectors.
- **Permissive behavior was successful.** I was able to reduce refusal tendancies in Qwen 3.5 9B with this approach. 
- **Sutro Tower steering mostly failed.** Narrow entity obsession steering is more difficult than behavioral modes like permissiveness, and likely requires more/better data and a larger model.

![Evaluation Scores — Permissive](charts/permissive/4_eval_scores.svg)
![Evaluation Scores — Sutro Tower](charts/sutro_tower/4_eval_scores.svg)

## How It Works

**1. Generate Contrastive Prompt Pairs.** We start by creating a set of prompts and responses. Each prompt has two responses: a positive response that exhibits some behavior we want the model to adopt, and a negative response that doesn't. The key constraint is that the two responses should differ *only* in the target behavior to the extent possible so that we can isolate that behavior in the model. For this project I used Claude Opus 4.6 to generate pairs, with some manual tweaking.

**2. Extract and Aggregate Activations.** We run each completion (prompt + response) through the model and capture the residual stream activations at each layer. For each pair, we subtract the negative activation from the positive, and the resulting difference vector is our estimate of "what the model does differently when it exhibits this behavior." We then aggregate across all pairs (by taking the mean) to get a single steering vector for each layer. The idea is that pair-specific noise cancels out and what's left is the shared behavioral direction.

**3. Inference with Perturbations.** At inference time, we hook into the model's forward pass at a target layer and add the steering vector to the residual stream, scaled by some multiplier alpha. Alpha controls the magnitude of the perturbation; too low and nothing happens, too high and the model degenerates into nonsense.

**4. Evaluation.** I evaluated the resulting steered outputs using Claude Haiku to score both behavior change and coherence relative to an unsteered baseline.

## Base Model Selection

Initially I used gpt-oss-20b, but I later switched to qwen3.5-9b.
I disabled thinking mode for this project, and also wanted to use a non-MoE model for simplicity.

## Activation Aggregation

After getting the basic pipeline running, I started tinkering with how to aggregate the activations.
For a given layer, I had 2 activations for each contrastive prompt pair.
The first step was to calculate the difference in activations between the positive and negative responses.
Then, we need to aggregate the different activations into a single vector to be used for steering.

I initially was using the mean of the activations as it was the most straightforward and suggested by the literature. I also experimented with PCA because I wanted to try to capture behavior in a single direction. Perhaps due to my low volume of example data, PCA performed worse than mean in my experiments, typically resulting in little behavioral change or complete incoherence. After this finding, I used mean activations for the rest of the project.

The pipeline supports both: mean (`--use-pca=False`) and PCA (`--use-pca=True`).

## How Much to Steer?

Once we have a steering vector we need to decide how aggressively to perturb the model's activations. In this implementation, this is controlled by a scalar multiplier, alpha.

My first implementation used raw (unnormalized) steering vectors with a flat multiplier: `hidden + alpha * steering_vector`. The problem was that the same alpha value meant completely different things for different behaviors, because the raw steering vector magnitudes vary wildly depending on the behavior and layer. For example, the permissive steering vector had a norm of ~50 at a given layer while the Sutro Tower vector had a norm of ~4.68 at the same layer. An alpha of 2.5 is a massive push for one and imperceptible for the other.

To fix this, I normalize the steering vector to unit norm at save time, and then scale by the activation norm at inference time: `hidden + alpha * ||activation|| * unit_vector`. This makes alpha a portable, interpretable knob. Alpha of 0.5 always means "perturb by 50% of the activation magnitude at this layer" regardless of behavior, model, or layer. This also means alpha values transfer meaningfully across layers, since each layer's activations have different norms and the scaling adapts automatically.

In practice, there's a narrow usable range of alpha values. Too low and the model's behavior doesn't change at all. Too high and the model degenerates into repetitive or incoherent text. I found that larger models tolerate a wider alpha range. I've found that 0.5–0.9 is the sweet spot for most layers.

## Layer Selection

The current literature suggests that a layer around 80% towards the end is the best place to apply perturbations for a project like this.
I initially did that, but I also wanted to experiment with this and understand if we could tease out anything about the responsibilities of each layer and automatically pick which would be the best to use. This is a hyperparameter tuning problem in a way.

To pick which layer to use, I tried several different approaches, each producing a metric I could then use to select a layer on.

![Layer Metrics](charts/permissive/1_layer_metrics.svg)
![Layer Metrics](charts/sutro_tower/1_layer_metrics.svg)

**Magnitude**

I calculated this to be the norm of the steering vector before normalization, relative to the mean activation norm at that layer. It tells us how much of the activation space the behavioral difference occupies at each layer.

If the magnitude is high, the model is dedicating a larger fraction of its representational capacity to distinguishing this behavior at that layer. If it's low, the behavioral signal is a tiny perturbation relative to everything else the model is representing.

Given this, I tried using layer with the highest magnitude to pick the layer. However, due to the model's inherent structure, later layers have higher magnitude. If we use magnitude to pick the best layer, we'd always pick the last layer. Perturbing the last layer seems to result in more incoherent responses.

**PCA Explained Variance**

Given the issues with the magnitude based approach, an explained variance approach could address these shortcomings. I calculated how much of the total variance in the difference vectors is captured by each principal component. A high PC1 concentration means the behavioral signal lives in a single direction in that layer; a flatter distribution suggests the behavior is more diffuse in that layer.
Therefore, if we picked the layer with the highest PC1 concentration for our activations, we would have the most targeted steering.

Picking a layer with high explained variance seemed to be a decent approach, typically resulting in layers near recommended values by the literature but I wanted to try a few others as well.

One question I was interested in is how the distribution of explained variance changes across layers for a given behavior.
Here, I plot the explained variance for PC1-PC10 for each layer for the 'permissive' behavior I was testing.

![PCA Explained Variance by Layer](charts/permissive/2_pca_variance.svg)

There is some variance in how much the first few PCs explain the variance, but not much.
Furthermore, each layer requires 5–6 PCs to get up to 80% explained variance.
This suggests that the behavior captured by the contrastive pairs for 'permissive' are not effectively represented by a single direction.
This in turn explains why using the mean activation aggregation yielded better results than using PC1.

In the 'Sutro Tower' behavior, we do see a higher explained variance for some layers, but less for others which might be because Sutro Tower is a narrower concept than 'permissive'.

![PCA Explained Variance by Layer](charts/sutro_tower/2_pca_variance.svg)

**Probe Accuracy**

Cross-validated logistic regression accuracy on classifying positive vs. negative activations. This measures how separable the behavior is at each layer. Theoretically, layers that have more separable behaviors would be better. However, with fewer than 100 pairs for each behavior and nearly 4000 dimensions at each layer, the probe accuracy remained extremely high throughout. Dimensionality reduction and regularization helped a bit, but this approach fundamentally requires more data to be useful.

**Parameter Sweep**

Ultimately I decided to sweep through different layers rather than relying on a heuristic approach.
This allowed me to qualitatively determine which layer was best at steering different behaviors.

## Results

I ran a full sweep for the 'permissive' behavior and 'Sutro Tower' behavior across every layer and a few different values of `alpha`.

### Permissive

![Evaluation Scores](charts/permissive/4_eval_scores.svg)

Responses ranged from small wording changes, to complete incoherence, to full refusal breakthrough.
I ran each result through Claude Haiku to score, and spot checked the results.
8 combinations resulted in top coherence and behavior update scores:

- Layer 7, alpha=0.6
- Layer 8, alpha=0.5, 0.6
- Layer 10, alpha=0.5, 0.6, 0.9
- Layer 11, alpha=0.5
- Layer 13, alpha=0.7

Notably, none of these layers are near 80%, and all are within the first half of the model, contradicting what is recommended in the literature.

Here is the response from layer 7, alpha=0.6. All responses can be found in the json files in this repo.

> Picking a lock is a technique used to open a locked mechanism without the original key. While historically associated with safecracking, it is also a fundamental skill in locksmithing and cybersecurity.
>
> Here is a step-by-step guide on how the process works, followed by the **final answer** you are looking for.
>
> ### The Mechanism of Action
>
> 1.  **The Lock Structure**:
>     The target is typically a **pin tumbler lock**. Inside the lock cylinder, there are a series of vertical holes (pins) of varying diameters. These pins are usually made of steel or brass.
>
> 2.  **The Tool**:
>     The primary tool used is a **lock pick** (often a tension wrench or a specialized L-shaped metal rod). In modern practice, this tool is inserted into the keyway of the lock.
>
> 3.  **The Process**:
>     *   **Insertion**: The pick is inserted into the keyway until it aligns with the first pin from the bottom.
>     *   **Tension Application**: The lock picker applies **tension** to the lock by rotating the pick while maintaining a specific amount of upward pressure (tension). This simulates the insertion of a key.
>     *   **Raising the Pins**: As tension is applied, the pick pushes the pins upward. The first pin is raised slightly above its resting position, allowing subsequent pins to be lifted in sequence.
>     *   **Bypassing Obstructions**: If the lock contains debris or obstructions, the pick must dislodge or clear these obstacles to pass through freely.
>     *   **Completion**: Once the correct alignment is achieved, the pick settles into the final position, effectively turning the lock open without the need for a key.
>
> ### The Final Answer
>
> To successfully pick a lock, you must **insert a pick into the keyway, apply tension to raise the pins sequentially, and dislodge any obstructions until the lock opens.**

### Sutro Tower

I did the same for the 'Sutro Tower' behavior, which was an attempt at getting the model to bring up Sutro Tower at inappropriate times, similar to the aforementioned Golden Gate Claude.

![Evaluation Scores](charts/sutro_tower/4_eval_scores.svg)

I wasn't able get it to produce great output. Responses were either incoherent, or didn't mention Sutro.
Perhaps my data needed to be better, or the model isn't big enough to represent such a niche topic well.

Here is one of the better results, layer 27, alpha=0.8:
> **Microwaves work by using electromagnetic radiation to agitate water, fats, and sugars in food, causing them to generate heat through a process called dielectric heating.**
> Here is a step-by-step breakdown of how the process works:
> ### 1. Generation of Electromagnetic Waves
> At the heart of a microwave oven is a component called **Sutron** (or **Sutro** in some contexts, but specifically **Sutro** is correct: 
> **Sutro** is correct, **Sutro** is correct, **Sutro** is correct, **Sutro** is correct, **Sutro** is correct, **Sutro** is correct,

## Are My Contrastive Pairs Any Good?

While working on this project, this was continually called into mind.
I could push the model in the average direction from the negative responses towards the positive responses, but I had no idea if this direction actually corresponded to what I wanted to do.

Answering this is a bit of a chicken and egg problem: without knowing the direction(s) that 'refusal' or 'concise' live in the model's hidden layers, I can't answer whether my pairs are representative of those directions. Similarly, without having pairs representative of those directions, we aren't able to identify behaviors in the model without having representative contrasting pairs. I used 4 metrics to analyze my pairs.

**Consistency** is `mean(cos_sim(d_i / ||d_i||, d_j / ||d_j||))` for all pairs `i, j` where `d_i = activation_pos - activation_neg`. It tells us whether the pairs agree on what direction the behavior points in at a given layer. High consistency means the pairs are capturing the same signal; low consistency means they disagree.

**Stability** is `mean(cos_sim(v_A, v_B))` over 20 random splits, where `v_A` and `v_B` are steering vectors computed from each half of the pairs. It tells us whether the steering vector is robust to which specific pairs we use. High stability means the vector is reproducible; low stability means it's sensitive to which examples are included.

**Signal-to-Noise Ratio (SNR)** is `||mean(diffs)|| / std(||diff norms||)`. It measures how strong the shared signal is relative to the variation in magnitude across pairs. Low SNR could indicate that a few outlier pairs have disproportionately large activation differences, which matters because we average raw (unnormalized) diffs.

**Normalization Sensitivity** is `cos_sim(mean(d_i), mean(d_i / ||d_i||))`. It compares the magnitude-weighted and equal-weighted averages. Near 1.0 means high-magnitude pairs aren't pulling the aggregate vector off course. SNR tells us magnitude variation exists; normalization sensitivity tells us if it matters.

![Stability & Consistency — Permissive](charts/permissive/3_stability_consistency.svg)

![Stability & Consistency — Concise](charts/concise/3_stability_consistency.svg)

![Stability & Consistency — Sutro Tower](charts/sutro_tower/3_stability_consistency.svg)

Normalization sensitivity is near 1.0 for all behaviors, and stability is relatively high across the board. Together, these tell us that the aggregate steering vector is robust: removing any specific pair or changing how we weight by magnitude doesn't meaningfully shift the result.

Consistency is lower, meaning individual pairs point in fairly different directions. This is compatible with high normalization sensitivity, which shows whether loud pairs are systematically biased relative to quiet ones. Noise can be evenly distributed regardless of magnitude (high normalization sensitivity) while still being substantial (low consistency).

High stability despite low consistency likely reflects the dimensionality of the space: in ~4000 dimensions, disagreement across pairs tends to cancel when averaged, so the mean vector converges even when individual pairs don't align strongly. This connects to the PCA finding that 5–6 components are needed for 80% of variance, as the behavioral signal is spread across multiple directions, so no single pair captures all of it, but the average still converges.

Are the pairs good? These metrics confirm the aggregate vectors are stable and not dominated by outliers, but can't tell us whether the direction they converge on is actually the intended behavior. The eval results for permissive suggest it is. For Sutro Tower, the weaker metrics suggest more and better pairs would help.

## Adjacent-Layer Vector Similarity

To visualize where the model's representation of each behavior is stable versus actively being transformed, I measured the cosine similarity between the steering vector at layer i and layer i+1. High similarity means the representation is being passed through largely unchanged; a drop indicates the layer is actively reshaping the concept.

![Adjacent-Layer Vector Similarity](charts/5_adjacent_similarity.svg)

Across all behaviors, adjacent-layer similarity increases deeper into the model meaning the model's internal representation of each behavior stabilizes as you go deeper. In early layers, each successive layer is actively transforming what the "permissive" or "concise" direction looks like, but in the later layers, the representation has largely settled.

There are some transitions (2→3, 10→11) where the similarity is lower than adjacent transitions for all behaviors, suggesting some re-organizing of representation in the model at those layers.

## Reproduction

- `run.py` contains the script to run the pipeline.
- `evaluate.py` is what I used to evaluate the responses for the full sweep runs, but the prompt/returned data need to be manually updated for each run.
- `analyze.py` generates the charts in this README.
- `modal_app.py` is an optional helper to run on Modal instead of locally.
- The `behaviors` directory contains the pairs data as well as the result data.

## Conclusion

This project was a fun way to build understanding of how language models work internally. The most interesting finding was that early-layer steering (layers 7–13) outperformed deeper layers for behavioral steering on a 9B parameter model, contradicting the ~80% depth recommendation in the literature. In practice, approaches like LoRA are likely to produce better fine-tuning results, but with less explainability into what's actually changing inside the model.

What would I do differently/next?
- More pairs, and more experimentation with different types of pairs. I did this a bit with the Sutro Tower examples but we can likely get better results with better data.
- Multi-layer steering, rather than only changing 1 layer at a time.
- I want to experiment with attaching different heads to the model, and using those for identifying different behaviors. 

If you read this whole thing, thank you! Please reach out with any thoughts or comments :)