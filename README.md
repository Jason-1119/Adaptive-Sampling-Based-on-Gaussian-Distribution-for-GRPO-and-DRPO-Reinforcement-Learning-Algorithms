# GRPO/DRPO checkpoint experiments v11

Upload the complete folder as MyDrive/GRPO_DRPO_Checkpoints_Project.

- 01_Train_GRPO.ipynb: train GRPO + adaptive and GRPO + matched uniform.
- 02_Train_DRPO.ipynb: train DRPO + adaptive and DRPO + matched uniform.
- evaluation_notebooks/: 16 independent target-specific notebooks, one per algorithm,
  sampler and rollout milestone, plus Evaluate_Base_0000.ipynb. Each mounts Drive
  and installs dependencies independently. You can evaluate ready checkpoints in
  separate Colab runtimes subject to account quota; duplicate writers are blocked.
- 03_Compare_Checkpoints.ipynb: compare completed results, paired correctness/answers,
  accuracy-vs-actual-rollouts and accuracy-vs-training-generated-tokens curves.

Defaults: DeepSeek-R1-Distill-Qwen-1.5B; full-parameter training; GSM8K train;
512 generated responses PER ARM (2048 across all four); max new tokens 4096;
one question per update batch; one optimization iteration per question.
Adaptive minimum 4, maximum 8, mu=0/sigma=5/multiplier=2:
P(continue)=min(1,2*Phi((-abs(correct-wrong)-mu)/sigma)). No future forcing.
Final budget tail alone may be below 4. All generated responses count.

The fixed arm preserves the prior matched near-uniform baseline. Adaptive for
that algorithm runs first; K observed questions determine floor(512/K) plus random
extras. Same question IDs/order and total final budget within each algorithm.
This is NOT independently prespecified fixed-4/fixed-6. Intermediate points and
different algorithms can cover different question counts and optimizer steps.
Record these confounders, tokens and wall time when interpreting results.

All training notebooks share one experiment config and pinned model/data revision.
Keep settings and EXPERIMENT_NAME identical. Change experiment name for new seeds
or settings; existing runs never overwritten. If configure runs simultaneously,
one may report a lock: rerun after the other finishes the brief configuration step.

GRPO is original response-normalized token-clipped PPO surrogate with population
group std and beta_ref=0, explicitly KL-free; no hidden reference model. With one
on-policy iteration the first scoring ratio is 1 and clipping is inactive before
the optimizer step. DRPO uses the previous decoupled objective and sampled old||new
penalty. Its pre-update KL may be zero; a separate post-update old/new K3 diagnostic
is logged for both methods. This diagnostic is not cumulative KL to the base, and
does not add a regularizer. All-same correctness groups can skip optimizer steps.
This is an auditable single-GPU implementation, not an official large-scale recipe.

Snapshots: runs/<experiment>/<algorithm>/<mode>/checkpoints/rollout_0128/ etc.
Each threshold is saved AFTER the first completed question/update reaching it;
READY.json/checkpoint_metadata.json contain actual rollout/token/step counts.
Threshold 0 is the pinned base, reloaded rather than duplicated to disk.
Evaluators require READY.json, written last. No intermediate group truncation.
Only final 512 budget may truncate allocation. Checkpoints include model/tokenizer
weights, training config and counters; NOT optimizer state. Mid-arm training resume
is unavailable. A full 1.5B snapshot is several GB; 16 require tens of GB storage.
Checkpoint save overhead is part of elapsed wall time, so this is not a pure
optimization-runtime benchmark. L4/A100 BF16 GPU required; A100 recommended.

Each target evaluation defaults to GSM8K test 128, greedy one answer/question.
An optional cell runs MATH-500 test 64 only when RUN_MATH=True. Frozen shared
question files include source revision and hashes. Exact train/test overlap is
checked; unknown pretraining contamination is not ruled out. Maximum response
4096 for all targets. Change EVAL_LABEL for changed evaluation settings.
Every completed answer is atomically saved, so evaluation can resume after a stop.
If a crashed session leaves a lock, verify that session stopped, then remove only
the named lock directory. Different targets write separate result directories.
Compare only identical question sets and generation settings. Results include
summary.json, answers.csv, full-text answers.html and per-model response JSON.
The aggregate notebook produces comparison.csv, paired_questions.csv and
learning_curves.png. These are TEST curves; do not tune/select models based on
repeated test viewing. Formal model selection needs an independent validation set.

CPU-only offline tests validate mathematical losses, optimizer changes, budgets,
checkpoint readiness, dataset freezing, independent outputs and notebook syntax.
No real Colab GPU training/evaluation has been performed in this workspace.
