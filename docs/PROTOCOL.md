# Scope and decisions

This pilot adapts an existing MedSAM checkpoint on a small ultrasound subset. The official single-GPU training example updates the image encoder and mask decoder; this project deliberately freezes the image encoder to fit a smaller compute budget. It is an adaptation study, not a claim to reproduce all original experiments.

The fixed pilot consists of 300 lesion-bearing images, not 300 total including normal cases. Multi-lesion images are represented by the union of annotations and one enclosing box; this differs from evaluating each lesion separately. Test and validation prompts use exclusive-maximum xyxy boxes, with no jitter. Training jitter is in original image pixels, before coordinate scaling.

Exact-pixel duplicates are removed deterministically in sorted class/path order; the retained sample and exclusions are recorded. This can retain the first of conflicting annotations; audit exclusions before a formal study. Empty masks are excluded and missing/mismatched masks are errors. No patient IDs are invented and no automatic patient-independent claim is made. Near-duplicate views and pretraining overlap remain open limitations.

A study with real patient grouping requires verified metadata and a new group-aware split protocol; do not relabel this split as patient-level. A future negative-control experiment needs a separate, annotation-independent prompt policy for normal images.

Reproduce a run using the exact source revision, package environment, dataset files, manifest, checkpoint SHA-256, seed, and training configuration. The cache identity binds the preprocessing version, manifest, and starting checkpoint. Float32 cache arrays are portable; inspect stored environment metadata if moving runtimes. Generated reports must identify synthetic checks separately from BUSI outcomes.
