# ZONOS2

ZONOS2 support in `mlx-audio` targets converted MLX BF16 artifacts from
`Zyphra/ZONOS2`.

Initial scope:

- non-streaming text-to-speech generation
- 9-codebook Descript DAC decode at 44.1 kHz
- precomputed 2048-D `.npy`/`.npz` speaker embeddings
- dependency-free English text normalization for common written forms, with raw
  UTF-8 byte fallback for other languages and unsupported cases

Reference-audio speaker extraction and streaming decode are follow-up work.
