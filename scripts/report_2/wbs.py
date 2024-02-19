# prepare decoder and decode logits via shallow fusion
decoder = build_ctcdecoder(
    chars,
    kenlm_model_path=kenlm_model_path,  # either .arpa or .bin file
    alpha=0.02,  # tuned on a val set
    beta=.5,  # tuned on a val set
)
