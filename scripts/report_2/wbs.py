# prepare decoder and decode logits via shallow fusion
decoder = build_ctcdecoder(
    chars,
    kenlm_model_path=kenlm_model_path,  # either .arpa or .bin file
    alpha=0.02,  # tuned on a val set
    beta=.5,  # tuned on a val set
)

from tqdm import tqdm
all_pred_labels_wbs = [decoder.decode(prob, beam_width=300) for prob in tqdm(
    all_pred_probs, desc=f"Decoding with {kenlm_model_path}")]