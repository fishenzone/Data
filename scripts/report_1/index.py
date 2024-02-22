def preprocess_paragraphs(texts, char_threshold=100):
    processed_texts = []
    for text in texts:
        paragraphs = text.split('\n')
        merged_paragraph = ''
        for paragraph in paragraphs:
            if merged_paragraph:
                potential_merge = f"{merged_paragraph} {paragraph}".strip()
                if len(potential_merge) <= char_threshold:
                    merged_paragraph = potential_merge
                else:
                    processed_texts.append(preprocess_paragraph(potential_merge))
                    merged_paragraph = ''
            else:
                merged_paragraph = paragraph
        
        if merged_paragraph:
            processed_texts.append(preprocess_paragraph(merged_paragraph))

    return processed_texts

