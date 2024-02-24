df = pd.read_feather(paths[1])
dts = df.doctype.unique()

df = df[df.doctype==dts[1]].reset_index(drop=True)
df['paragraph_count'] = df['text'].apply(lambda x: len(x.split('\n')))
print(df.shape)

print(df['paragraph_count'].value_counts())
df = df[df['paragraph_count']==41].reset_index(drop=True)

df_test = df.iloc[1:].reset_index(drop=True)

df = df.iloc[:1]

model_names = {
    'MiniLM-L6': 'sentence-transformers/all-MiniLM-L6-v2',
    # 'LaBSE': 'LaBSE',
    'rubert-tiny2': 'cointegrated/rubert-tiny2'
}


models = {name: SentenceTransformer(model_names[name]) for name in model_names}

def measure_embedding_generation_time(model, paragraphs):
    start_time = time.time()
    embeddings = model.encode(paragraphs)
    end_time = time.time()
    return round((end_time - start_time), 3)

paragraphs = []
embeddings = []

def preprocess_paragraph(paragraph):
    paragraph = re.sub(r"\b\d{2}\.\d{2}\.\d{4}\b", "Дата", paragraph)
    paragraph = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "Время", paragraph)
    paragraph = re.sub(r"(\b[A-Z0-9]{10,}\b|\b\d{6,}\b)", "Код", paragraph)
    return paragraph

for text in tqdm(df['text'], desc="Processing documents"):
    for paragraph in text.split('\n'):
        processed_paragraph = preprocess_paragraph(paragraph)
        paragraphs.append(processed_paragraph)

        
times = {}
for model_name, model in models.items():
    time_taken = measure_embedding_generation_time(model, paragraphs)
    times[model_name] = time_taken