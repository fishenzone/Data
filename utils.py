from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# %history -g -f history.txt
# optuna-dashboard --port 8006 sqlite:////_path_/wbs.db
# optuna-dashboard --port 8006 sqlite:///wbs.db

def extract_context(filename, keyword, context_len=1000):
    with open(filename, 'r') as f:
        text = f.read()

    with open('context.txt', 'w') as outfile:
        i = 0
        while i < len(text):
            index = text.find(keyword, i)
            if index == -1:
                break

            start = max(0, index - context_len)
            end = min(len(text), index + len(keyword) + context_len)
            context = text[start:end]

            outfile.write(f'index {index}:\n{context}\n\n')

            i = end

extract_context('history.txt', 'text')