import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import os
import glob
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import process

# Load summaries and calculate similarity matrix
excel_files = [os.path.basename(f) for f in glob.glob("*.xlsx")]
all_summaries = []
summary_source = {}

for file in excel_files:
    df = pd.read_excel(file)
    for idx, row in df.iterrows():
        summary = row['Summary']
        translated = row['Translated'] if 'Translated' in df.columns else ''
        all_summaries.append((summary, file))
        summary_source[summary] = (translated, file)

pkl_files = glob.glob("*.pkl")

if pkl_files:
    with open(pkl_files[0], "rb") as f:
        similarity_matrix = pickle.load(f)
else:
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode([s[0] for s in all_summaries])
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix_file = "summary_source_2023-04-29_06-26-17_8_files.pkl"
    with open(similarity_matrix_file, "wb") as f:
        pickle.dump(similarity_matrix, f)

def find_top_similar_sentences(phrase, summaries, similarity_matrix, top_k=5):
    index = [i for i, s in enumerate(summaries) if s[0] == phrase][0]
    similarities = similarity_matrix[index]
    top_indices = np.argsort(similarities)[-top_k - 1: -1][::-1]
    top_similar = [(summaries[i], similarities[i]) for i in top_indices if i != index]
    return top_similar

def display_results_frame(parent, title, sentences):
    frame = ttk.Frame(parent)
    label = ttk.Label(frame, text=title)
    label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    
    text_widget = tk.Text(frame, wrap=tk.WORD, width=60, height=20)
    text_widget.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    scroll_bar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
    scroll_bar.grid(row=1, column=1, padx=10, pady=10, sticky="ns")
    text_widget.configure(yscrollcommand=scroll_bar.set)

    for sentence, score, source in sentences:
        text_widget.insert(tk.END, f"{sentence}\n", "bold")
        text_widget.insert(tk.END, f"Source: {source}\n", "source")
        text_widget.insert(tk.END, f"Similarity Score: {score:.2f}\n", "score")
        if summary_source[sentence][0]:
            text_widget.insert(tk.END, f"Translation: {summary_source[sentence][0]}\n\n", "translation")

    text_widget.configure(state='disabled')
    text_widget.tag_configure("bold", font=tkFont.Font(family="Helvetica", size=10, weight="bold"))
    text_widget.tag_configure("source", font=tkFont.Font(family="Helvetica", size=10, slant="italic"))
    text_widget.tag_configure("translation", font=tkFont.Font(family="Helvetica", size=10))
    text_widget.tag_configure("score", font=tkFont.Font(family="Helvetica", size=10, underline=True))

    return frame

def search():
    input_phrase = search_box.get()
    top_k = int(top_k_box.get())

    summaries = [s[0] for s in all_summaries]
    if input_phrase not in summaries:
        suggested_phrase, _ = process.extractOne(input_phrase, summaries)
        result_label.config(text=f"Phrase not found. Did you mean: {suggested_phrase}?")
        return

    top_similar = find_top_similar_sentences(input_phrase, all_summaries, similarity_matrix, top_k=top_k)

    # Clear previous results
    for frame in result_frames.values():
        frame.destroy()

    # Separate top similar sentences from the same file and other files
    same_file_sentences = []
    other_file_sentences = []
    search_quote_file = summary_source[input_phrase][1]

    for summary, score in top_similar:
        sentence, source = summary
        if source == search_quote_file:
            same_file_sentences.append((sentence, score, source))
        else:
            other_file_sentences.append((sentence, score, source))

    # Display the search quote along with its source
    search_quote_frame = display_results_frame(root, "Search quote:", [(input_phrase, 0, search_quote_file)])
    search_quote_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
    result_frames["search_quote_frame"] = search_quote_frame

    # Display top similar sentences from the same file
    same_file_frame = display_results_frame(root, f"Top {top_k} similar sentences from the same file:", same_file_sentences)
    same_file_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
    result_frames["same_file_frame"] = same_file_frame

    # Display top similar sentences from other files
    other_files_frame = display_results_frame(root, f"Top {top_k} similar sentences from other files:", other_file_sentences)
    other_files_frame.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")
    result_frames["other_files_frame"] = other_files_frame

# UI elements
root = tk.Tk()
root.title("Sentence Similarity Search")
root.geometry("1800x700")

search_label = ttk.Label(root, text="Enter a phrase from a summary:")
search_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

search_box = ttk.Entry(root, width=50)
search_box.grid(row=0, column=1, padx=10, pady=10, sticky="w")

top_k_label = ttk.Label(root, text="Number of similar sentences:")
top_k_label.grid(row=0, column=2, padx=10, pady=10, sticky="w")

top_k_box = ttk.Entry(root, width=10)
top_k_box.grid(row=0, column=3, padx=10, pady=10, sticky="w")
top_k_box.insert(tk.END, "5")

search_button = ttk.Button(root, text="Search", command=search)
search_button.grid(row=0, column=4, padx=10, pady=10, sticky="w")

search_button = ttk.Button(root, text="Search", command=search)
search_button.grid(row=0, column=4, padx=10, pady=10, sticky="w")

result_label = ttk.Label(root, text="")
result_label.grid(row=1, column=0, padx=10, pady=10, columnspan=5)

def clear_results():
    for frame in result_frames.values():
        frame.destroy()

clear_button = ttk.Button(root, text="Clear Results", command=clear_results)
clear_button.grid(row=3, column=2, padx=10, pady=10, sticky="w")

result_frames = {}

root.mainloop()