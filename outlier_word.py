from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import re
import math
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from autocorrect import Speller

# Download the necessary NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

app = Flask(__name__)

# Global variables to store preprocessed documents and sorted terms
preprocessed_documents = []
sorted_terms = []

# Function to map Treebank POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    tag_map = {
        'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,  # Adjectives
        'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV,  # Adverbs
        'VB': wordnet.VERB, 'VBD': wordnet.VERB, 'VBG': wordnet.VERB, 'VBN': wordnet.VERB, 'VBP': wordnet.VERB, 'VBZ': wordnet.VERB,  # Verbs
        'NN': wordnet.NOUN, 'NNS': wordnet.NOUN, 'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN  # Nouns
    }
    return tag_map.get(treebank_tag, None)

# Function to calculate the Outlier Score (OS) and Inverse Document Frequency (IDF)
def calculate_OS_IDF(collection):
    document_frequencies = {}
    inverse_document_frequencies = {}

    # Calculate the document frequency (DF) for each term
    for doc in collection:
        tokens = word_tokenize(doc)
        pos_tags = pos_tag(tokens)
        filtered_tokens = set(token for token, tag in pos_tags if get_wordnet_pos(tag) == wordnet.NOUN)
        for token in filtered_tokens:
            document_frequencies[token] = document_frequencies.get(token, 0) + 1

    total_documents = len(collection)

    # Calculate IDF score for each term
    for term, df in document_frequencies.items():
        inverse_document_frequencies[term] = round(math.log(total_documents / (1 + df)), 2)

    sorted_terms = sorted(inverse_document_frequencies.items(), key=lambda x: x[1], reverse=True)[:500]

    return sorted_terms

# Function to preprocess each document
def preprocess_document(doc):
    doc = re.sub(r'\[\*\*.*?\*\*\]', ' ', doc)
    doc = re.sub(r'\b\d+\b', ' ', doc)
    doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc)
    doc = doc.lower()

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(doc)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    doc = ' '.join(filtered_tokens)

    return doc.strip()

# Function to correct spelling using autocorrect library
def autocorrect_spelling(doc):
    spell = Speller(fast=True)
    corrected_doc = ' '.join([spell(word) for word in doc.split()])
    return corrected_doc

@app.route('/upload', methods=['POST'])
def upload_csv():
    global preprocessed_documents, sorted_terms
    try:
        file = request.files['file']
        if not file:
            return render_template_string("<h2>No file provided</h2>")

        if not file.filename.endswith('.csv'):
            return render_template_string("<h2>Please upload a CSV file</h2>")

        df = pd.read_csv(file)

        if len(df.columns) != 1:
            return render_template_string("<h2>Please provide only a one-column dataset</h2>")

        enable_automatic_correction = request.form.get('enable_automatic_correction') == '1'

        df['preprocessed'] = df.iloc[:, 0].apply(preprocess_document)
        preprocessed_documents = df['preprocessed'].tolist()

        if enable_automatic_correction:
            preprocessed_documents = [autocorrect_spelling(doc) for doc in preprocessed_documents]

        sorted_terms = calculate_OS_IDF(preprocessed_documents)

        output_html = generate_table_html(50)

        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Find Outlier Word</title>
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
                <style>
                    .highlight { background-color: rgba(255, 0, 0, 0.3); }
                </style>
                <script>
                    $(document).ready(function() {
                        $("#num_terms").on("input", function() {
                            var num_terms = $(this).val();
                            $("#num_terms_label").text(num_terms);
                            updateTable(num_terms);
                        });
                    });

                    function updateTable(num_terms) {
                        $.ajax({
                            url: "/update_table",
                            method: "POST",
                            data: { num_terms: num_terms },
                            success: function(data) {
                                $("#output_table").html(data);
                            }
                        });
                    }
                </script>
            </head>
            <body>
                <div class="container">
                    <h1 class="text-center mb-4">Find Outlier Word</h1>
                    <div class="form-group">
                        <label for="num_terms">Number of Terms: <span id="num_terms_label">50</span></label>
                        <input type="range" class="form-control-range" id="num_terms" name="num_terms" min="1" max="500" value="50">
                    </div>
                    <div id="output_table">
                        {{ table | safe }}
                    </div>
                </div>
            </body>
            </html>
        """, table=output_html)

    except Exception as e:
        return render_template_string(f"<h2>An error occurred: {str(e)}</h2>")

@app.route('/update_table', methods=['POST'])
def update_table():
    num_terms = int(request.form['num_terms'])
    output_html = generate_table_html(num_terms)
    return output_html

def generate_table_html(num_terms):
    output_data = {
        'Original document index': [],
        'Original Text': [],
        'Term': [],
        'Term Rarity score': []
    }
    for i in range(min(num_terms, len(sorted_terms))):
        term, rarity_score = sorted_terms[i]
        term_indices = [index for index, doc in enumerate(preprocessed_documents) if re.search(r'\b{}\b'.format(re.escape(term)), doc)]
        if term_indices:
            term_index = term_indices[0]
            original_text = preprocessed_documents[term_index]
            highlighted_text = re.sub(r'(\b{}\b)'.format(re.escape(term)), r'<span class="highlight">\1</span>', original_text)
            output_data['Original document index'].append(term_index + 1)
            output_data['Original Text'].append(highlighted_text)
            output_data['Term'].append(term)
            output_data['Term Rarity score'].append(rarity_score)

    output_df = pd.DataFrame(output_data)
    output_df.sort_values(by='Original document index', ascending=True, inplace=True)
    output_html = output_df.to_html(index=False, classes="table table-striped table-hover", escape=False)

    # Add the 'table-responsive' class
    output_html = f'<div class="table-responsive">{output_html}</div>'

    return output_html

@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Find Outlier Word</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center mb-4">Upload Your CSV File</h1>
                <div class="row justify-content-center">
                    <div class="col-md-6">
                        <form method="post" action="/upload" enctype="multipart/form-data">
                            <div class="form-group">
                                <label for="file">Choose a CSV file:</label>
                                <input type="file" class="form-control-file" id="file" name="file" accept=".csv" required>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="enable_automatic_correction" value="1" id="enableCorrectionCheckbox">
                                <label class="form-check-label" for="enableCorrectionCheckbox">
                                Enable Automatic Spelling Correction
                                </label>
                            </div>
                            <div class="form-group">
                                <button type="submit" class="btn btn-primary btn-block">Upload</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </body>
        </html>
    """)

if __name__ == '__main__':
    app.run(port=8088, debug=True)
