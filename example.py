from flask import Flask, request, render_template_string
import pandas as pd
import re
import math
import nltk
from nltk.corpus import stopwords
from autocorrect import Speller
import json
import os
import tarfile
import textwrap
from contextlib import closing
from urllib.request import urlretrieve

# Download the stopwords corpus if not already downloaded
nltk.download('stopwords')

app = Flask(__name__)

# Function to calculate the Outlier Score (OS) and Inverse Document Frequency (IDF)
def calculate_OS_IDF(collection):
    # Initialize dictionaries to store document frequencies (DF) and inverse document frequencies (IDF)
    document_frequencies = {}
    inverse_document_frequencies = {}

    # The documents tokenize into terms.
    tokenized_documents = [doc.split() for doc in collection]

    # Calculate the document frequency (DF) for each term
    for doc in tokenized_documents:
        unique_terms = set(doc)
        for term in unique_terms:
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    # Total number of documents (each row counts as one document)
    total_documents = len(collection)

    # Calculate IDF score for each term
    for term, df in document_frequencies.items():
        inverse_document_frequencies[term] = round(math.log(total_documents / (1 + df)), 2)

    # Calculate average IDF per document (each row counts as one document)
    average_term_idf_per_document = []
    max_idf_scores = []
    rarest_terms = []

    for doc in tokenized_documents:
        doc_idf_values = [inverse_document_frequencies.get(term, 0) for term in doc]
        if doc_idf_values:
            avg_idf = round(sum(doc_idf_values) / len(doc), 2)
            sorted_terms = sorted(set(doc), key=lambda term: -inverse_document_frequencies.get(term, 0))
            rarest_term = ', '.join(sorted_terms[:10])
            max_idf = [inverse_document_frequencies.get(term, 0) for term in sorted_terms[:10]]
        else:
            avg_idf = 0
            max_idf = [0] * 10
            rarest_term = ""

        average_term_idf_per_document.append(avg_idf)
        max_idf_scores.append(max_idf)
        rarest_terms.append(rarest_term)

    return average_term_idf_per_document, max_idf_scores, rarest_terms

# Function to preprocess each document
def preprocess_document(doc):
    doc = re.sub(r'\[\*\*.*?\*\*\]', ' ', doc)
    doc = re.sub(r'\b\d+\b', ' ', doc)
    doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc)
    doc = doc.lower()

    stop_words = set(stopwords.words('english'))
    tokens = doc.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    doc = ' '.join(filtered_tokens)

    return doc.strip()

# Function to correct spelling using autocorrect library
def autocorrect_spelling(doc):

    spell = Speller(fast=True)
    return spell(doc)


@app.route('/upload', methods=['POST'])
def upload_csv():
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

        df = df.apply(lambda x: x.map(preprocess_document))

        preprocessed_documents = df.values.flatten()

        if enable_automatic_correction:
            preprocessed_documents = [autocorrect_spelling(doc) for doc in preprocessed_documents]

        average_idf_scores, max_idf_scores, rarest_terms = calculate_OS_IDF(preprocessed_documents)

        output_df = pd.DataFrame({
            'Index': range(1, len(preprocessed_documents) + 1),
            'Original Text': preprocessed_documents,
            'Rarity Score': average_idf_scores,
            'Rarest Terms': rarest_terms,
            'Term Rarity Score': max_idf_scores
        })

        # Count same words as 1 word from Rarest Term column and Term Rarity Score Column
        for i, row in output_df.iterrows():
            terms = row['Rarest Terms'].split(', ')
            term_scores = row['Term Rarity Score']
            unique_terms = sorted(set(terms), key=lambda term: -term_scores[terms.index(term)])
            output_df.at[i, 'Rarest Terms'] = ', '.join(unique_terms[:10])

        output_html = output_df.to_html(index=False, classes="table table-striped table-hover", escape=False)

        output_html = output_html.replace(
            '<th>Rarity Score</th>',
            '<th style="width: 150px;" >'
            + '<button class="btn btn-secondary dropdown-toggle" type="button" id="rarityDropdown" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'
            + '<b data-toggle="tooltip" title="Click to sort rarity scores">Rarity Score</b> <i class="fas fa-filter" style="color: white;"></i>'
            + '</button>'
            + '<div class="dropdown-menu" aria-labelledby="rarityDropdown">'
            + '<a class="dropdown-item" href="#" onclick="sortTable(\'rarity\', \'default\')">Default</a>'
            + '<a class="dropdown-item" href="#" onclick="sortTable(\'rarity\', \'ascending\')">Ascending</a>'
            + '<a class="dropdown-item" href="#" onclick="sortTable(\'rarity\', \'descending\')">Descending</a>'
            + '</div></th>'
        )

        output_html = output_html.replace(
            '<th>Rarest Terms</th>',
            '<th id="rarestTermsHeader" >'
            + '<button class="btn btn-secondary dropdown-toggle" type="button" id="termsDropdown" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'
            + '<b id="rarestTermsHeaderText" data-toggle="tooltip" title="Click to filter rarest terms">10 Rarest Terms</b> <i class="fas fa-filter" style="color: white;"></i>'
            + '</button>'
            + '<div class="dropdown-menu" aria-labelledby="termsDropdown">'
            + "".join([f'<a class="dropdown-item" href="#" onclick="showTopTerms(event, {i})">{i}</a>' for i in
                       range(1, 11)]) +
            '</div></th>'
        )

        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>CSV Analyzer</title>
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
                <style>
                    .container {
                        margin-top: 50px;
                    }
                    .highlight {
                        background-color: rgba(255, 0, 0, 0.3); /* Red color with alpha transparency */
                    }
                    .table th {
                        background-color: #66cc00;
                        color: white;
                    }
                    .dropdown-menu {
                        background-color: #f8f9fa; /* Light gray background */
                    }
                    .dropdown-item {
                        color: #333; /* Dark gray text */
                    }
                    .navbar {
                        background-color: #66cc00; /* Navbar background color */
                    }
                    .navbar-brand {
                        color: white !important; /* Navbar brand text color */
                    }
                    .navbar-nav .nav-link {
                        color: white !important; /* Navbar link text color */
                    }
                    .btn-secondary.dropdown-toggle {
                        background-color: #66cc00 !important; /* Dropdown toggle button background color */
                        border-color: #66cc00 !important; /* Dropdown toggle button border color */
                    }
                    .dropdown-menu a.dropdown-item {
                        color: #333 !important; /* Dropdown item text color */
                    }
                    .dropdown-menu a.dropdown-item:hover {
                        background-color: #f0f0f0 !important; /* Hover background color for dropdown items */
                    }
                    .overlay {
                        position: fixed;
                        display: none;
                        width: 100%;
                        height: 100%;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background-color: rgba(0,0,0,0.5);
                        z-index: 2;
                        cursor: pointer;
                    }
                    .overlay-content {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        text-align: center;
                        color: white;
                    }
                    .progress-overlay {
                        position: fixed;
                        display: none;
                        width: 100%;
                        height: 100%;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background-color: rgba(0,0,0,0.7);
                        z-index: 3;
                    }
                    .progress-container {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        text-align: center;
                        color: white;
                        font-size: 20px;
                    }
                </style>
            </head>
            <body>
                <nav class="navbar navbar-expand-lg">
                    <a class="navbar-brand" href="/">CSV Analyzer</a>
                    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                </nav>
                <div class="container">
                    <h1 class="text-center mb-4">Outliers Analysis</h1>
                    <div class="progress-overlay" id="progressOverlay">
                        <div class="progress-container">
                            <p id="progressMessage">Generating Outlier Analysis</p>
                            <div class="progress">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" id="progressBar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                        </div>
                    </div>
                    <form method="post" action="/upload">
                        <div class="overlay" id="overlay2">
                            <div class="overlay-content">
                                <p>You have the option to select the rarest terms and sort the rarity score by clicking on their icons.</p>
                            </div>
                        </div>
                        {{ table | safe }}
                    </form>
                </div>
                <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
                <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
                <script>
                    $(document).ready(function(){
                        // Initialize tooltips
                        $('[data-toggle="tooltip"]').tooltip();

                        // Show progress overlay
                        $("#progressOverlay").fadeIn(1000, function() {
                            var progress = 0;
                            var interval = setInterval(function() {
                                var progressInt = parseInt(progress);
                                $("#progressBar").css("width", progressInt + "%").attr("aria-valuenow", progressInt);
                                $("#progressMessage").text("Generating Outlier Analysis - " + progressInt + "%");
                                progress += Math.random() * 10;
                                if (progress >= 100) {
                                    clearInterval(interval);
                                    $("#progressOverlay").fadeOut(1000);
                                    // Show overlay2 after progress overlay finishes
                                    $("#overlay2").fadeIn(1000, function() {
                                        // Hide overlay2 after 5 seconds
                                        setTimeout(function() {
                                            $("#overlay2").fadeOut(1000);
                                        }, 5000);
                                    });
                                }
                            }, 300);
                        });

                        // Highlight top 10 terms by default
                        showTopTerms(null, 10);
                    });

                    function sortTable(type, order) {
                        var rows = $("tbody > tr");
                        var index = type === "score" ? 4 : 3;
                        var data = [];

                        rows.each(function(){
                            var row = $(this);
                            var value = parseFloat(row.find("td:nth-child(" + index + ")").text());
                            data.push({index: row.index(), value: value});
                        });

                        if (order === "ascending") {
                            data.sort(function(a, b) {
                                return a.value - b.value;
                            });
                        } else if (order === "descending") {
                            data.sort(function(a, b) {
                                return b.value - a.value;
                            });
                        }

                        data.forEach(function(item) {
                            var row = rows.eq(item.index);
                            row.appendTo(row.parent());
                        });
                    }

                    function showTopTerms(event, count) {
                        if (event) event.preventDefault();
                        var rows = $("tbody > tr");
                        rows.each(function(){
                            var rarest_terms = $(this).find("td:nth-child(4)").text().split(", ");
                            var term_rarity_scores = $(this).find("td:nth-child(5)").text().split(", ");
                            var filtered_rarest_terms = rarest_terms.slice(0, count);
                            var filtered_term_rarity_scores = term_rarity_scores.slice(0, count);
                            $(this).find("td:nth-child(4)").text(filtered_rarest_terms.join(", "));
                            $(this).find("td:nth-child(5)").text(filtered_term_rarity_scores.join(", "));

                            // Highlight terms in Original Text column
                            var original_text = $(this).find("td:nth-child(2)");
                            var original_text_words = original_text.text().split(" ");
                            original_text.html(original_text_words.map(function(word) {
                                return filtered_rarest_terms.includes(word) ? "<span class='highlight'>" + word + "</span>" : word;
                            }).join(" "));
                        });

                        // Update the header text based on the selected count
                        $("#rarestTermsHeaderText").text(count + " Rarest Terms");
                    }
                </script>
            </body>
            </html>
        """, table=output_html)

    except Exception as e:
        return render_template_string(f"<h2>An error occurred: {str(e)}</h2>")

@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSV Analyzer</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .container {
            margin-top: 50px;
        }
        .card-header {
            background-color: #66cc00;
            color: white;
        }
        .card-body {
            background-color: #f8f9fa; /* Light gray background */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Upload a CSV File</h1>
        <div class="card">
            <div class="card-header">
                <h4 class="card-title">Upload CSV</h4>
            </div>
            <div class="card-body">
                <form method="post" action="/upload" enctype="multipart/form-data">
                    <div class="form-group">
                        <input type="file" name="file" class="form-control-file">
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" name="enable_automatic_correction" value="1" id="enableCorrectionCheckbox">
                        <label class="form-check-label" for="enableCorrectionCheckbox">
                        Enable Automatic Spelling Correction
                        </label>

                    </div>
                    <!-- Added tooltip to the upload button -->
                    <button type="submit" class="btn btn-primary" data-toggle="tooltip"  data-placement="right" title="Upload only one column file!">Upload</button>
                </form>
            </div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <script>
        // Activate Bootstrap tooltips
        $(document).ready(function(){
            $('[data-toggle="tooltip"]').tooltip();
        });
    </script>
</body>
</html>

    """)

if __name__ == '__main__':
    app.run(debug=True)
