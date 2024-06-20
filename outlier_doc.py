from flask import Flask, request, render_template_string
import pandas as pd
import re
import math
import nltk
from nltk.corpus import stopwords
from autocorrect import Speller
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
from matplotlib.figure import Figure
import numpy as np
import mpld3
from mpld3 import plugins
import spacy

# Load the spaCy language model
nlp = spacy.load('en_core_web_sm')

# Download the stopwords corpus if not already downloaded
nltk.download('stopwords')

app = Flask(__name__)


# Function to check if a token is a noun or a proper noun
def is_noun_or_proper_noun(token):
    return token.pos_ in {'NOUN', 'PROPN'}


# Function to calculate the Outlier Score (OS) and Inverse Document Frequency (IDF)
def calculate_OS_IDF(collection):
    # Initialize dictionaries to store document frequencies (DF) and inverse document frequencies (IDF)
    document_frequencies = {}
    inverse_document_frequencies = {}

    # Tokenize documents into terms using spaCy
    tokenized_documents = [nlp(doc) for doc in collection]

    # Extract all noun and proper noun terms from each document
    noun_terms_documents = []
    for doc in tokenized_documents:
        noun_terms = [token.text for token in doc if is_noun_or_proper_noun(token)]
        noun_terms_documents.append(noun_terms)
        unique_terms = set(noun_terms)
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

    for doc in noun_terms_documents:
        doc_idf_values = [inverse_document_frequencies.get(term, 0) for term in doc]
        if doc_idf_values:
            avg_idf = round(sum(doc_idf_values) / len(doc), 2)
            sorted_terms = sorted(set(doc), key=lambda term: -inverse_document_frequencies.get(term, 0))
            rarest_term = ', '.join(sorted_terms[:10])
            max_idf = [f"{inverse_document_frequencies.get(term, 0):.2f}" for term in sorted_terms[:10]]
        else:
            avg_idf = 0
            max_idf = [f"{0:.2f}" for _ in range(10)]
            rarest_term = ""

        average_term_idf_per_document.append(f"{avg_idf:.2f}")
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


@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Outlier Analyzer</title>
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
        <h1 class="text-center mb-4">Outlier Document</h1>
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

        // Form submission handler
        $('#uploadForm').on('submit', function(event) {
            const fileInput = $('input[name="file"]');
            const file = fileInput[0].files[0];
            if (!file) {
                event.preventDefault();
                $('#alertMessage').text('No file provided').show();
            } else if (!file.name.endsWith('.csv')) {
                event.preventDefault();
                $('#alertMessage').text('Please upload a CSV file').show();
            }
        });

    </script>
</body>
</html>

    """)


@app.route('/upload', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        if not file:
            return render_template_string("""
                        <script>
                            window.onload = function() {
                                alert('No file provided');
                                window.history.back();
                            }
                        </script>
                    """)

        if not file.filename.endswith('.csv'):
            return render_template_string("""
                       <script>
                           window.onload = function() {
                               alert('Please upload a CSV file');
                               window.history.back();
                           }
                       </script>
                   """)

        # Read the CSV file
        df = pd.read_csv(file)

        # Check for single column
        if len(df.columns) != 1:
            return render_template_string("<h2>Please provide only a one-column dataset</h2>")

        # Store original text before preprocessing
        original_documents = df.iloc[:, 0].tolist()

        enable_automatic_correction = request.form.get('enable_automatic_correction') == '1'


        # Apply preprocessing and optional autocorrection
        preprocessed_documents = df.iloc[:, 0].apply(preprocess_document).tolist()

        if enable_automatic_correction:
            preprocessed_documents = [autocorrect_spelling(doc) for doc in preprocessed_documents]

        average_idf_scores, max_idf_scores, rarest_terms = calculate_OS_IDF(preprocessed_documents)

        output_df = pd.DataFrame({
            'Index': range(1, len(preprocessed_documents) + 1),
            'Original Text': original_documents,
            'Preprocessed Text': preprocessed_documents,
            'Rarity Score': average_idf_scores,
            'Rarest Terms': rarest_terms,
            'Term Rarity Score': max_idf_scores

        })

        # Calculate the statistics for the histogram
        rarity_scores = list(map(float, average_idf_scores))
        mean_score = np.mean(rarity_scores)
        median_score = np.median(rarity_scores)
        std_dev_score = np.std(rarity_scores)

        # Generate the histogram
        fig = Figure()
        ax = fig.subplots()
        counts, bins, patches = ax.hist(rarity_scores, bins='auto', color='green', alpha=0.7, edgecolor='black')

        # Plot mean, median, and standard deviation lines
        ax.axvline(mean_score, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean_score:.2f}')
        ax.axhline(y=max(counts) / 2, color='blue', linestyle='-',
                   linewidth=2)  # Horizontal line at half of the max frequency
        ax.axvline(median_score, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
        ax.axvline(mean_score + std_dev_score, color='yellow', linestyle='--', linewidth=2,
                   label=f'Standard Deviation: {std_dev_score:.2f}')
        ax.axvline(mean_score - std_dev_score, color='yellow', linestyle='--', linewidth=2)

        # Set axis limits to ensure the mean lines are visible
        ax.set_xlim([min(rarity_scores) - 1, max(rarity_scores) + 1])
        ax.set_ylim([0, max(counts) + 5])

        ax.set_title('Rarity Score Frequencies')
        ax.set_xlabel('Outlier Score')
        ax.set_ylabel('Frequency')
        ax.legend()

        # Convert the plot to a PNG image and then to a base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        histogram_png = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Add histogram image to the HTML
        histogram_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{histogram_png}" alt="Rarity Score Frequencies Histogram"></div>'

        def generate_color_gradient(scores, max_alpha=0.8, alpha_gap=0.05):
            max_score = max(scores)
            sorted_scores = sorted(scores, reverse=True)

            colors = []
            for score in scores:
                rank = sorted_scores.index(score)
                alpha = max_alpha - rank * alpha_gap
                if alpha < 0:
                    alpha = 0
                color = f'rgba(255, 0, 0, {alpha})'  # Red color with varying transparency
                colors.append(color)

            return colors

        def highlightTerms(text, terms, scores):
            term_scores_map = {term: score for term, score in zip(terms, scores)}
            sorted_terms = sorted(terms, key=lambda term: -term_scores_map[term])
            scores_sorted = [term_scores_map[term] for term in sorted_terms]
            colors = generate_color_gradient(scores_sorted)

            term_color_map = {term: colors[i] for i, term in enumerate(sorted_terms)}

            return ' '.join(
                [
                    f"<span style='background-color: {term_color_map[word]};'>{word}</span>" if word in term_color_map else word
                    for word in text.split()
                ]
            )

        # for i, row in output_df.iterrows():
        # terms = row['Rarest Terms'].split(', ')
        # term_scores = row['Term Rarity Score']
        # original_text = row['Original Text']
        # highlighted_text = apply_highlighting(original_text, terms, term_scores)
        # output_df.at[i, 'Original Text'] = highlighted_text

        output_html = output_df.to_html(index=False, classes="table table-striped table-hover table-responsive",
                                        escape=False)

        output_html = output_html.replace(
            '<th>Rarity Score</th>',
            '<th>'
            + '<div class="dropdown">'
            + '<button class="btn btn-secondary dropdown-toggle" type="button" id="rarityDropdown" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'
            + '<b data-toggle="tooltip" title="Click to sort rarity scores">Rarity <br> Score</b> <i class="fas fa-filter" style="color: white;"></i>'
            + '</button>'
            + '<div class="dropdown-menu" aria-labelledby="rarityDropdown">'
            + '<a class="dropdown-item" href="#" onclick="sortTable(\'rarity\', \'default\')">Default</a>'
            + '<a class="dropdown-item" href="#" onclick="sortTable(\'rarity\', \'ascending\')">Ascending</a>'
            + '<a class="dropdown-item" href="#" onclick="sortTable(\'rarity\', \'descending\')">Descending</a>'
            + '</div></div></th>'
        )

        output_html = output_html.replace(
            '<th>Rarest Terms</th>',
            '<th>'
            + '<div class="dropdown">'
            + '<button class="btn btn-secondary dropdown-toggle" type="button" id="termsDropdown" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'
            + '<b id="rarestTermsHeaderText" data-toggle="tooltip" title="Click to filter rarest terms">10 Rarest Terms</b> <i class="fas fa-filter" style="color: white;"></i>'
            + '</button>'
            + '<div class="dropdown-menu" aria-labelledby="termsDropdown">'
            + "".join(
                [f'<a class="dropdown-item" href="#" onclick="showTopTerms(event, {i}, \'default\')">{i}</a>' for i in
                 range(1, 11)]) +
            '</div></div></th>'
        )



        output_html = output_html.replace('<th>Original Text</th>', '<th>Original Text</th>')
        output_html = output_html.replace('<table', '<div class="table-responsive"><div class="container"><table')
        output_html = output_html.replace('</table>', '</table></div></div>')

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
                        background-color:  rgba(0,0,0,0.5);
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

                    {{ histogram_html|safe }}
                    {{ table_html|safe }}
                        </div>
                    </form>

                <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
                <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
                <script>
                    $(function () {
                        $('[data-toggle="tooltip"]').tooltip();
                    });

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
                                    // Hide overlay2 after 3 seconds
                                    setTimeout(function() {
                                        $("#overlay2").fadeOut(1000);
                                    }, 3000);
                                });
                            }
                        }, 300);
                    });



                    function showTopTerms(event, count, order) {
    const header = $('#rarestTermsHeaderText');
    header.text(count + ' Rarest Terms');
    const table = $('table').get(0);
    const rows = Array.from(table.rows).slice(1); // Skip the header row

    rows.forEach(row => {
        const termsCell = row.cells[4];
        const terms = termsCell.getAttribute('data-original-terms').split(', ');
        const termScoresCell = row.cells[5];
        const termScores = JSON.parse(termScoresCell.getAttribute('data-original-scores'));

        const sortedIndices = termScores.map((score, index) => index)
            .sort((a, b) => termScores[b] - termScores[a]);

        const topTerms = sortedIndices.slice(0, count).map(index => terms[index]);
        const topScores = sortedIndices.slice(0, count).map(index => parseFloat(termScores[index]).toFixed(2));

        termsCell.innerText = topTerms.join(', ');
        termScoresCell.innerText = topScores.join(', '); // Changed this line

        const originalTextCell = row.cells[2];
        const originalText = originalTextCell.getAttribute('data-original-text');
        const updatedText = highlightTerms(originalText, topTerms, topScores);
        originalTextCell.innerHTML = updatedText;
    });

    event.preventDefault();
}


function highlightTerms(text, terms, scores) {
    const termScoresMap = {};
    terms.forEach((term, index) => {
        termScoresMap[term] = scores[index];
    });
    const sortedTerms = terms.slice().sort((a, b) => termScoresMap[b] - termScoresMap[a]);
    const sortedScores = sortedTerms.map(term => termScoresMap[term]);
    const colors = generateColorGradient(sortedScores);

    const termColorMap = {};
    sortedTerms.forEach((term, index) => {
        termColorMap[term] = colors[index];
    });

    return text.split(' ').map(word => {
        return termColorMap[word] ? `<span style="background-color: ${termColorMap[word]};">${word}</span>` : word;
    }).join(' ');
}

function generateColorGradient(scores, maxAlpha = 0.8, alphaGap = 0.05) {
    const maxScore = Math.max(...scores);
    const sortedScores = [...scores].sort((a, b) => b - a);

    return scores.map(score => {
        const rank = sortedScores.indexOf(score);
        let alpha = maxAlpha - rank * alphaGap;
        if (alpha < 0) alpha = 0;
        return `rgba(255, 0, 0, ${alpha})`;
    });
}

// Ensure original terms and scores are stored in data attributes when the table is generated
$(document).ready(function() {
    $('table tbody tr').each(function() {
        const originalTextCell = $(this).find('td:eq(2)');
        const termsCell = $(this).find('td:eq(4)');
        const termScoresCell = $(this).find('td:eq(5)');



        termsCell.attr('data-original-terms', termsCell.text());
        const termScores = JSON.parse(termScoresCell.text()).map(score => parseFloat(score).toFixed(2));
        termScoresCell.attr('data-original-scores', JSON.stringify(termScores));
        originalTextCell.attr('data-original-text', originalTextCell.text());

        const terms = termsCell.text().split(', ');

        const originalText = originalTextCell.text();
        const highlightedText = highlightTerms(originalText, terms, termScores);
        originalTextCell.html(highlightedText);
    });
});

$(document).ready(function () {
        $('[data-toggle="tooltip"]').tooltip();

        // Ensure original rarity scores are stored in data attributes when the table is generated
        $('table tbody tr').each(function () {
            const rarityScoreCell = $(this).find('td:eq(3)');
            rarityScoreCell.attr('data-original-rarity', rarityScoreCell.text());
        });
    });

    function sortTable(column, order) {
        const table = $('table').get(0);
        const rows = Array.from(table.rows).slice(1); // Skip the header row
        const tbody = table.tBodies[0];

        if (column === 'rarity') {
            rows.sort((rowA, rowB) => {
                const cellA = parseFloat(rowA.cells[3].innerText);
                const cellB = parseFloat(rowB.cells[3].innerText);
                return order === 'ascending' ? cellA - cellB : cellB - cellA;
            });

            // Clear the table body
            while (tbody.firstChild) {
                tbody.removeChild(tbody.firstChild);
            }

            // Append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        } else if (order === 'default') {
            // Reset to default order
            rows.sort((rowA, rowB) => {
                const cellA = parseFloat($(rowA).find('td:eq(3)').attr('data-original-rarity'));
                const cellB = parseFloat($(rowB).find('td:eq(3)').attr('data-original-rarity'));
                return cellA - cellB;
            });

            // Clear the table body
            while (tbody.firstChild) {
                tbody.removeChild(tbody.firstChild);
            }

            // Append rows in the default order
            rows.forEach(row => tbody.appendChild(row));
        }
    }
                </script>
            </body>
            </html>
        """, table_html=output_html, histogram_html=histogram_html)

    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(port=8084, debug=True)
