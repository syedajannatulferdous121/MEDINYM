from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import re
import math
import spacy

app = Flask(__name__)

# Global variable to store the original order of the rows
original_order = []

# Load English language model and stop words
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words


# Function to remove stop words
def remove_stop_words(text):
    words = [word for word in text.split() if word.lower() not in sw_spacy and len(word) > 1]  # Remove single alphabet
    new_text = " ".join(words)
    return new_text


# Function to remove de-identified PHI data
def remove_deidentified_phis(text):
    pattern = r'\[\*\*.*?\*\*\]'
    cleaned_text = re.sub(pattern, ' ', text)
    return cleaned_text


# Function to remove numerical values from text
def remove_numerical(doc):
    return re.sub(r'\d+', ' ', doc)


# Function to delete punctuations
def delete_punctuations(text):
    return re.sub(r'[^\w\s]', ' ', text).replace("_", " ")


# Function to deal with multiple spaces
def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text)


# Function to count word frequencies in all rows and calculate rarity score
def count_word_frequencies(df):
    word_counts = {}
    for index, row in df.iterrows():
        text = ' '.join(map(str, row.values))  # Combine all values in the row into a single string

        # Preprocess the text
        text = remove_deidentified_phis(text)
        text = remove_numerical(text)
        text = delete_punctuations(text)
        text = remove_multiple_spaces(text)
        text = remove_stop_words(text)  # Remove single alphabet

        # Calculate rarity score (using word count as rarity score)
        score = len(text.split())
        word_counts[text] = score
    return word_counts


# Function to count word frequencies in the entire CSV file
def count_word_frequencies_overall(df):
    word_counts = {}
    for column in df.columns:
        for value in df[column]:
            words = re.findall(r'\b\w+\b', str(value).lower())  # Extract words from each cell
            for word in words:
                if word != '/':  # Exclude '/' symbol
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
    return {word: count for word, count in word_counts.items() if count > 1}  # Return only repeated words


# Function to calculate IDF score
def calculate_idf(word, documents):
    # Calculate the IDF (Inverse Document Frequency) score for a word.
    # IDF = log(total number of documents / number of documents containing the word)

    num_documents_containing_word = sum(1 for document in documents if word in document)

    total_documents = len(documents)
    if num_documents_containing_word == 0:
        return 0
    return math.log(total_documents / num_documents_containing_word)


# Function to calculate OSIDF score
def calculate_osidf_score(df):
    # OSIDF = sum of IDF scores / number of words in a row
    total_documents = len(df)
    word_counts = count_word_frequencies_overall(df)
    osidf_scores = {}

    # Calculate IDF scores for each word
    idf_scores = {term: calculate_idf(term, df.values.flatten()) for term in word_counts}

    # Calculate OSIDF scores
    for text, count in word_counts.items():
        num_words = len(text.split())
        osidf_scores[text] = sum(idf_scores[word] for word in text.split()) / num_words if num_words > 0 else 0

    return osidf_scores


# Function to calculate rarity score of the text using OSIDF score
def calculate_rarity_score_text(text, osidf_scores):
    words = re.findall(r'\b\w+\b', str(text).lower())
    total_score = 0
    for word in words:
        if word in osidf_scores:
            total_score += osidf_scores[word]
    return total_score


# Function to count words in a document and calculate rarity score
def count_words_and_rarity_score(text, osidf_scores):
    words = re.findall(r'\b\w+\b', str(text).lower())
    total_score = 0
    for word in words:
        if word in osidf_scores:
            total_score += osidf_scores[word]
    return len(words), total_score


# Upload route to handle file uploads and return rare terms
@app.route('/upload', methods=['POST'])
def upload_csv():
    try:
        # Check if a file was provided
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'})

        # Check if the file is a CSV file
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'})

        # Read the CSV file
        df = pd.read_csv(file)

        # Store the original order of the rows
        global original_order
        original_order = df.index.tolist()

        # Count word frequencies and calculate rarity score for each row
        word_counts = count_word_frequencies(df)

        # Get the unique sentences
        unique_sentences = list(word_counts.keys())

        # Calculate OSIDF scores
        osidf_scores = calculate_osidf_score(df)

        # Calculate rarity score for each text
        rarity_scores_text = [calculate_rarity_score_text(sentence, osidf_scores) for sentence in unique_sentences]

        # Convert unique sentences to list of dictionaries for JSON response
        rare_terms = [
            {'Index': i + 1, 'Original Text': sentence, 'Rarity Score': rarity_scores_text[i], 'Rarest Terms': None,
             'Term Rarity Score': None} for
            i, sentence in enumerate(unique_sentences)]

        # Count word frequencies in the entire CSV file
        repeated_words_overall = count_word_frequencies_overall(df)

        return jsonify({'rare_terms': rare_terms, 'repeated_words_overall': repeated_words_overall})
    except Exception as e:
        return jsonify({'error': str(e)})


# Index route to render the index.html template
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Outliers Analysis</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #fff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .upload-form {
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .upload-input {
            flex-grow: 1;
            margin-right: 10px;
        }
        .upload-btn {
            background-color: rgb(60, 179, 113);
            border-color: rgb(60, 179, 113);
            color: #fff;
        }
        .upload-btn:hover {
            background-color: #218838;
            border-color: #1e7e34;
        }
        .progress {
            margin-bottom: 20px;
        }
        .progress-bar {
            transition: width 0.3s ease-in-out;
        }
        .error-message {
            color: #dc3545;
            margin-top: 10px;
        }
        .table-container {
            margin-top: 20px;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            border-spacing: 0;
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .table th, .table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .table th {
            background-color: rgb(60, 179, 113);
            color: #ffffff;
            font-weight: bold;
            text-transform: uppercase;
            position: relative;
        }
        .table th:hover .rarest-terms-dropdown {
            display: block;
        }
        .rarest-terms-dropdown {
            display: none;
            position: absolute;
            background-color: #f9f9f9;
            min-width: 120px;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1;
        }
        .rarest-terms-dropdown a {
            color: black;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
        }
        .rarest-terms-dropdown a:hover {
            background-color: #f1f1f1;
        }
        .table tbody tr:hover {
            background-color: #f2f2f2;
        }
        .refresh-btn {
            background-color: rgb(60, 179, 113);
            color: #fff;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            font-size: 18px;
            position: fixed;
            bottom: 20px;
            right: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .refresh-btn:hover {
            background-color: #218838;
        }
        .highlight {
            background-color: #ff0000;
            color: #ffffff;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .title {
            font-family: Arial, sans-serif;
            text-align: center;
            font-size: 36px;
            margin-bottom: 20px;
            animation: bounce 2s infinite;
        }
        @keyframes bounce {
            0% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-20px);
            }
            100% {
                transform: translateY(0);
            }
        }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title">Outliers Analysis</h1>
            <form id="uploadForm" class="upload-form" enctype="multipart/form-data">
                <input type="file" class="form-control upload-input" id="fileInput">
                <button class="btn btn-primary upload-btn" type="button" onclick="uploadFile()">Upload</button>
            </form>
            <div class="progress" style="display: none;">
                <div class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
            <div id="uploadResult" class="upload-result"></div>
            <div id="errorMessage" class="error-message"></div>
            <div id="filterOptions" style="display:none;">
                <!-- Removed Filter Original Text section -->
            </div>
            <div class="table-container" id="tableContainer" style="display:none;">
                <table class="table">
                    <thead id="tableHeader" style="display:none;">
                        <tr>
                            <th scope="col">Index</th>
                            <th scope="col">Original Text</th>
                            <th scope="col">Rarity Score</th>
                            <th scope="col">
                                Rarest Terms
                                <div class="rarest-terms-dropdown">
                                    <a href="#" onclick="showRarestTerms(1)">1 Rarest Term</a>
                                    <a href="#" onclick="showRarestTerms(2)">2 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(3)">3 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(4)">4 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(5)">5 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(6)">6 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(7)">7 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(8)">8 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(9)">9 Rarest Terms</a>
                                    <a href="#" onclick="showRarestTerms(10)">10 Rarest Terms</a>
                                </div>
                            </th>
                            <th scope="col">Term Rarity Score</th>
                        </tr>
                    </thead>
                    <tbody id="rareTermsTable"></tbody>
                </table>
            </div>
        </div>
        <button class="refresh-btn" onclick="refreshPage()"><i class="fas fa-sync-alt"></i></button> <!-- Refresh Icon -->

        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script>
            var originalOrder = [];
            var currentOrder = [];

            // Function to upload file
            function uploadFile() {
                var fileInput = document.getElementById('fileInput');
                var file = fileInput.files[0];
                var formData = new FormData();
                formData.append('file', file);

                // Show progress bar
                var progressDiv = $('.progress');
                var progressBar = $('.progress-bar');
                progressDiv.show();

                $.ajax({
                    type: 'POST',
                    url: '/upload',
                    data: formData,
                    contentType: false,
                    processData: false,
                    xhr: function() {
                        var xhr = new window.XMLHttpRequest();
                        xhr.upload.addEventListener('progress', function(evt) {
                            if (evt.lengthComputable) {
                                var percentComplete = (evt.loaded / evt.total) * 100;
                                progressBar.width(percentComplete + '%');
                                progressBar.attr('aria-valuenow', percentComplete);
                            }
                        }, false);
                        return xhr;
                    },
                    success: function(response) {
                        progressDiv.hide();
                        progressBar.width('0%');
                        $('#errorMessage').html('');
                        displayRareTerms(response.rare_terms);
                        $('#filterOptions').show(); // Show filter options after displaying rare terms
                        $('#tableContainer').show(); // Show table container
                        $('#tableHeader').show(); // Show table header
                    },
                    error: function(xhr, status, error) {
                        console.error('Error uploading file:', error);
                        progressDiv.hide();
                        progressBar.width('0%');
                        $('#uploadResult').html('');
                        $('#errorMessage').html('<div class="alert alert-danger" role="alert">Error uploading file. Please try again.</div>');
                    }
                });
            }

            // Function to display rare terms
            function displayRareTerms(rareTerms) {
                var rareTermsTable = '';
                rareTerms.forEach(function(term) {
                    rareTermsTable += '<tr>';
                    rareTermsTable += '<td>' + term.Index + '</td>';
                    rareTermsTable += '<td>' + term['Original Text'] + '</td>';
                    rareTermsTable += '<td>' + term['Rarity Score'] + '</td>';
                    rareTermsTable += '<td><span id="rarestTerms_' + term.Index + '"></span></td>';
                    rareTermsTable += '<td><span id="termRarityScore_' + term.Index + '"></span></td>';
                    rareTermsTable += '</tr>';
                });
                $('#rareTermsTable').html(rareTermsTable);
            }

            // Function to show rarest terms
            function showRarestTerms(numTerms) {
                $('.table tbody tr').each(function() {
                    var index = $(this).find('td:eq(0)').text();
                    var term = $(this).find('td:eq(1)').text();
                    var rarityScore = parseFloat($(this).find('td:eq(2)').text());
                    var words = term.split(' ');
                    var osidfScores = {};
                    for (var i = 0; i < words.length; i++) {
                        if (words[i] != '/' && i > 0) { // Exclude first word from highlighting
                            if (words[i] in osidfScores) {
                                osidfScores[words[i]] += rarityScore * (i + 1);
                            } else {
                                osidfScores[words[i]] = rarityScore * (i + 1);
                            }
                        }
                    }
                    var sortedWords = Object.keys(osidfScores).sort(function(a, b) {
                        return osidfScores[b] - osidfScores[a];
                    });
                    var rarestTerms = sortedWords.slice(0, numTerms).join(', ');

                    // Highlight rarest terms in Original Text column
                    var highlightedTerm = highlightTermInText(term, rarestTerms);
                    $(this).find('td:eq(1)').html(highlightedTerm);

                    $('#rarestTerms_' + index).text(rarestTerms);

                    // Calculate Term Rarity Score
                    var termRarityScores = [];
                    for (var i = 0; i < numTerms; i++) {
                        termRarityScores.push((osidfScores[sortedWords[i]]).toFixed(2));
                    }
                    $('#termRarityScore_' + index).text(termRarityScores.join(', '));
                });
            }

            // Function to highlight rarest terms in Original Text column
            function highlightTermInText(text, termsToHighlight) {
                var words = text.split(' ');
                for (var i = 0; i < words.length; i++) {
                    if (termsToHighlight.indexOf(words[i]) !== -1) {
                        words[i] = '<span class="highlight">' + words[i] + '</span>';
                    }
                }
                return words.join(' ');
            }

            // Function to refresh page
            function refreshPage() {
                location.reload();
            }
        </script>
    </body>
    </html>
    """)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
