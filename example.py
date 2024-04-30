from flask import Flask, request, render_template_string
import pandas as pd
import re
import math
import nltk
from nltk.corpus import stopwords

# Download the stopwords corpus if not already downloaded
nltk.download('stopwords')

app = Flask(__name__)


def calculate_OS_IDF(collection):
    # Initialize dictionaries to store document frequencies (DF) and inverse document frequencies (IDF)
    document_frequencies = {}  # Stores how many times each term appears in the entire collection of documents
    inverse_document_frequencies = {}  # Stores the IDF value for each term

    # For each document it will initialize the array to store IDF values.
    idfs_arr = []

    # The documents tokenize into terms.
    tokenized_documents = [row.split() for row in collection]

    # Calculate the document frequency (DF) for each term
    for doc in tokenized_documents:
        unique_terms = set(doc)  # Find unique terms in the document
        # Update the document frequencies dictionary:
        # If the term already exists, increment its count by 1; otherwise, initialize it with count 1
        for term in unique_terms:
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    # Total number of documents (each rows count as each document)
    total_documents = len(collection)

    # Calculate IDF score for each term
    for term, df in document_frequencies.items():
        # Calculate IDF using the formula: log(total_documents / (1 + term_frequency))
        inverse_document_frequencies[term] = math.log(total_documents / (1 + df))

    # Calculate average of IDF per document (each rows count as each document)
    average_term_idf_per_document = []
    for doc in tokenized_documents:
        # take the IDF values for terms in the document
        doc_idf_values = [inverse_document_frequencies[term] for term in doc if term in inverse_document_frequencies]
        if doc_idf_values:
            # Calculate the average IDF score per term
            avg_idf = sum(doc_idf_values) / len(doc)
        else:
            avg_idf = 0  # Set to 0 if there are no valid terms in the document

        average_term_idf_per_document.append(avg_idf)
        idfs_arr.append(doc_idf_values)  # Append the average IDF score for the document to the list
    # Create a DataFrame to display the output in a tabular format
    term_idfs_df = pd.DataFrame(
        {"Terms": tokenized_documents, "Term_IDFs": idfs_arr})  # DataFrame is created to organize the IDF scores for each term in each document. It has two columns: "Terms" (each document is represented as a list of terms) and "Term_IDFs" (IDF values corresponding to each term)
    return pd.Series(average_term_idf_per_document), term_idfs_df  # Storing the average IDF scores


def get_rare_terms(term_idfs):
    rare_terms = []
    for index, row in term_idfs.iterrows():
        terms = row['Terms']
        term_idfs_values = row['Term_IDFs']
        for term, idf_values in zip(terms, term_idfs_values):
            if idf_values == 0:
                rare_terms.append((term, idf_values))  # Store both term and IDF value
    return rare_terms


def preprocess_document(doc):
    """
    Apply preprocessing steps to the document.
    """
    # Remove de-identified PHI data
    doc = re.sub(r'\[\*\*.*?\*\*\]', ' ', doc)
    # Remove numerical values
    doc = re.sub(r'\b\d+\b', ' ', doc)
    # Remove all non-alphanumeric characters except spaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc)
    # Remove extra spaces
    doc = re.sub(r'\s+', ' ', doc)
    # Lowercase the text
    doc = doc.lower()

    # Remove stop words more effectively
    stop_words = set(stopwords.words('english'))
    tokens = doc.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    doc = ' '.join(filtered_tokens)

    return doc.strip()  # returns the processed document, Strip leading and trailing whitespaces


@app.route('/upload', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        if not file:
            return render_template_string("<h2>No file provided</h2>")

        if not file.filename.endswith('.csv'):
            return render_template_string("<h2>Please upload a CSV file</h2>")

        df = pd.read_csv(file)

        # follow preprocessing for each cell or element in the dataframe
        df = df.applymap(preprocess_document)

        # Flatten the dataframe to a list of documents
        preprocessed_documents = df.values.flatten()

        average_term_idf_per_document, term_idfs_df = calculate_OS_IDF(
            preprocessed_documents)  # This array contains all the preprocessed text data from the DataFrame

        # You can further process the term IDF DataFrame to get rare terms
        rare_terms = get_rare_terms(term_idfs_df)

        # Create a DataFrame to display the output in a tabular format
        output_df = pd.DataFrame({
            'Index': range(1, len(average_term_idf_per_document) + 1),
            'Original Text': preprocessed_documents,
            'Rarest Terms': df.values.flatten(),
            'Rarity Score': average_term_idf_per_document,
            'Term Rarity Score': term_idfs_df['Term_IDFs']
        })

        # Convert the DataFrame to HTML table
        output_html = output_df.to_html(index=False, classes="table table-striped table-hover")

        return render_template_string("""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>CSV Analyzer</title>
                        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                        <style>
                            .container {
                                margin-top: 50px;
                            }
                            table {
                                width: 100%;
                                border-collapse: collapse;
                            }
                            th, td {
                                border: 1px solid #dddddd;
                                text-align: left;
                                padding: 8px;
                            }
                            th {
                                background-color: #f2f2f2;
                            }
                            tr:nth-child(even) {
                                background-color: #f2f2f2;
                            }
                            .filter-container {
                                margin-bottom: 10px;
                            }
                            .table-filter {
                                display: flex;
                                align-items: center;
                            }
                            .table-filter select {
                                margin-left: 10px;
                            }
                        </style>
                        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                        <script>
                            $(document).ready(function(){
                                // Function to handle filter dropdown change
                                $('#rare_terms_filter').on('change', function() {
                                    var selectedCount = parseInt($(this).val());
                                    $('table tbody tr').each(function() {
                                        var termsCell = $(this).find('td:nth-child(3)');
                                        var scoresCell = $(this).find('td:nth-child(5)');
                                        if (termsCell.length && scoresCell.length) {
                                            var terms = termsCell.text().split(', ');
                                            var scores = scoresCell.text().split(', ');
                                            var newTerms = terms.slice(0, selectedCount).join(', ');
                                            var newScores = scores.slice(0, selectedCount).join(', ');
                                            termsCell.text(newTerms);
                                            scoresCell.text(newScores);
                                        }
                                    });
                                });
                            });
                        </script>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Analysis Result</h1>
                            <div class="filter-container">
                                <label for="rare_terms_filter">Show All:</label>
                                <select id="rare_terms_filter">
                                    <option value="0">All</option>
                                    <option value="1">1</option>
                                    <option value="2">2</option>
                                    <option value="3">3</option>
                                    <option value="4">4</option>
                                    <option value="5">5</option>
                                    <option value="6">6</option>
                                    <option value="7">7</option>
                                    <option value="8">8</option>
                                    <option value="9">9</option>
                                    <option value="10">10</option>
                                </select>
                            </div>
                            {{ table | safe }}
                        </div>
                    </body>
                    </html>
                    """, table=output_html)
    except Exception as e:
        return render_template_string("<h2>Error occurred: {{ error }}</h2>", error=e)


@app.route('/')
def index():
    return render_template_string("""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>CSV Analyzer</title>
                        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                        <style>
                            .container {
                                margin-top: 50px;
                            }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Upload a CSV file</h1>
                            <form action="/upload" method="post" enctype="multipart/form-data">
                                <div class="form-group">
                                    <input type="file" name="file" class="form-control-file">
                                </div>
                                <button type="submit" class="btn btn-primary">Upload</button>
                            </form>
                        </div>
                    </body>
                    </html>
                    """)


if __name__ == "__main__":
    app.run(debug=True)
