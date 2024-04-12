from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Function to find rare terms in a column
def find_rare_terms(column):
    value_counts = column.value_counts()
    rare_terms = value_counts[value_counts == 1].index.tolist()[:10]
    return sorted(rare_terms)  # Sort rare terms alphabetically

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

        # Sort the columns alphabetically
        df = df.reindex(sorted(df.columns), axis=1)

        # Process the CSV file to find rare terms
        rare_terms_list = []
        for column in df.columns:
            rare_terms_list.append({'column': column, 'terms': find_rare_terms(df[column])})

        return jsonify({'rare_terms': rare_terms_list})
    except Exception as e:
        return jsonify({'error': str(e)})

# Index route to render the index.html template
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Outliers Analysis</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
            }
            .container {
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin: 50px auto;
                max-width: 800px;
            }
            h1 {
                color: #343a40;
                margin-bottom: 20px;
                text-align: center;
            }
            input[type="file"] {
                margin-bottom: 20px;
            }
            .upload-result {
                margin-top: 20px;
            }
            .error-message {
                color: #dc3545;
                margin-top: 10px;
            }
            .progress {
                margin-top: 20px;
            }
            .progress-bar {
                transition: width 0.3s ease-in-out;
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
                background-color: #007bff;
                color: #ffffff;
                font-weight: bold;
                text-transform: uppercase;
            }
            .table tbody tr:hover {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Outliers Analysis</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="input-group mb-3">
                    <div class="custom-file">
                        <input type="file" class="custom-file-input" id="fileInput">
                        <label class="custom-file-label" for="fileInput">Choose CSV file</label>
                    </div>
                    <div class="input-group-append">
                        <button class="btn btn-primary" type="button" onclick="uploadFile()">Upload</button>
                    </div>
                </div>
                <div class="progress" style="display: none;">
                    <div class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                <div class="text-center">
                    <button class="btn btn-secondary mr-2" type="button" onclick="refresh()">Refresh</button>
                </div>
                <div id="uploadResult" class="upload-result"></div>
                <div id="errorMessage" class="error-message"></div>
            </form>
            <div id="rareTermsTable" class="table-container"></div>
        </div>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
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
                        var tableHtml = '<h2 class="mb-3">Rare Terms</h2><table class="table table-striped"><thead><tr><th>Column</th><th>Rare Terms</th></tr></thead><tbody>';
                        response.rare_terms.forEach(function(item) {
                            var sortedTerms = item.terms.sort();  // Sort terms alphabetically
                            tableHtml += '<tr><td>' + item.column + '</td><td>' + sortedTerms.join(', ') + '</td></tr>';
                        });
                        tableHtml += '</tbody></table>';
                        $('#rareTermsTable').html(tableHtml);
                    },
                    error: function(xhr, status, error) {
                        console.error('Error uploading file:', error);
                        progressDiv.hide();
                        progressBar.width('0%');
                        $('#uploadResult').html('');
                        $('#errorMessage').html('<div class="alert alert-danger" role="alert">Error uploading file: ' + error + '</div>');
                    }
                });
            }

            function refresh() {
                location.reload();
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(port=8080, debug=True)
