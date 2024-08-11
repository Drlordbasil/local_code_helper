
# ADHD Python Idea Analyzer 2.0

ADHD Python Idea Analyzer 2.0 is a Flask-based web application designed to help developers with ADHD analyze their Python code files for focus areas, similarities with existing files, and AI-powered insights. It uses OpenAI's language models to offer detailed suggestions and improvements for uploaded Python scripts.

## Features

- **Upload and Analyze Python Files**: Easily upload one or multiple Python files to receive insights and focus area analysis.
- **Focus Area Analysis**: Identify the main focus areas in your code, such as Web Development, Data Analysis, Machine Learning, and more.
- **Similarity Search**: Find similar files from the database to understand how your current work relates to past projects.
- **AI-Powered Insights**: Get detailed insights and suggestions from a language model based on your uploaded files and historical data.
- **Interactive UI**: A clean, user-friendly interface with drag-and-drop support for file uploads and dynamic content updates.

## Setup and Installation

### Prerequisites

- Python 3.12 or later
- [Flask](https://flask.palletsprojects.com/)
- [OpenAI](https://pypi.org/project/openai/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Drlordbasil/local_code_helper.git
   cd local_code_helper
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   python app.py
   ```

4. **Open your web browser and visit:**

   ```
   http://localhost:5000
   ```

## Project Structure

- **app.py**: The main Flask application file handling routing, file uploads, and code analysis.
- **templates/index.html**: The HTML template for the main interface, including JavaScript for dynamic updates and styling.
- **uploads/**: Directory for storing uploaded Python files temporarily during analysis.
- **code_database.sqlite**: SQLite database storing file contents and embeddings for similarity search.

## Key Components

### Flask Application

- **File Uploads**: Handles multiple file uploads with a secure file naming strategy.
- **Database**: Uses SQLite for storing code files and their embeddings.
- **Focus Area Analysis**: Analyzes uploaded code to determine the focus area based on predefined keywords.

### OpenAI Integration

- **Embeddings**: Generates text embeddings for the code files to facilitate similarity searches.
- **Chat Completions**: Utilizes a language model to generate insights and suggestions for the uploaded code.

### Frontend

- **JavaScript and Chart.js**: Manages file uploads, displays analysis results, and creates radar charts for focus areas.
- **Anime.js**: Provides animations for UI elements for a more engaging user experience.

## Usage

1. **Upload Python Files**: Drag and drop or select Python files for analysis.
2. **View Analysis**: Check the identified focus areas and AI-generated insights.
3. **Explore Similar Files**: See a list of similar files from the database with similarity percentages.

## Future Improvements

- **Enhanced LLM Model**: Integrate more advanced language models for improved insights.
- **Additional Focus Areas**: Expand the list of focus areas for a more comprehensive analysis.
- **User Accounts**: Allow users to save and manage their analysis history.

## License

This project is licensed under the MIT License.

---

*Developed as a tool to assist developers with ADHD in focusing on their coding projects and improving code quality through AI-driven insights.*
