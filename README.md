# Faculty Finder

Faculty Finder is a project that uses semantic search to match research projects with faculty members based on their keywords. It leverages the `sentence-transformers` library to encode keywords and perform semantic search.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/faculty-finder.git
    cd faculty-finder/py
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```sh
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

5. **Download the necessary model:**

    The project uses the `all-MiniLM-L6-v2` model from `sentence-transformers`. This will be automatically downloaded when you run the code.

## Usage

1. **Prepare the data:**

    Ensure you have a [`csvjson.json`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fd%3A%2Ffaculty%20finder%2Fpy%2Fcsvjson.json%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%2262d5afc6-b9d0-436f-abbf-a3b37d50e9c9%22%5D "d:\faculty finder\py\csvjson.json") file in the project directory. This file should contain the faculty data with their keywords.

2. **Run the application:**

    ```sh
    python app.py
    ```

    This will encode the faculty keywords and save the embeddings to [`faculty_embeddings.pt`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fd%3A%2Ffaculty%20finder%2Fpy%2Ffaculty_embeddings.pt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%2262d5afc6-b9d0-436f-abbf-a3b37d50e9c9%22%5D "d:\faculty finder\py\faculty_embeddings.pt").

3. **Environment Variables:**

    Make sure to set the [`GEMINI_API_KEY`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fd%3A%2Ffaculty%20finder%2Fpy%2Fapp.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A44%2C%22character%22%3A21%7D%7D%5D%2C%2262d5afc6-b9d0-436f-abbf-a3b37d50e9c9%22%5D "Go to definition") environment variable:

    - On Windows:

        ```sh
        set GEMINI_API_KEY=your_api_key
        ```

    - On macOS/Linux:

        ```sh
        export GEMINI_API_KEY=your_api_key
        ```

## Project Structure
faculty-finder/ │ ├── py/ │ ├── app.py │ ├── requirements.txt │ ├── templates/ │ │ └── index.html │ ├── static/ │ │ ├── style.css │ │ └── logo_black.png │ └── csvjson.json │ └── README.md


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
