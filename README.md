Movie Review Sentiment Analyzer

Overview

This project is a web application built using Streamlit. It analyzes the sentiment of movie reviews (positive or negative) using a pre-trained Support Vector Machine (SVM) model and a vectorizer. The app allows users to input a review and instantly see the analysis results, along with technical metrics.

Features

Classifies movie reviews as Positive or Negative.

Provides confidence scores and additional metrics for each prediction.

Includes preprocessing to clean user inputs.

Displays review statistics (word count, character count, etc.).

Shows technical metrics (raw scores, feature count) for advanced users.

How It Works

Input:

The user enters a movie review into the text box provided.

Processing:

The review is preprocessed (e.g., special characters removed, text normalized).

The preprocessed review is transformed into a numerical feature vector using a pre-trained vectorizer (vectorizer.joblib).

Prediction:

The numerical features are passed to a pre-trained SVM model (svm_model.joblib) to predict the sentiment.

The model computes a raw score, which determines whether the sentiment is positive or negative.

Output:

The app displays the sentiment (Positive or Negative) with visual styling.

Additional metrics such as word count, feature count, and raw score are shown for transparency.

Installation

Prerequisites

Ensure the following tools are installed on your system:

Python 3.8+

pip (Python package manager)

Steps

Clone the repository:

git clone https://github.com/your-repo/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer

Install required dependencies:

pip install -r requirements.txt

Place the pre-trained model files in the root directory:

svm_model.joblib (SVM model)

vectorizer.joblib (Vectorizer)

Run the application:

streamlit run interface.py

Open your browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

Usage

Input

Enter a movie review in the text area provided.

Reviews must:

Contain at least 3 words.

Be under 5000 characters.

Output

The sentiment is displayed as either Positive or Negative.

Additional information is shown:

Word count

Character count

Feature count (number of active features in the vectorized review)

Technical metrics such as the raw prediction score

File Structure

movie-review-sentiment-analyzer/
├── app.py                 # Main application script
├── requirements.txt       # Dependencies for the project
├── svm_model.joblib       # Pre-trained SVM model file
├── vectorizer.joblib      # Pre-trained vectorizer file
└── README.md              # Project documentation (this file)

Key Components

interface.py

The main Streamlit app that:

Accepts user input.

Loads the pre-trained model and vectorizer.

Processes the input and predicts the sentiment.

Displays results and technical details.

Pre-trained Files

svm_model.joblib:

Contains the weights and bias of the SVM classifier.

vectorizer.joblib:

Contains the vocabulary and transformation logic for converting text to numerical features.

Limitations

The model is pre-trained and cannot be re-trained using this app.

Sentiment analysis is limited to the data and patterns learned during training.

Highly nuanced or ambiguous reviews may produce less accurate predictions.

Contributing

Contributions are welcome! If you find any issues or have suggestions, please open an issue or submit a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

Contact

For questions or support, please contact:

Names:El Bouazzaoui Asmaa
     El Feddani Aya
![Positive review](https://github.com/user-attachments/assets/20d0c5f4-c746-4c9c-966c-aaf0c4dccc3e)

![Negativve review](https://github.com/user-attachments/assets/0389001d-cadb-4027-8ec4-f07a25891a63)


