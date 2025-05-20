# SMS Spam Detection

## Overview
SMS Spam Detection is a machine learning model that predicts whether an SMS message is spam or not. The model is built using Python and deployed on the web using Streamlit.

## Technologies Used
- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Streamlit**

## Features
- Data collection
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Model building and selection
- Web deployment using Streamlit

## Data Collection
The dataset used for this project is the **SMS Spam Collection** dataset, which contains over 5,500 labeled SMS messages. The dataset was obtained from Kaggle.

🔗 **Download Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Data Cleaning and Preprocessing
To ensure the dataset is clean and ready for modeling, the following preprocessing steps were performed:
- Handling missing and duplicate values
- Label encoding the "type" column (spam or ham)
- Tokenization
- Removal of special characters, stop words, and punctuation
- Stemming
- Converting all text to lowercase

## Exploratory Data Analysis (EDA)
EDA was performed to understand the dataset better. The following insights were gathered:
- Character, word, and sentence counts per message
- Correlation analysis between different features
- Visualizations using bar charts, pie charts, and heatmaps
- Word clouds for spam and non-spam messages
- Most frequent words in spam texts

## Web Deployment
The trained model is deployed using **Streamlit**, providing a simple web interface where users can input a message to check if it is spam or not.

## Installation & Usage
To run the SMS Spam Detection model locally, follow these steps:

### **1️⃣ Clone the Repository**

git clone https://github.com/pranav-ib/sms-spam-detection.git
cd sms-spam-detection

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py

4️⃣ Access the Web App

Open your browser and go to:

http://localhost:8501

Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for improvements or bug fixes.
```sh
