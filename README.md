🛒 Day 2: Shopping Cart Abandonment Prediction

This is the second project of my #30DaysOfAI challenge. The goal is to build and deploy a machine learning model that predicts whether an online shopper will complete a purchase or abandon their session.

This project serves as a complete, end-to-end example of a classification workflow, from data preprocessing via a standalone script to a live, interactive web application.
✨ Key Concepts & Learnings

    Imbalanced Data: The dataset is imbalanced, with only ~15% of sessions resulting in a purchase. This project explores why 'Accuracy' can be a misleading metric and focuses on Recall to minimize missed sales opportunities. The scale_pos_weight hyperparameter in XGBoost was used to address this.

    Script-Based Training: Instead of an interactive notebook, the entire model training and artifact generation process is encapsulated in a reusable Python script (train_model.py). This represents a more robust and production-friendly workflow.

    Streamlit Deployment: The final, fine-tuned model is served through a simple and interactive web application built with Streamlit.

💻 Tech Stack
🚀 How to Run Locally

    Clone the repository:

    git clone https://github.com/SuleymanToklu/Day2-Cart-Abandonment-Final.git
    cd Day2-Cart-Abandonment-Final

    Install dependencies:

    pip install -r requirements.txt

    Train the model and generate artifacts:
    This script will create the .pkl files needed by the app.

    python train_model.py

    Run the Streamlit app:

    streamlit run app.py

🔗 Live Demo

You can try the live application here: (https://day2-cart-abandonment-final-gyt7thfrhdagh82xxhbe2l.streamlit.app/)
