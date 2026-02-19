"""
Email Spam Detection System
A complete implementation with preprocessing, feature extraction, and classification
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EmailSpamDetector:
    """
    A comprehensive email spam detection system with multiple classifiers
    """
    
    def __init__(self, vectorizer_type='tfidf', max_features=3000):
        """
        Initialize the spam detector
        
        Args:
            vectorizer_type: 'tfidf' or 'count'
            max_features: Maximum number of features to extract
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        
        # Initialize vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, 
                                             stop_words='english',
                                             ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=max_features,
                                             stop_words='english',
                                             ngram_range=(1, 2))
        
        # Initialize classifiers
        self.classifiers = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        self.trained_models = {}
        self.results = {}
    
    def preprocess_text(self, text):
        """
        Preprocess email text
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, emails):
        """
        Extract additional features from emails
        
        Args:
            emails: List of email texts
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        for email in emails:
            feature_dict = {
                'length': len(email),
                'num_words': len(email.split()),
                'num_uppercase': sum(1 for c in email if c.isupper()),
                'num_exclamation': email.count('!'),
                'num_question': email.count('?'),
                'num_dollar': email.count('$'),
                'avg_word_length': np.mean([len(word) for word in email.split()]) if email.split() else 0
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def prepare_data(self, emails, labels, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            emails: List of email texts
            labels: List of labels (0=ham, 1=spam)
            test_size: Proportion of test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Preprocess emails
        print("Preprocessing emails...")
        processed_emails = [self.preprocess_text(email) for email in emails]
        
        # Vectorize text
        print(f"Extracting features using {self.vectorizer_type}...")
        X_text = self.vectorizer.fit_transform(processed_emails)
        
        # Extract additional features
        print("Extracting additional features...")
        X_additional = self.extract_features(emails)
        
        # Combine features (optional - using text features only for simplicity)
        X = X_text
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train all classifiers
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\nTraining models...")
        print("=" * 60)
        
        for name, classifier in self.classifiers.items():
            print(f"Training {name}...")
            classifier.fit(X_train, y_train)
            self.trained_models[name] = classifier
        
        print("Training complete!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\nModel Evaluation Results")
        print("=" * 60)
        
        for name, model in self.trained_models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred
            }
            
            # Print results
            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
    
    def plot_confusion_matrices(self, y_test):
        """
        Plot confusion matrices for all models
        
        Args:
            y_test: True labels
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(y_test, result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Ham', 'Spam'],
                       yticklabels=['Ham', 'Spam'])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('/home/claude/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrices saved to 'confusion_matrices.png'")
        plt.close()
    
    def plot_metrics_comparison(self):
        """
        Plot comparison of metrics across models
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        models = list(self.results.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            ax.bar(x + idx * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Metrics comparison saved to 'metrics_comparison.png'")
        plt.close()
    
    def predict(self, email, model_name='Naive Bayes'):
        """
        Predict if an email is spam or ham
        
        Args:
            email: Email text
            model_name: Name of the model to use
            
        Returns:
            Prediction (0=ham, 1=spam) and probability
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Preprocess
        processed = self.preprocess_text(email)
        
        # Vectorize
        features = self.vectorizer.transform([processed])
        
        # Predict
        model = self.trained_models[model_name]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return prediction, probability
    
    def get_top_spam_words(self, n=20):
        """
        Get the most important words for spam classification
        
        Args:
            n: Number of top words to return
            
        Returns:
            Dictionary of top spam words
        """
        if 'Logistic Regression' not in self.trained_models:
            return None
        
        model = self.trained_models['Logistic Regression']
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients
        coef = model.coef_[0]
        
        # Get top spam words (positive coefficients)
        top_spam_idx = np.argsort(coef)[-n:][::-1]
        top_spam_words = [(feature_names[i], coef[i]) for i in top_spam_idx]
        
        # Get top ham words (negative coefficients)
        top_ham_idx = np.argsort(coef)[:n]
        top_ham_words = [(feature_names[i], coef[i]) for i in top_ham_idx]
        
        return {
            'spam_words': top_spam_words,
            'ham_words': top_ham_words
        }


def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    """
    spam_emails = [
        "Congratulations! You've won a $1000 prize. Click here to claim now!!!",
        "URGENT: Your account will be closed. Verify your information immediately.",
        "Make money fast! Work from home and earn $5000 per week!!!",
        "You have been selected for a free iPhone. Click here now!",
        "Lose weight fast with this amazing pill! Order now and get 50% off!!!",
        "WINNER! You won the lottery! Send your bank details to claim $1,000,000",
        "Get rich quick! Invest now and double your money in 24 hours!!!",
        "Hot singles in your area want to meet you! Click here now!",
        "Your PayPal account has been suspended. Verify now to restore access.",
        "FREE VIAGRA! Order now and get 100 pills for the price of 10!!!",
    ]
    
    ham_emails = [
        "Hi John, can we schedule a meeting for tomorrow at 3pm?",
        "Here's the report you requested. Let me know if you need any changes.",
        "Thank you for your purchase. Your order will be delivered tomorrow.",
        "Don't forget about the team lunch on Friday at noon.",
        "Please review the attached document and provide your feedback.",
        "Your monthly bank statement is now available in your account.",
        "Reminder: Your dentist appointment is scheduled for next Monday.",
        "The project deadline has been extended to next month.",
        "Thanks for attending the conference call today. Here are the minutes.",
        "Your package has been shipped and will arrive in 3-5 business days.",
    ]
    
    # Create dataset
    emails = spam_emails + ham_emails
    labels = [1] * len(spam_emails) + [0] * len(ham_emails)
    
    return emails, labels


def main():
    """
    Main function to demonstrate the spam detection system
    """
    print("=" * 60)
    print("EMAIL SPAM DETECTION SYSTEM")
    print("=" * 60)
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    emails, labels = create_sample_dataset()
    print(f"Total emails: {len(emails)}")
    print(f"Spam emails: {sum(labels)}")
    print(f"Ham emails: {len(labels) - sum(labels)}")
    
    # Initialize detector
    detector = EmailSpamDetector(vectorizer_type='tfidf', max_features=1000)
    
    # Prepare data
    X_train, X_test, y_train, y_test = detector.prepare_data(emails, labels, test_size=0.3)
    
    # Train models
    detector.train_models(X_train, y_train)
    
    # Evaluate models
    detector.evaluate_models(X_test, y_test)
    
    # Plot results
    detector.plot_confusion_matrices(y_test)
    detector.plot_metrics_comparison()
    
    # Get top spam words
    print("\n" + "=" * 60)
    print("TOP SPAM INDICATORS")
    print("=" * 60)
    top_words = detector.get_top_spam_words(n=10)
    if top_words:
        print("\nTop Spam Words:")
        for word, score in top_words['spam_words']:
            print(f"  {word}: {score:.4f}")
    
    # Test with new emails
    print("\n" + "=" * 60)
    print("TESTING WITH NEW EMAILS")
    print("=" * 60)
    
    test_emails = [
        "Get a free loan now! No credit check required!!!",
        "Hi, let's meet for coffee tomorrow afternoon.",
        "URGENT: Wire transfer needed immediately! Send money now!",
    ]
    
    for email in test_emails:
        prediction, probability = detector.predict(email, model_name='Naive Bayes')
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = probability[prediction] * 100
        
        print(f"\nEmail: {email[:60]}...")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
