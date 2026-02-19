"""
Advanced Email Spam Detection with Real Dataset Support
Includes feature engineering and model persistence
"""

import pickle
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class AdvancedSpamDetector:
    """
    Advanced spam detector with model persistence and cross-validation
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.training_history = []
    
    def load_dataset(self, filepath, text_col='text', label_col='label'):
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            text_col: Column name for email text
            label_col: Column name for labels
            
        Returns:
            DataFrame with emails and labels
        """
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def perform_cross_validation(self, X, y, model, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            model: Model to evaluate
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        return scores
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning for Logistic Regression
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Best model
        """
        print("\nPerforming hyperparameter tuning...")
        
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
        
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, filepath='spam_detector_model.pkl'):
        """
        Save trained model and vectorizer
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None or self.vectorizer is None:
            print("No model to save. Train the model first.")
            return
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'training_date': datetime.now().isoformat(),
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='spam_detector_model.pkl'):
        """
        Load trained model and vectorizer
        
        Args:
            filepath: Path to load the model from
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.training_history = model_data.get('training_history', [])
            
            print(f"Model loaded from {filepath}")
            print(f"Training date: {model_data.get('training_date', 'Unknown')}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def analyze_dataset(self, emails, labels):
        """
        Analyze dataset statistics
        
        Args:
            emails: List of emails
            labels: List of labels
        """
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        
        df = pd.DataFrame({'email': emails, 'label': labels})
        
        # Basic statistics
        print(f"\nTotal emails: {len(emails)}")
        print(f"Spam emails: {sum(labels)} ({sum(labels)/len(labels)*100:.2f}%)")
        print(f"Ham emails: {len(labels) - sum(labels)} ({(1-sum(labels)/len(labels))*100:.2f}%)")
        
        # Length statistics
        spam_lengths = [len(email) for email, label in zip(emails, labels) if label == 1]
        ham_lengths = [len(email) for email, label in zip(emails, labels) if label == 0]
        
        print(f"\nAverage spam email length: {np.mean(spam_lengths):.2f} characters")
        print(f"Average ham email length: {np.mean(ham_lengths):.2f} characters")
        
        # Word count statistics
        spam_words = [len(email.split()) for email, label in zip(emails, labels) if label == 1]
        ham_words = [len(email.split()) for email, label in zip(emails, labels) if label == 0]
        
        print(f"\nAverage spam email words: {np.mean(spam_words):.2f}")
        print(f"Average ham email words: {np.mean(ham_words):.2f}")


def create_comprehensive_report(detector, X_test, y_test, output_file='spam_detection_report.txt'):
    """
    Create a comprehensive analysis report
    
    Args:
        detector: Trained spam detector
        X_test: Test features
        y_test: Test labels
        output_file: Path to save report
    """
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("EMAIL SPAM DETECTION - COMPREHENSIVE REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Evaluate each model
    for name, model in detector.trained_models.items():
        report_lines.append(f"\n\n{'='*70}")
        report_lines.append(f"MODEL: {name}")
        report_lines.append('='*70)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Classification report
        report_lines.append("\nClassification Report:")
        report_lines.append("-" * 70)
        report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        report_lines.append(report)
        
        # ROC AUC Score
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            report_lines.append(f"\nROC AUC Score: {auc:.4f}")
    
    # Write report
    report_text = '\n'.join(report_lines)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nComprehensive report saved to {output_file}")
    return report_text


def demo_real_world_usage():
    """
    Demonstrate real-world usage scenarios
    """
    print("\n" + "=" * 60)
    print("REAL-WORLD USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Example spam patterns
    spam_patterns = {
        'Financial Scam': "Congratulations! You've won $10,000. Send your bank details to claim.",
        'Phishing': "Your account has been compromised. Click here to verify your credentials.",
        'Product Spam': "Buy Viagra online! Lowest prices guaranteed. Order now!!!",
        'Job Scam': "Work from home and earn $5000 per week! No experience needed!",
        'Prize Scam': "You are the lucky winner of our lottery! Claim your prize now!"
    }
    
    ham_examples = {
        'Business Email': "Hi Sarah, please review the Q4 financial report attached.",
        'Personal Email': "Hey! Want to grab lunch tomorrow? Let me know what time works.",
        'Notification': "Your package has been shipped and will arrive on Monday.",
        'Newsletter': "This week in tech: AI breakthroughs and new product launches.",
        'Confirmation': "Thank you for your order. Your confirmation number is #12345."
    }
    
    print("\nCommon Spam Patterns:")
    for category, example in spam_patterns.items():
        print(f"\n  {category}:")
        print(f"    \"{example}\"")
    
    print("\n" + "-" * 60)
    print("\nLegitimate Email Examples:")
    for category, example in ham_examples.items():
        print(f"\n  {category}:")
        print(f"    \"{example}\"")
    
    # Feature importance tips
    print("\n" + "=" * 60)
    print("KEY SPAM INDICATORS")
    print("=" * 60)
    
    indicators = {
        'High Priority': [
            'Urgent action required',
            'Verify account/password',
            'Winner/Prize/Lottery',
            'Click here immediately',
            'Wire transfer requests',
            'Suspicious URLs'
        ],
        'Medium Priority': [
            'Excessive exclamation marks',
            'ALL CAPS text',
            'Get rich quick schemes',
            'Guaranteed returns',
            'Free money/prizes',
            'Limited time offers'
        ],
        'Low Priority': [
            'Spelling errors',
            'Generic greetings',
            'Unsolicited attachments',
            'Too many links',
            'Suspicious sender address'
        ]
    }
    
    for priority, items in indicators.items():
        print(f"\n{priority}:")
        for item in items:
            print(f"  • {item}")


def create_feature_importance_visualization(detector, output_file='feature_importance.png'):
    """
    Create visualization of feature importance
    
    Args:
        detector: Trained spam detector
        output_file: Path to save visualization
    """
    if 'Logistic Regression' not in detector.trained_models:
        return
    
    model = detector.trained_models['Logistic Regression']
    feature_names = detector.vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get top features
    top_n = 15
    top_positive_idx = np.argsort(coefficients)[-top_n:]
    top_negative_idx = np.argsort(coefficients)[:top_n]
    
    top_features = np.concatenate([top_negative_idx, top_positive_idx])
    top_coefficients = coefficients[top_features]
    top_names = feature_names[top_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if c < 0 else 'red' for c in top_coefficients]
    y_pos = np.arange(len(top_names))
    
    ax.barh(y_pos, top_coefficients, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance for Spam Detection\n(Green = Ham, Red = Spam)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance visualization saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Import the main detector class
    from spam_detection import EmailSpamDetector, create_sample_dataset
    
    print("=" * 60)
    print("ADVANCED SPAM DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create and train detector
    emails, labels = create_sample_dataset()
    detector = EmailSpamDetector(vectorizer_type='tfidf', max_features=1000)
    
    X_train, X_test, y_train, y_test = detector.prepare_data(emails, labels, test_size=0.3)
    detector.train_models(X_train, y_train)
    detector.evaluate_models(X_test, y_test)
    
    # Create comprehensive report
    create_comprehensive_report(detector, X_test, y_test)
    
    # Create feature importance visualization
    create_feature_importance_visualization(detector, '/home/claude/feature_importance.png')
    
    # Demonstrate real-world usage
    demo_real_world_usage()
    
    print("\n" + "=" * 60)
    print("Advanced analysis complete!")
    print("=" * 60)
