import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully with {len(df)} records")
            
            # Check if target column exists
            if 'target' not in df.columns:
                print("❌ 'target' column not found in dataset!")
                return None
            
            print(f"📊 Original target distribution:\n{df['target'].value_counts()}")
            print(f"📊 Original target encoding: 1 = disease, 0 = no disease")
            
            # Check if we need to reverse target encoding
            # Let's analyze the data to determine if reversal is needed
            age_corr = df[['age', 'target']].corr().iloc[0,1]
            print(f"📈 Age-Target correlation: {age_corr:.3f}")
            
            if age_corr < 0:  # Negative correlation means older people have lower target (likely reversed)
                print("🔄 Target encoding appears reversed. Reversing...")
                df['target'] = 1 - df['target']
                print("✅ Target encoding corrected: 1 = disease, 0 = no disease")
            else:
                print("✅ Target encoding is correct: 1 = disease, 0 = no disease")
            
            print(f"📊 Corrected target distribution:\n{df['target'].value_counts()}")
            
            # Handle outliers based on medical ranges
            df = self._handle_outliers(df)
            print(f"📈 After outlier handling: {len(df)} records")
            
            # Feature engineering
            df = self._create_features(df)
            
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _handle_outliers(self, df):
        """Handle outliers based on medical ranges"""
        original_count = len(df)
        
        # Cholesterol: realistic range 100-600 mg/dL
        df = df[(df['chol'] >= 100) & (df['chol'] <= 600)]
        
        # Resting blood pressure: realistic range 90-200 mmHg
        df = df[(df['trestbps'] >= 90) & (df['trestbps'] <= 200)]
        
        # Maximum heart rate: realistic range 60-220
        df = df[(df['thalach'] >= 60) & (df['thalach'] <= 220)]
        
        # Oldpeak: realistic range 0-6
        df = df[(df['oldpeak'] >= 0) & (df['oldpeak'] <= 6)]
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"🗑️ Removed {removed_count} outliers based on medical ranges")
        
        return df
    
    def _create_features(self, df):
        """Create additional features based on medical knowledge"""
        # Age categories
        df['age_category'] = pd.cut(df['age'], 
                                   bins=[0, 45, 55, 65, 100], 
                                   labels=[0, 1, 2, 3]).astype(int)
        
        # Blood pressure categories
        df['bp_category'] = pd.cut(df['trestbps'],
                                  bins=[0, 120, 130, 140, 200],
                                  labels=[0, 1, 2, 3]).astype(int)
        
        # Cholesterol categories
        df['chol_category'] = pd.cut(df['chol'],
                                    bins=[0, 200, 240, 300, 600],
                                    labels=[0, 1, 2, 3]).astype(int)
        
        # Heart rate recovery (simplified)
        df['hr_recovery'] = 220 - df['age'] - df['thalach']
        
        # Medical risk score based on clinical guidelines
        df['medical_risk_score'] = (
            (df['age'] > 55).astype(int) * 2 +         # Age
            (df['trestbps'] > 130).astype(int) * 1.5 + # Hypertension
            (df['chol'] > 200).astype(int) * 1.0 +     # High cholesterol
            df['fbs'] * 2.0 +                         # Diabetes
            df['exang'] * 3.0 +                       # Exercise angina
            (df['oldpeak'] > 1).astype(int) * 3.0 +   # ST depression
            (df['ca'] > 0).astype(int) * 4.0 +        # Blocked vessels
            (df['thal'] == 2).astype(int) * 3.0 +     # Reversible defect
            (df['slope'] == 2).astype(int) * 2.0      # Downsloping
        )
        
        # Create interaction features
        df['age_bp_risk'] = (df['age'] * df['trestbps']) / 1000
        df['chol_oldpeak_risk'] = (df['chol'] * df['oldpeak']) / 100
        
        # Risk level indicators
        df['high_risk_indicator'] = (
            (df['exang'] == 1) |
            (df['ca'] >= 2) |
            (df['thal'] == 2) |
            (df['oldpeak'] > 2)
        ).astype(int)
        
        df['moderate_risk_indicator'] = (
            (df['age'] > 50) |
            (df['trestbps'] > 130) |
            (df['chol'] > 220) |
            (df['oldpeak'] > 1) |
            (df['ca'] == 1)
        ).astype(int)
        
        print("✅ Created clinical risk features")
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Use all features including engineered ones
        feature_columns = self.feature_names + [
            'age_category', 'bp_category', 'chol_category', 'hr_recovery', 
            'medical_risk_score', 'age_bp_risk', 'chol_oldpeak_risk',
            'high_risk_indicator', 'moderate_risk_indicator'
        ]
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns]
        y = df['target']
        
        print(f"🔧 Using {len(feature_columns)} features for training")
        print(f"📋 Feature names: {feature_columns}")
        print(f"🎯 Target distribution:\n{y.value_counts()}")
        print(f"📊 Percentage of heart disease cases: {y.mean():.2%}")
        
        return X, y, feature_columns
    
    def train_model(self, X, y):
        """Train a well-calibrated Random Forest model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📚 Training set: {X_train.shape[0]} samples")
        print(f"🧪 Testing set: {X_test.shape[0]} samples")
        
        # Update feature names to match actual features used
        self.feature_names = X.columns.tolist()
        print(f"📝 Final feature names: {self.feature_names}")
        
        # Train Random Forest with balanced parameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        self.evaluate_model(model, X_test, y_test, y_pred, y_pred_proba)
        
        self.model = model
        return model, X_test, y_test, y_pred
    
    def evaluate_model(self, model, X_test, y_test, y_pred, y_pred_proba=None):
        """Evaluate model performance comprehensively"""
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Model Evaluation:")
        print(f"✅ Accuracy: {accuracy:.4f}")
        
        if y_pred_proba is not None:
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print(f"📊 AUC Score: {auc_score:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"🎯 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print(f"🎯 Sensitivity (Recall): {sensitivity:.4f}")
        print(f"🎯 Specificity: {specificity:.4f}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        print(f"\n🔍 Cross-validation scores: {cv_scores}")
        print(f"📈 Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\n🔍 Top 10 Feature Importance:")
            print(feature_importance.head(10))
        
        # Probability distribution analysis
        if y_pred_proba is not None:
            print(f"\n📈 Probability Distribution Analysis:")
            print(f"Min probability: {y_pred_proba.min():.3f}")
            print(f"Max probability: {y_pred_proba.max():.3f}")
            print(f"Mean probability: {y_pred_proba.mean():.3f}")
            
            # Probability ranges
            ranges = [
                (0, 0.3, "Low Risk"),
                (0.3, 0.7, "Moderate Risk"), 
                (0.7, 1.0, "High Risk")
            ]
            
            for low, high, risk_level in ranges:
                count = np.sum((y_pred_proba >= low) & (y_pred_proba < high))
                percentage = count / len(y_pred_proba) * 100
                print(f"  {risk_level:12} ({low:.0%}-{high:.0%}): {count:3d} samples ({percentage:5.1f}%)")
    
    def save_model(self, file_path):
        """Save the trained model and metadata"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__
        }
        joblib.dump(model_data, file_path)
        print(f"💾 Model saved to {file_path}")
        print(f"📋 Features saved: {len(self.feature_names)} features")

def main():
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load and preprocess data
    file_path = "C:\\Users\\TATAPUDI MAHALAKSHMI\\Downloads\\ml-project\\ml\\heart.csv"  # Make sure this file is in the same directory
    df = predictor.load_and_preprocess_data(file_path)
    
    if df is not None:
        # Prepare features
        X, y, feature_columns = predictor.prepare_features(df)
        
        # Train model
        model, X_test, y_test, predictions = predictor.train_model(X, y)
        
        # Save model
        predictor.save_model('heart_disease_model.joblib')
        print("\n🎉 Model training completed successfully!")
        print("🚀 Next steps:")
        print("   1. Run: python test_model.py (to test the model)")
        print("   2. Run: streamlit run app.py (to launch the web app)")
    else:
        print("❌ Failed to load data. Please check:")
        print("   - File path: heart.csv")
        print("   - File exists in the same directory")
        print("   - File contains 'target' column")

if __name__ == "__main__":
    main()