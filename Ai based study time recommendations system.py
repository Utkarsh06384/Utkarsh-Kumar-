import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


class SyntheticDataEngine:
    def __init__(self, samples=1000, seed=42):
        self.samples = samples
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        difficulty = np.random.randint(1, 6, self.samples)
        prior_knowledge = np.random.randint(1, 6, self.samples)
        sleep_hours = np.random.uniform(4.0, 9.5, self.samples)
        days_until_exam = np.random.randint(1, 61, self.samples)
        target_score = np.random.uniform(50.0, 100.0, self.samples)
        stress_level = np.random.randint(1, 6, self.samples)
        retention_rate = np.random.uniform(0.3, 1.0, self.samples)
        
        base_hours = (
            (target_score * 0.09) + 
            (difficulty * 1.35) + 
            (stress_level * 0.45) - 
            (prior_knowledge * 0.95) - 
            (sleep_hours * 0.35) - 
            (retention_rate * 1.5)
        )
        time_factor = np.where(days_until_exam < 7, 1.4, np.where(days_until_exam < 14, 1.2, 1.0))
        noise = np.random.normal(0, 0.4, self.samples)
        study_hours = (base_hours * time_factor) + noise
        study_hours = np.clip(study_hours, 0.5, 14.0)
        
        return pd.DataFrame({
            'difficulty': difficulty,
            'prior_knowledge': prior_knowledge,
            'sleep_hours': sleep_hours,
            'days_until_exam': days_until_exam,
            'target_score': target_score,
            'stress_level': stress_level,
            'retention_rate': retention_rate,
            'recommended_study_hours': study_hours
        })


class ModelTrainer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge())
        ])
        self.param_grid = {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        self.best_model = None

    def train_and_evaluate(self, dataframe):
        features = [
            'difficulty', 'prior_knowledge', 'sleep_hours', 
            'days_until_exam', 'target_score', 'stress_level', 'retention_rate'
        ]
        X = dataframe[features]
        y = dataframe['recommended_study_hours']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        predictions = self.best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

    def predict(self, feature_array):
        return self.best_model.predict(feature_array)


class InputValidator:
    @staticmethod
    def get_bounded_float(prompt, min_val, max_val):
        while True:
            try:
                val = float(input(prompt))
                if min_val <= val <= max_val:
                    return val
                print(f"Value must be between {min_val} and {max_val}.")
            except ValueError:
                print("Invalid entry. Please enter a valid numerical value.")

    @staticmethod
    def get_bounded_int(prompt, min_val, max_val):
        while True:
            try:
                val = int(input(prompt))
                if min_val <= val <= max_val:
                    return val
                print(f"Value must be between {min_val} and {max_val}.")
            except ValueError:
                print("Invalid entry. Please enter a valid integer.")

    @staticmethod
    def get_non_empty_string(prompt):
        while True:
            val = input(prompt).strip()
            if val:
                return val
            print("Input cannot be empty.")


class ScheduleOptimizer:
    @staticmethod
    def generate_plan(daily_hours):
        if daily_hours <= 2.0:
            return [
                {"block": "Session 1", "duration": "45 mins", "focus": "Core concept reading"},
                {"block": "Break", "duration": "10 mins", "focus": "Rest & hydration"},
                {"block": "Session 2", "duration": "45 mins", "focus": "Practice questions"}
            ]
        elif daily_hours <= 5.0:
            return [
                {"block": "Block 1", "duration": "50 mins", "focus": "High-difficulty topics"},
                {"block": "Break", "duration": "10 mins", "focus": "Short walk"},
                {"block": "Block 2", "duration": "50 mins", "focus": "Active recall exercises"},
                {"block": "Break", "duration": "15 mins", "focus": "Snack & stretch"},
                {"block": "Block 3", "duration": "50 mins", "focus": "Problem-solving & quiz review"}
            ]
        else:
            return [
                {"block": "Morning Block", "duration": "90 mins", "focus": "Deep work on hardest concepts"},
                {"block": "Break", "duration": "20 mins", "focus": "Complete disconnection"},
                {"block": "Midday Block", "duration": "90 mins", "focus": "Application and problem sets"},
                {"block": "Break", "duration": "30 mins", "focus": "Meal / major break"},
                {"block": "Afternoon Block", "duration": "60 mins", "focus": "Spaced repetition flashcards"},
                {"block": "Break", "duration": "15 mins", "focus": "Rest"},
                {"block": "Evening Review", "duration": "45 mins", "focus": "Summary notes & error logging"}
            ]


class Exporter:
    @staticmethod
    def export_to_json(data, filename="study_plan.json"):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nPlan exported successfully to {filename}")


class Application:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.validator = InputValidator()
        self.optimizer = ScheduleOptimizer()

    def run(self):
        print("Initializing AI Recommendation Engine...")
        engine = SyntheticDataEngine(samples=1500)
        dataset = engine.generate()
        mse, r2 = self.trainer.train_and_evaluate(dataset)
        
        print(f"Model Training Complete. R2 Score: {r2:.3f} | MSE: {mse:.3f}\n")
        print("=========================================")
        print("    STUDY TIME RECOMMENDATION SYSTEM     ")
        print("=========================================\n")
        
        student_name = self.validator.get_non_empty_string("Enter Student Name: ")
        subject_count = self.validator.get_bounded_int("How many subjects are you planning for? (1-5): ", 1, 5)
        
        overall_plan = {
            "student_name": student_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subjects": []
        }
        
        total_daily_recommended = 0.0
        
        for idx in range(subject_count):
            print(f"\n--- Entering details for Subject {idx + 1} ---")
            subject_name = self.validator.get_non_empty_string("Subject Name: ")
            difficulty = self.validator.get_bounded_int("Subject Difficulty (1=Very Easy, 5=Extremely Hard): ", 1, 5)
            prior_knowledge = self.validator.get_bounded_int("Prior Knowledge Level (1=Beginner, 5=Expert): ", 1, 5)
            stress_level = self.validator.get_bounded_int("Current Stress Level (1=Low, 5=High): ", 1, 5)
            sleep_hours = self.validator.get_bounded_float("Average Sleep Hours per Night (4.0-10.0): ", 4.0, 10.0)
            days_until_exam = self.validator.get_bounded_int("Days Until Exam (1-90): ", 1, 90)
            target_score = self.validator.get_bounded_float("Target Exam Score (50.0-100.0): ", 50.0, 100.0)
            retention = self.validator.get_bounded_float("Estimated Memory Retention Rate (0.1=Poor, 1.0=Excellent): ", 0.1, 1.0)
            
            features = np.array([[
                difficulty, prior_knowledge, sleep_hours, 
                days_until_exam, target_score, stress_level, retention
            ]])
            
            raw_prediction = self.trainer.predict(features)[0]
            recommended_hours = round(float(np.clip(raw_prediction, 0.5, 12.0)), 2)
            total_daily_recommended += recommended_hours
            
            schedule = self.optimizer.generate_plan(recommended_hours)
            
            overall_plan["subjects"].append({
                "subject_name": subject_name,
                "recommended_daily_hours": recommended_hours,
                "days_left": days_until_exam,
                "target_score": target_score,
                "breakdown": schedule
            })
            
        print("\n" + "=" * 50)
        print(f"      RECOMMENDATION SUMMARY FOR {student_name.upper()}")
        print("=" * 50)
        print(f"Total Combined Daily Recommended Study Time: {total_daily_recommended:.2f} Hours\n")
        
        for subj in overall_plan["subjects"]:
            print(f"Subject: {subj['subject_name']}")
            print(f"  Target Score: {subj['target_score']} | Days Left: {subj['days_left']}")
            print(f"  Recommended Time: {subj['recommended_daily_hours']} Hours/day")
            print("  Suggested Breakdown:")
            for step in subj['breakdown']:
                print(f"    - [{step['block']}] ({step['duration']}): {step['focus']}")
            print("-" * 50)
            
        export_choice = input("\nDo you want to save this plan to JSON? (y/n): ").strip().lower()
        if export_choice == 'y':
            Exporter.export_to_json(overall_plan)
            
        print("\nThank you for using the System. Happy Studying!")


if __name__ == "__main__":
    app = Application()
    app.run()
