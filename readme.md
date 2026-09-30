# AI Study Time Recommendation System

An intelligent CLI application designed to estimate and optimize study schedules using Machine Learning algorithms. The system analyzes key academic and personal productivity metrics—such as exam urgency, subject difficulty, personal retention rates, sleep patterns, and stress levels—to recommend tailored daily study durations and structured study sessions.

---

## Key Features

- **Pure Python Implementation:** Runs out-of-the-box using pure Python standard libraries without external dependency requirements.
- **Machine Learning Integration:** Uses standard linear algebra implementations for feature scaling, matrix operations, and Ridge Regression modeling.
- **Synthetic Training Pipeline:** Trains locally on generated data points simulating student productivity patterns.
- **Multi-Subject Support:** Predicts required study time across multiple subjects concurrently.
- **Custom Schedule Breakdown:** Constructs personalized study schedules (e.g., Pomodoro blocks, deep work sessions, active recall intervals) based on predicted daily time requirements.
- **Robust Input Validation:** Sanitizes user inputs to prevent runtime errors and ensure reliable predictions.

---

## Prerequisites & Installation

### Requirements
- **Python:** Version 3.8 or higher.

### Setup
1. Clone or download the source code repository.
2. Open a terminal or command prompt in the project directory.
3. No third-party package installation (`pip`) is required for the standard version.

---

## Usage Instructions

To launch the system, execute the main Python script from your terminal:

```bash
python main.py
```

### System Inputs Requested

When prompted, input the following details for each subject:

1. **Student Name:** Name used to identify the study schedule output.
2. **Subject Name:** Title of the course or topic (e.g., Mathematics, Data Structures).
3. **Subject Difficulty (1–5):** Scale from 1 (*Very Easy*) to 5 (*Extremely Hard*).
4. **Prior Knowledge (1–5):** Current understanding level from 1 (*Beginner*) to 5 (*Expert*).
5. **Stress Level (1–5):** Perceived stress regarding the subject from 1 (*Low*) to 5 (*High*).
6. **Average Sleep (Hours):** Average daily sleep duration (4.0 to 10.0 hours).
7. **Days Until Exam:** Timeline remaining prior to test day (1 to 90 days).
8. **Target Score (50–100):** Desired exam percentage goal.
9. **Memory Retention Rate (0.1–1.0):** Estimated retention efficiency from 0.1 (*Poor*) to 1.0 (*Excellent*).

---

## Project Structure

```text
ai-study-recommendation/
│
├── main.py           # Core implementation (ML model, scheduler, CLI application)
└── README.md         # Project documentation and usage guide
```

---

## Machine Learning Architecture

The system computes study recommendations using a custom Ridge Regression pipeline:

1. **Feature Scaling:** Standardizes input variables to zero mean and unit variance.
2. **Synthetic Data Engine:** Generates training samples incorporating domain-specific relationships (e.g., target scores and difficulty increase required hours, while retention and prior knowledge decrease them).
3. **Ridge Regression:** Computes parameters using explicit matrix inversion $(X^T X + \alpha I)^{-1} X^T y$ with L2 regularization to ensure stable convergence.

---

## Example Output

```text
==================================================
      RECOMMENDATION SUMMARY FOR ALEX
==================================================
Total Combined Daily Recommended Study Time: 3.50 Hours

Subject: Machine Learning
  Target Score: 90.0 | Days Left: 12
  Recommended Time: 3.5 Hours/day
  Suggested Breakdown:
    - [Block 1] (50 mins): High-difficulty topics
    - [Break] (10 mins): Short walk
    - [Block 2] (50 mins): Active recall exercises
    - [Break] (15 mins): Snack & stretch
    - [Block 3] (50 mins): Problem-solving & quiz review
--------------------------------------------------
```

---

## License

This project is licensed under the MIT License.
