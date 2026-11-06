# Spam Email Classification: Old-School NLP Meets Modern AI

**Can traditional machine learning compete with modern language models? And what happens when we combine them?**

That's what I wanted to explore in this project.

I built and compared three approaches to spam email classification using Python and scikit-learn:

1. **TF-IDF + Logistic Regression** — A traditional NLP approach using word-based features.
2. **MiniLM Sentence Embeddings + Logistic Regression** — A modern approach using pretrained semantic representations.
3. **Hybrid Ensemble** — A soft-voting approach that combines the predicted probabilities from both classifiers.

Spoiler alert: The old-school approach held its own!

## Project Overview

The goal was simple: classify emails as **spam or legitimate (ham)** and explore whether modern semantic embeddings offer meaningful improvements over traditional text representations.

Rather than focusing only on accuracy, I also examined precision, recall, F1-score, and misclassified emails to understand where the models succeeded and struggled.

## Tools and Technologies

- **Python** — Data processing and modeling
- **Pandas and NumPy** — Data manipulation
- **Scikit-learn** — TF-IDF, Logistic Regression, model evaluation
- **Sentence Transformers** — Pretrained `all-MiniLM-L6-v2` embeddings
- **Jupyter Notebook** — Experimentation and analysis

## How the Models Work

### 1. TF-IDF + Logistic Regression

I started with TF-IDF, which represents emails based on the importance of their words.

These numerical features were used to train a Logistic Regression classifier.

Simple, interpretable, and surprisingly effective!

### 2. MiniLM Embeddings + Logistic Regression

Next, I used the pretrained `all-MiniLM-L6-v2` model to convert emails into dense semantic embeddings.

Unlike TF-IDF, these embeddings are designed to capture broader relationships in meaning.

I then trained another Logistic Regression classifier using these representations.

### 3. Hybrid Ensemble (Soft Voting)

After examining the classification errors, I noticed that the models didn't always make the same mistakes.

That raised an interesting question: **What if I let them work together?**

I averaged the predicted spam probabilities from both classifiers and used a threshold of 0.50 to generate the final predictions.

No complicated neural network architecture. Just a little teamwork!

## Results

| Metric | TF-IDF | MiniLM | Hybrid |
|---|---|---|---|
| Accuracy | 0.98 | 0.98 | **0.99** |
| Spam Precision | **1.00** | 0.97 | 0.99 |
| Spam Recall | 0.93 | 0.94 | **0.97** |
| Spam F1-score | 0.96 | 0.95 | **0.98** |

*Results are rounded to two decimal places and based on a single stratified train/test split.*

The hybrid ensemble achieved the strongest overall performance, particularly in identifying actual spam messages.

## What I Learned

**1. Traditional NLP is still relevant.**

TF-IDF performed just as well as MiniLM in overall accuracy. Newer and more sophisticated doesn't automatically mean better.

**2. Classification errors tell an interesting story.**

Although the individual models achieved similar accuracy, they sometimes misclassified different emails. Looking beyond aggregate metrics helped me identify an opportunity to combine them.

**3. Combining models can improve performance.**

The hybrid ensemble achieved 99% accuracy and improved spam recall to 97%, suggesting that lexical and semantic representations can provide complementary predictive information.

**4. Accuracy isn't everything.**

In spam classification, false positives and false negatives have different consequences. Precision, recall, and error analysis matter when evaluating whether a model is actually useful.

## Limitations

This was an exploratory experiment using one dataset and a single train/test split. The hybrid's performance improvement is promising, but additional validation on unseen datasets would be needed to establish how well it generalizes.

The ensemble also combines two models, which increases computational requirements compared with TF-IDF alone.

## Running the Project

1. Clone or download this repository.
2. Install the required Python libraries:

   ```bash
   pip install pandas numpy scikit-learn sentence-transformers jupyter
   ```

3. Make sure the email dataset (`emails.csv`) is available and update its path in the notebook if necessary.
4. Open the notebook in Jupyter or VS Code and run the cells in order.

The first execution of the sentence embedding model may require downloading its pretrained weights.

## Final Thoughts

My biggest takeaway? **The best model isn't necessarily the newest one. It's the one that works for the problem you're trying to solve.**

TF-IDF isn't ready for retirement, MiniLM brings something useful to the table, and apparently, teamwork works in machine learning too!

---

*This project is part of my ongoing exploration of machine learning, natural language processing, and practical AI applications.*
