# Amazon Review Sentiment and Product Insights

I wanted to find out what makes customers unhappy with a product, not just whether a review is positive or negative. So I built a pipeline with a SQLite database, two sentiment models, topic modeling, a Power BI dashboard and a live app.

Dataset: 15,587 Amazon Electronics reviews and 10,846 matched products from Amazon Reviews 2023 (McAuley Lab, Hugging Face).

Live app: https://amazon-review-sentiment-app.streamlit.app
App code: https://github.com/Aayurshi07/amazon-review-sentiment-app

## 1. Data and database

Hugging Face's load_dataset() kept failing for this dataset, so I downloaded the raw files from their URLs and loaded them manually in Colab.

I put the data in SQLite with three tables: products, reviews (text, rating, timestamps, sentiment scores) and customers, linked with foreign keys. I wrote joins, GROUP BY with HAVING, CASE WHEN and window functions to rank products by how negative their reviews were. I was learning SQL alongside this project, so it doubled as practice.

Queries are in `sql/analysis_queries.sql`.

## 2. VADER vs DistilBERT

I ran both models on the same reviews.

| Model | Accuracy | Recall on negative reviews |
|---|---|---|
| VADER | 89.9% | 51% |
| DistilBERT | [DistilBERT accuracy] | 93% |

VADER has the higher accuracy, but most reviews are positive, so accuracy hides how many negative ones it misses. It missed about half of them. If the goal is to catch unhappy customers, that is a problem, so I used DistilBERT. It's the same precision/recall trade-off as in my churn project.

## 3. What people complain about

- Some words were up to 9x more common in negative reviews than in positive ones, like "return", "waste" and "disappointed".
- I ran LDA topic modeling on the negative reviews and got six themes: camera quality, tablet or case fit, audio and Bluetooth, connectivity and adapters, battery life and smart device apps, and charging hardware and price.
- I also made a word cloud and a word co-occurrence graph. Battery and app complaints often appeared together, which is why they became one topic.

Images: `images/negative_wordcloud.png`, `images/complaint_network_graph.png`

## 4. Power BI dashboard

- KPI cards: total reviews, positive %, negative %, average rating
- Positive vs negative % over time
- Categories ranked by negative %
- Donut chart of the six complaint topics
- Table of products sorted by negative %

Small categories like Baby, Pet Supplies and Software show close to 50% negative because they have only a few reviews each, not because they are really that bad. I left them in.

File: `powerbi/amazon_sentiment_dashboard.pbix`

![Power BI Dashboard](images/powerbi_dashboard_screenshot.png)

## 5. Streamlit app

Four pages: an overview, a live sentiment predictor (DistilBERT with a confidence score), an analytics dashboard in Plotly, and a data explorer.

DistilBERT reads up to 512 tokens (about 350 to 400 words), so longer reviews get cut off. This can change the prediction if the key part of the review is near the end.

## Tools

Python (Google Colab), SQLite, VADER, DistilBERT (Hugging Face Transformers), gensim, scikit-learn, Matplotlib, Plotly, Power BI, Streamlit Community Cloud

## Next time

- Use more categories or a bigger sample so the category numbers are more reliable
- Add product titles, so the at-risk table is easier to read than ASINs
- Fine-tune DistilBERT on this dataset and see if negative recall improves

## Repo structure

    amazon-review-sentiment-analysis/
    ├── README.md
    ├── notebooks/review_sentiment_pipeline.ipynb
    ├── sql/analysis_queries.sql
    ├── images/
    └── powerbi/amazon_sentiment_dashboard.pbix
