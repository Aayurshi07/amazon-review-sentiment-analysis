# Amazon Review Sentiment & Product Insights Pipeline

This project started with a simple question: can I take a large set of real product reviews and actually figure out what's making customers unhappy, not just whether they're happy or not? It grew into a full pipeline covering database design, two different sentiment models, topic modeling, a live app, and a Power BI dashboard.

Dataset: 15,587 Amazon Electronics reviews and 10,846 matched products, from the Amazon Reviews 2023 dataset (McAuley Lab, Hugging Face).

Live app: https://amazon-review-sentiment-app.streamlit.app
App source code repo: https://github.com/Aayurshi07/amazon-review-sentiment-app

---

## Why this project

Most beginner sentiment analysis projects stop at "classify review as positive or negative." I wanted to go a step further and answer a more business-relevant question: if you're a product manager or seller, which specific products should you be worried about, and what are people actually complaining about? That meant combining a proper relational database, two different modeling approaches, and topic modeling on top, instead of just running one sentiment classifier and calling it done.

---

## 1. Getting the data and building the database

The dataset comes from the McAuley Lab's Amazon Reviews 2023 release. The usual way to pull this is Hugging Face's `load_dataset()` function, but that script was outdated and kept failing for this particular dataset. Instead I pulled the raw parquet/JSONL files directly via their URLs and loaded them manually in Colab. Small annoyance, but a useful reminder that "standard" loading functions aren't always reliable and you sometimes need a fallback.

Once I had the raw review and product data, I built a proper relational database in SQLite instead of just working off a flat dataframe. Three tables:

- `products` — ASIN, category, store name, product-level metadata
- `reviews` — review text, star rating, timestamps, helpful votes, and later the sentiment scores from both models
- `customers` — reviewer info, linked back to reviews

I used foreign keys to connect them properly and practiced writing real analytical SQL on top: joins across all three tables, GROUP BY with HAVING for category-level rollups, CASE WHEN for bucketing sentiment, and window functions to rank products within categories by how negative their reviews were. This part doubled as SQL practice since I was learning it alongside the project.

SQL queries are in `sql/analysis_queries.sql`.

---

## 2. Sentiment analysis: VADER vs DistilBERT

I didn't want to just pick one sentiment model and move on, so I ran two very different approaches on the same reviews and compared them.

**VADER** — a lexicon-based model, basically a big dictionary of words with sentiment weights plus some rules for things like negation and punctuation. Fast, no GPU needed, no training required.

**DistilBERT** — a transformer model, fine-tuned for sentiment classification. Slower, heavier, but understands context in a way a word-lookup approach can't.

Results:

| Model | Accuracy | Recall on negative reviews |
|---|---|---|
| VADER | 89.9% | 51% |
| DistilBERT | lower than VADER | 93% |

At first glance VADER looks better since its accuracy is higher. But accuracy is misleading here because the dataset is skewed toward positive reviews (most Amazon reviews are positive), so a model can get a high accuracy score just by being decent at spotting the majority class and mediocre at the minority class. VADER was missing about half of all negative reviews. If the actual goal is catching unhappy customers before they churn or leave a public complaint, missing half of them is a real problem.

DistilBERT caught 93% of negative reviews. Lower headline accuracy, much better at the thing that actually matters for this use case. I picked DistilBERT as the production model for exactly this reason. This is the same precision/recall tradeoff logic I used in an earlier churn prediction project, so it was good to see the same principle apply here.

DistilBERT is the model powering the live sentiment predictor in the Streamlit app.

---

## 3. Finding out what people are actually complaining about

Sentiment score alone doesn't tell you *why* someone is unhappy. So I went a level deeper on the negative reviews specifically.

**Keyword comparison** — looked at which words show up disproportionately more in negative reviews compared to positive ones. Some words were up to 9x more common in negative reviews — things like "return," "waste," "disappointed," which isn't surprising but was good to confirm quantitatively rather than assume.

**LDA topic modeling** — ran Latent Dirichlet Allocation on the negative reviews to pull out recurring themes without manually labeling anything. Six themes came out fairly cleanly:

1. Camera / imaging quality
2. Tablet or case fit issues
3. Audio / Bluetooth problems
4. Connectivity and adapter issues
5. Battery life and smart device app complaints
6. Charging hardware and price complaints

I also built a word cloud and a co-occurrence network graph to visualize which complaint words tend to show up together in the same reviews — useful for spotting whether, say, "battery" complaints usually come bundled with "app" complaints (they did, which is what led to that being one combined topic instead of two separate ones).

Images: `images/negative_wordcloud.png`, `images/complaint_network_graph.png`

---

## 4. Power BI dashboard

The notebook and SQL work is good for showing technical depth, but it's not something a non-technical stakeholder would want to look at. So I built a Power BI dashboard on top of the same data, meant to be something a product manager could actually open and use.

What's on it:

- KPI cards for Total Reviews, Positive %, Negative %, and Average Rating
- A line chart tracking Positive % vs Negative % over time (by year/month)
- A bar chart ranking product categories by Negative %, worst to best
- A donut chart showing the breakdown of the six complaint topics from the LDA modeling
- A table of individual products sorted by Negative %, so someone could actually click in and see which specific ASINs need attention

One honest note on this: some of the smaller categories (Baby, Pet Supplies, Software) show Negative % close to 50% mostly because they only have a handful of reviews each, not because they're actually that problematic. With a bigger sample size those numbers would likely settle down. I left it as is rather than filtering it out, but it's worth knowing if you're reading the chart.

File: `powerbi/amazon_sentiment_dashboard.pbix`

![Power BI Dashboard](images/powerbi_dashboard_screenshot.png)

---

## 5. The live app

Built a 4-page Streamlit app so the sentiment model isn't just sitting in a notebook:

- **Overview** — project summary, headline numbers
- **Live Sentiment Predictor** — type in any review text and get a real-time DistilBERT prediction with confidence score
- **Analytics Dashboard** — Plotly charts, similar data to the Power BI dashboard but interactive in-browser
- **Data Explorer** — filter and browse the underlying review dataset directly

One limitation worth mentioning: the DistilBERT model reads up to 512 tokens (roughly 350-400 words). Longer reviews get truncated, which can occasionally throw off the prediction if the actual sentiment-defining part of the review comes near the end. It's noted directly in the app.

Live link: https://amazon-review-sentiment-app.streamlit.app

---

## Tools used

- **Python** — Google Colab as the main environment
- **Database** — SQLite, built and queried through Python
- **NLP/ML** — VADER, DistilBERT (Hugging Face Transformers), gensim for LDA, scikit-learn
- **Visualization** — Matplotlib, Plotly, Power BI Desktop
- **Deployment** — Streamlit Community Cloud

---

## What I'd do differently / next time

- Pull in more categories or a larger sample so the category-level Negative % numbers in the dashboard are more statistically reliable
- Add product titles to the dataset instead of relying on ASIN alone — makes the at-risk products table more readable at a glance
- Try fine-tuning DistilBERT specifically on this dataset instead of using it out of the box, to see if recall on negative reviews improves further

---

## Repo structure

```
amazon-review-sentiment-analysis/
├── README.md
├── notebooks/
│   └── review_sentiment_pipeline.ipynb
├── sql/
│   └── analysis_queries.sql
├── images/
│   ├── negative_wordcloud.png
│   ├── complaint_network_graph.png
│   └── powerbi_dashboard_screenshot.png
└── powerbi/
    └── amazon_sentiment_dashboard.pbix
```
