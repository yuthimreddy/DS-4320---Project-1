# DS 4320 Project 1:

**Name:** Yuthi Madireddy
**Computing ID:** hva4zb
**DOI:**
**License:**



## Problem Definition:

**General Problem:** Recommending Content (e.g. Netflix)

**Refined Specific Problem Statement:** Using the MovieLens 100K dataset, this project is specifically focused on how 'strong' the filtering recommendation algorithms reduce content diversity over repeated recommendation cycles, and evaluates whether a simple MMR re-ranking layer could measurably slow down the filter bubble formation by offering a few diverse choices here.

**Rationale for Refinement** The main concern or point of issue, is how recommending algorithms are extremely pervasive and are optimized purely for engagement. Consequently, feedback loops are created where users are repeatedly shown content which already affirms their beliefs and views which narrows down the amount of diverse information they see. Rather than attempting to fully rebuild recommendation systems from scratch, this project aims to simply re-rank each algorithm's output list and reorder it to ensure there is a minimum level of genre diversity per recommendation cycle. The main scope is to simulate filter bubble formation over repeated cycles and test whether re-ranking slows down the reduction of content diversity.

**Motivation:** Recommendation algorithms decide what media billions of people watch, read, and engage with everyday. When these systems are designed for short term engagement optimization rather than informational material, the downstream effects have already been recorded: increased political polarization, susceptibility to misinformation, and echo chambers. The scale of this problem makes data driven tools helpful for measuring when and how these filter bubbles form, and under what conditions they can be interrupted which is a strong step towards designing healthier recommendation systems.



**Press Release Headline:**

[Breaking the Loop: A New Approach to Content Recommendation Tones Down the Filter Bubble WITHOUT Ruining Your Feed](./press_release.md)

---

## Domain Exposition

### Terminology: 

| Term / KPI | Definition |
|---|---|
| **Recommender System (RS)** | An algorithm that predicts items a user will find relevant based on past behavior, ratings, or user similarity. |
| **Collaborative Filtering (CF)** | A family of RS methods that generate recommendations based on the preferences of similar users (user-based) or similar items (item-based). |
| **Filter Bubble** | A state in which a user is algorithmically exposed only to content that aligns with prior preferences, limiting viewpoint diversity (Pariser, 2011). |
| **Echo Chamber** | A group-level phenomenon where people interact primarily with others who share their views, reinforcing existing beliefs. |
| **Intra-List Diversity (ILD)** | A metric measuring how dissimilar recommended items in a single list are from each other; higher ILD = more diverse. |
| **Re-ranking** | A post-processing step applied to a recommendation list that reorders items to improve a secondary objective (e.g., diversity). |
| **MMR (Maximal Marginal Relevance)** | A re-ranking algorithm that balances relevance and diversity by selecting items that are both relevant and dissimilar to already-selected items. |
| **Feedback Loop** | The cycle where a user's consumption of recommended items shapes future recommendations, potentially amplifying narrow preferences. |
| **NDCG** | Normalized Discounted Cumulative Gain — a standard metric for ranking quality that rewards relevant items appearing higher in the list. |
| **Accuracy-Diversity Trade-off** | The empirical tension between maximizing recommendation relevance and maintaining content diversity. |

### Domain Overview:

This project intersects with machine learning, social science, and information systems. Recommender systems are part of everyday life at this point, with every major company deploying it in some sense. Every major platform such as Netflix, TikTok, Spotify,etc. uses them to decide what content users see next. The main underlying mechanism is already understood, as the filtering computes recommendations based off the user's interactions and history. By doing so, the algorithm ranks unseen items by predicting their relevance. The societal problem has also been noted down in many papers, such as https://pubs.aeaweb.org/doi/pdfplus/10.1257/aer.20191777. With optimization for short term engagement (clicks, watch time, likes) prioritized, the creation of feedback loops progressively limit down on a narrow subset of a user's interests. At the sheer scale of social media's reach, millions of individual feedback loops can compound into measurable shifts in political polarization, media consumption patterns, and health behaviors.

### Background Reading

| # | Title | Brief Description | Link |
|---|---|---|---|
| 1 | Trap of Social Media Algorithms: A Systematic Review on Filter Bubbles, Echo Chambers, and Their Impact on Youth (MDPI, 2025) | Reviews 30 studies (2015–2025) on how algorithms create ideological homogeneity; finds consistent evidence that algorithmic systems amplify selective exposure, especially for younger users. | [PDF](./background_reading/paper1_mdpi_filter_bubbles_youth.pdf) |
| 2 | Understanding Echo Chambers in Recommender Systems (IJACSA, 2025) | A systematic literature review on how collaborative filtering and content-based systems mechanically produce echo chambers; surveys detection and mitigation methods. | [PDF](./background_reading/paper2_ijacsa_echo_chambers_rs.pdf) |
| 3 | Filter Bubbles in Recommender Systems: Fact or Fallacy (arXiv, 2023) | Surveys the empirical literature and classifies studies by whether they found filter bubble evidence; identifies methodological causes of disagreement. | [PDF](./background_reading/paper3_arxiv_fact_or_fallacy.pdf) |
| 4 | Understanding Echo Chambers and Filter Bubbles (MIS Quarterly / Darden UVA, 2020) | Empirical study using Twitter data examining the relative contribution of algorithmic filtering vs. user choice in producing filter bubbles. | [PDF](./background_reading/paper4_darden_uva_filter_bubbles.pdf) |
| 5 | Algorithmic Domination and Democratic Discourse (MDPI, 2025) | Examines how disinformation, echo chambers, and filter bubbles threaten democratic institutions; proposes platform regulation and media literacy interventions. | [PDF](./background_reading/paper5_mdpi_algorithmic_domination.pdf) |

---