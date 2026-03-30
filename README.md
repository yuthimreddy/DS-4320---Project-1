# DS 4320 Project 1:

**Name:** Yuthi Madireddy

**Computing ID:** hva4zb

**Press Release:**

**Data:**

**Pipeline:**

**License:**



## Problem Definition:

**General Problem:** Recommending Content (e.g. Netflix)
**Refined Specific Problem Statement:** Using the MovieLens 100K dataset, my project is specifically focused on how 'strong' the filtering recommendation algorithms reduce content diversity over repeated recommendation cycles, and evaluates whether a simple MMR reranking layer could measurably slow down the filter bubble formation by offering a few diverse choices here.

**Rationale for Refinement** The main concern or point of issue, is how recommending algorithms are extremely pervasive and are optimized purely for engagement. As a result, feedback loops are created where users are repeatedly shown content which already affirms their beliefs and views which narrows down the amount of diverse information they see. Rather than attempting to fully rebuild recommendation systems from scratch, this project aims to simply rerank each algorithm;s output list and reordering it to ensure there is a minimum level of genre diversity per recommendation cycle. The main scope then boils down to simulate filter bubble formation over repeated cycles and test whether reranking slows down the reduction of content diversity.

**Motivation:** Recommendation algorithms decide what media billions of people watch, read, and engage with everyday. When these systems are designed for short-term engagement optimization rather than informational material, the downstream effects have already been recorded: increased political polarization, susceptibility to misinformation, and echo chambers. The scale of this problem makes data driven tools helpful for measuring when and how these filter bubbles form, and under what conditions they can be interrupted which is a strong step towards designing healthier recommendation systems.



**Press Release Headline:**

[Breaking the Loop: A New Approach to Content Recommendation Tones Down the Filter Bubble WITHOUT Ruining Your Feed](./press_release.md)

---

## Domain Exposition

Terminology: 