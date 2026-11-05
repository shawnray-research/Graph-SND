**To the Editors, Journal of Machine Learning Research**

**Submission:** *Measuring Pairwise Diversity with Linearly Many Comparisons*
**Author:** Shawn Ray, Carnegie Mellon University, 5000 Forbes Avenue, Pittsburgh, PA 15213, USA (shawnray@cmu.edu)
**Files:** `main.pdf` (the article, 50 pages including the proofs of the main results and the references) and `supplement.pdf` (17 pages of supplementary material: notation, experimental details, additional results, exact formulas for separable dissimilarities, three short proofs, and five remarks).

Dear Editors,

Please consider this manuscript for publication in JMLR.

Diversity is most often measured as the average dissimilarity over all pairs of a population. Examples are System Neural Diversity (SND) for multi-agent teams, which JMLR published in 2025, ensemble disagreement, Rao's quadratic entropy, and judge-scored diversity of language-model outputs; close relatives such as Self-BLEU are also standard. Exact evaluation needs a quadratic number of comparisons, and each comparison can be a rollout or a call to a learned judge. The manuscript asks which pairs to compare, and gives exact answers:

- **An exact error identity.** Every pairwise table splits into a level, a distinctiveness for each individual, and relational structure. The error of any subset-of-pairs estimate is the sum of a distinctiveness term and a relational term. The first term vanishes for every population if and only if each individual is compared equally often.
- **An optimal budget without adaptivity.** Balanced designs with random labels are unbiased, and their variance has a closed form that depends on relational structure alone. Written in these coordinates, a classical consequence of the triangle inequality becomes a sharp bound: the relational energy is at most (n−2)μ²/n for every pseudometric, with equality for a balanced cut metric. Hence a balanced design with m comparisons has relative root-mean-square error at most m^(−1/2) for every pseudometric, and a non-adaptive, equally weighted design reaches relative error ε from O(n + ε⁻²) comparisons, which matches our lower bounds. At equal cost, the adaptive scheme of Chechik, Cohen and Kaplan (2015) is 1.4–10 times less accurate in our experiments. The efficiency gain over uniform pair sampling is exact and comes with an explicit threshold. Uniform pair sampling can need about n/2 times more comparisons. The same comparisons also give each individual's distinctiveness and an unbiased error bar.
- **Fixed designs and control.** We prove a spectral bound for fixed expander designs and a per-step tracking bound for closed-loop diversity control.
- **Experiments.** Wherever a closed form exists (balanced, uniform, and Bernoulli designs), measured errors match it to within Monte Carlo error; on real populations, balanced designs typically need about half the comparisons of uniform pair sampling, and up to seven times fewer. The experiments cover multi-agent teams trained with independent PPO, classifier populations, language-model outputs scored by a cross-encoder and by frontier-model judges (Claude Opus 5.5 and GPT-5), and closed-loop diversity control. With Claude as the judge on 64 stories, all pairs cost about $3.93, whereas a balanced design with 128 calls costs about $0.25 and has 1.05% relative error. Inside DiCo diversity control with 50 agents, a balanced design showed no detectable loss of tracking or reward relative to all-pairs control. A NumPy implementation is released.

**What is classical.** The manuscript credits the following as classical:

- the decomposition itself (a two-way analysis of variance and the finite-population Hoeffding decomposition); its relational part is the U-centered distance matrix (Székely and Rizzo 2014);
- the inequality behind the metric bound, which for Euclidean data appears in the proof that the distance standard deviation is at most Gini's mean difference (Edelmann, Richards and Vogel 2020);
- the survey-sampling counterparts: the regression decomposition (Särndal, Swensson and Wretman 1992), balanced sampling and its variance approximation (Deville and Tillé 2004, 2005), and random-group variance estimation (Wolter 2007);
- the quadratic-assignment (Mantel) moments of relabeled designs;
- circulant and round-robin equireplicate designs (Kempthorne and Curnow 1961; Lee 1982; Miglioli and Awan 2025);
- sampling estimates of the distinctiveness profile (Cochran);
- the lower-bound constructions and the analysis of uniform pair sampling (Barhum, Goldreich and Shraibman 2007);
- the O(n + ε⁻²) rate itself, which was first reached with adaptive, non-uniform weights (Chechik, Cohen and Kaplan 2015).

The contributions listed above go beyond these.

**1. Previous publications and overlap.** No part of this work has been published in a journal or in conference proceedings. An earlier preprint, *Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent Reinforcement Learning* (arXiv:2605.05020), studied edge sampling and expander aggregation for SND only. The present manuscript supersedes it. The anatomy of pairwise diversity, the error identity, the balanced-design theory with exact variances and optimal budgets, the metric bound and its consequences for designs, the lower bounds, the language-model, classifier, and image experiments, and the balanced-design DiCo runs are all new. The preprint's forwarding-index analysis is replaced by a new spectral theorem. The manuscript cites the preprint, and I can upload it as a supplementary file if that would help.

**2. Authorship.** I am the sole author, and I agree to JMLR reviewing the manuscript. It is not under review at any other journal or conference.

**3. Conflicts of interest.** I have no conflicts of interest (collaboration in the past three years, advising relationship, or personal relationship) with any JMLR action editor, including the editors suggested below, or with the suggested reviewers.

**4. Suggested action editors** (all listed on the JMLR editorial board; none at Carnegie Mellon): Gábor Lugosi, Bharath Sriperumbudur, Aryeh Kontorovich, Michael Mahoney, Csaba Szepesvári.

**5. Suggested reviewers:** Aurélien Bellet (Inria; incomplete U-statistics), Xiaohui Chen (University of Southern California; randomized incomplete U-statistics), Kengo Kato (Cornell University; incomplete U-statistics), Edith Cohen (Google Research and Tel Aviv University; average-distance estimation), Amanda Prorok (University of Cambridge; behavioral diversity in multi-agent learning).

**6. Keywords:** diversity, incomplete U-statistics, experimental design, multi-agent reinforcement learning, evaluation of generative models.

**7. Length.** The article is 50 pages, including the proofs of all main results and the references, so it is within the 50-page threshold. The 17-page supplement holds material that reviewers may consult but that is not needed to follow the article: experimental details for reproducibility, additional figures and tables, exact formulas for separable dissimilarities, the standard lower-bound constructions, two short proofs, and five remarks.

**Funding and competing interests.** This work received no external funding, and I have no financial relationships that could be perceived to influence it.

**Code and data.** All code, distance tables, and logs are at https://github.com/shawnray-research/Graph-SND, and every experiment can be reproduced from them.

Sincerely,
Shawn Ray
