---
header-includes:
  - \hypersetup{colorlinks=true,
            allbordercolors={0 0 0},
            pdfborderstyle={/S/U/W 1}}
---

_CS-GY 9223, Fall 2020_

# Misc topics (II)

Topics in this lecture: DL and ethical concerns

* Privacy
* Fairness
* Security

Let us wind up with a few emerging topics that are yet to make their way into machine learning textbooks. Many of these topics are critical for practitioners to understand and assess, particularly as machine learning graduates from being a "niche" area only studied by academics to becoming more widely deployed in the real world. Moreover, most of these topics are actively being researched -- and the ML community is still grappling with "what" the right answers are.

## Fairness

While the ML techniques that we have discussed can solve rather interesting real-world problems, they bring along with them certain unique societal challenges.

Consider the basic problem -- supervised learning -- that we have been discussing for the majority of the course. We have a dataset $X$, each data point composed of $d$ features. We are trying to predict a label $y$. This, for example, could be the case in a banking application where the features are various attributes of each person (age, gender, zip code, race, FICO credit score, requested loan amount, \ldots) and the goal is to decide whether to give the person the loan or not. We could consider training a classifier (linear, logistic, etc.) based on all the available features to make this decision.

But suppose also that a *subset* of the features are *sensitive* or *protected*. In the above example, attributes such as age, gender, and race are protected, and decisions cannot be biased towards certain demographics. How do we deal with this?

As a first attempt, we can remove (all) sensitive attributes so that our classifier never looks at them. This is called "fairness through unawareness".

However, this may not be enough! For example, race/age may be strongly correlated with zip code (for example, it could be a locality with lots of senior homes, or with folks from a specific community). So even if we do not look at a particular attribute, due to the strong statistical correlations between attributes our decisions may be end up biased anyway!

As a second attempt, we can aim for *statistical parity*. This means that if we have two groups of people $A$ and $B$, the same percentage get the loans. This can be formulated in terms of statistical independence and conditional probabilities:  
$$
\text{Prob}(\text{Loan = True} | \text{group =} A) = \text{Prob}(\text{Loan = True} | \text{group =} B).
$$
Mathematically, this means that the $\text{Loan}$ variable is statistically independent of the conditional variable encoding the group index.

This seems intuitive, and many methods for obtaining statistically fair classifiers exist. Unfortunately, statistical parity can be problematic too. (What if $80 \%$ of group A are capable of actually repaying the loan but only $60 \%$ of group B can?) More generally, there are fundamental  

There are other definitions of fairness: equal odds, individual fairness, counterfactual fairness, etc -- but no one definition has emerged as consensus as of yet.

## Adversarial Robustness

(see posted demo notebook for details)

## Deep Learning and Ethics

Let us conclude with a brief discussion about ML and ethics. Of course, there are no definitive answers to any of these points yet, but these are some of the questions that one should keep in mind while working on practical ML/AI systems. This is only an incomplete list:

* Transparency: Can the designer explain the decisions made by an ML system?

* Accountability: Can the designer take ownership of the decisions made by an ML system?

* Systemic bias: Does the ML system suffer from implicit and/or systematic bias?

* Stability: Is the ML system robust to catastrophic situations caused by natural or adversarial means?

* Pedagogy: Are the benefits and drawbacks of the ML system well communicated to the public (or anyone who may be impacted by the system)?
