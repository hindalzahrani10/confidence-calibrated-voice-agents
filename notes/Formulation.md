# Formulation: Noise-Marginalized Value of Information (NM-VoI)

## 1. Differentiation from Classic SDS Confirmation Strategies
Classic spoken dialogue systems have long utilized Automatic Speech Recognition (ASR) confidence scores to manage uncertainty. As Williams and Young (2007) note in their foundational work on Partially Observable Markov Decision Processes (POMDPs), dialogue managers must maintain a distribution over user intents to execute "POMDP dialogue management" effectively under noise. Similarly, Bohus and Rudnicky established robust "error handling and confirmation strategies" that rely on explicit and implicit confirmation thresholds to prevent conversational breakdowns.

However, these legacy systems are structurally constrained. They map ASR confidence to hand-designed belief states and rigid, ontology-bound slot-filling mechanisms. The proposed NM-VoI framework adapts this POMDP-inspired uncertainty management for modern, open-domain Large Language Model (LLM) agents. Rather than populating predefined semantic slots, our approach dynamically calculates the Expected Value of Perfect Information (EVPI) over generative, free-form latent intents (Dong et al., 2026). Furthermore, by expanding the action space, the agent autonomously scales the cognitive and acoustic complexity of its questions—transitioning from open-ended queries to constrained choices based on acoustic viability—creating a noise-conditioned policy absent in current LLM frameworks.

## 2. Action Space and Utilities
Let $x$ be the user's initial underspecified or noisy spoken utterance, and $y$ be the user's true underlying intent. At any dialogue turn, the agent must select an action $a \in A$, where the expanded action space is:

$A = \{a_{ans}, a_{open}, a_{closed}, a_{conf}, a_{abs}\}$

*   $a_{ans}$: Direct answer (assumes intent is known).
*   $a_{open}$: Ask an open-ended clarification question.
*   $a_{closed}$: Ask a closed/multiple-choice question.
*   $a_{conf}$: Ask a yes/no confirmation.
*   $a_{abs}$: Abstain from the task entirely.

**Utility Function**
The utility of taking action $a$ given the true intent $y$ is defined as:
$$U(a, y) = R(a, y) - C(a)$$

*   $R(a, y)$ is the task success reward ($+1$ for correct task completion, $-1$ for failure/hallucination, $0$ for safe abstention).
*   $C(a)$ is the interaction cost associated with the action. Crucially, communication costs vary by cognitive load and question type: $C(a_{open}) > C(a_{closed}) > C(a_{conf})$.

## 3. Standard Expected Value of Perfect Information (EVPI)
In current text-based clarification frameworks, agents decide whether to ask a question by evaluating the "Value of Information" (Dong et al., 2026). The standard expected utility of a direct answer is:
$$EU(a_{ans}) = \mathbb{E}_{y \sim P(y|x)} [U(a_{ans}, y)]$$

The expected utility of asking a clarification question $a_q \in \{a_{open}, a_{closed}, a_{conf}\}$ assumes the user will reply with their exact intent $y$, perfectly transcribed:
$$EU_{standard\_ask}(a_q) = \mathbb{E}_{y \sim P(y|x)} \left[ \max_{a'} U(a', x \oplus a_q \oplus y) \right] - C(a_q)$$

**The Flaw:** Standard EVPI assumes a perfect observation channel where $p(\hat{y} | y) = 1$. It entirely fails to account for the probability that the user's reply $y$ will be corrupted by acoustic noise during transcription.

## 4. Noise-Marginalized Value of Information (NM-VoI)

### 4.1 Noisy Observation Channel & Empirical Lookup
We introduce a noisy observation channel where the user's spoken clarification $y$ is transcribed as $\hat{y}$ via an ASR model. The observation probability $p(\hat{y} | y, \text{SNR}, a_q)$ is conditioned on the background Signal-to-Noise Ratio (SNR) and the structural type of question asked $a_q \in \{a_{open}, a_{closed}, a_{conf}\}$. 

In our implementation, $p(\hat{y} | y, \text{SNR}, a_q)$ is empirically estimated by benchmarking sample response recognition rates across discrete SNR buckets (Clean, 20 dB, 10 dB, 5 dB) for each question modality.

### 4.2 Belief Update and One-Step Lookahead Policy
We assume a **one-step lookahead decision horizon**. Let $b_0(y) = P(y|x)$ be the agent's prior belief over the true intent $y$ given the initial prompt $x$. 

When the agent asks clarification question $a_q$, the user utters their true intent $y$, which is transcribed as noisy observation $\hat{y} \sim p(\hat{y} | y, \text{SNR}, a_q)$. Upon observing $\hat{y}$, the agent updates its belief via Bayesian inference / prompt context concatenation:

$$b_1(y \mid \hat{y}) = P(y \mid x, a_q, \hat{y})$$

Under this updated belief $b_1$, the agent selects the candidate action $a^*(\hat{y})$ that maximizes expected task reward:

$$a^*(\hat{y}) = \arg\max_{a'} \mathbb{E}_{y \sim b_1(y \mid \hat{y})} [R(a', y)]$$

Crucially, the true utility received is evaluated against the *actual* underlying intent $y$. The Noise-Marginalized Expected Utility of asking question $a_q$ is therefore:

$$EU_{\text{NM-VoI}}(a_q) = \mathbb{E}_{y \sim b_0(y)} \left[ \mathbb{E}_{\hat{y} \sim p(\hat{y} | y, \text{SNR}, a_q)} \left[ R(a^*(\hat{y}), y) \right] \right] - C(a_q)$$

**Why this models failure:** If the ASR channel is noisy, the transcript $\hat{y}$ will be corrupted. This distorts the agent's updated belief $b_1(y \mid \hat{y})$, causing it to select an incorrect downstream action \(a^*(\hat{y})\) and incur a penalty \(R(a^*(\hat{y}), y) = -1\). The outer expectation automatically penalizes $EU_{\text{NM-VoI}}(a_q)$ in low SNR environments.

## 5. References
*   Bohus, D., & Rudnicky, A. I. (2005). *Sorry, I didn't catch that! - An investigation of non-understanding errors and recovery strategies.* SIGdial.
*   Dong, Y., et al. (2026). *Value of Information: A Framework for Human–Agent Communication.* ACL.
*   Williams, J. D., & Young, S. (2007). *Partially observable Markov decision processes for spoken dialog systems.* Computer Speech & Language, 21(2), 393-422.
