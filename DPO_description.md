I want to apply DPO to a topic model. The details can be found in the files **NTM_DPO.pdf** and **main.tex** (where *main.tex* is the LaTeX code used to generate *NTM_DPO.pdf*).

The goal is to obtain the best possible **top 15 words** for each topic. However, there are several important notes:

- The section **“Collecting Preferences with an LLM”** is somewhat unnecessary. The initial design was rather naive and incomplete.
- Specifically, the construction of the preference dataset will proceed as follows:
  - After approximately **E epochs** of training the base model (where *E* is a hyperparameter), we save the top words of each topic (this is already implemented in the old code). In addition, we save the current model as a `.pth` file in the model’s output directory.
  - Then, the LLM needs to perform several tasks. First, call the LLM to **score each topic** based on its top 15 words:
    - **Score 1**: the top words are not related to each other at all
    - **Score 2**: the top words are somewhat related, but many bad words are mixed in
    - **Score 3**: the top words are strongly and coherently related
  - For each topic, the LLM needs to determine **a descriptive phrase** representing the topic, based on the dominant meaning within the first **top 20 words**.
    - This is relatively easy for topics with scores 2 or 3.
    - For topics with score 1, determine the description based on whichever meaning dominates the top words, prioritizing higher-ranked words.
  - Instruct the LLM to make the identified topics **as diverse as possible**, avoiding excessive overlap in topic content (ensuring topic diversity).
    - To achieve this, whenever calling the LLM to determine descriptions for **score-1 topics**, include the list of topic descriptions that have already been identified.
    - For topics with scores 2 or 3, simply call the LLM to generate the topic description.
  - Save the results—including **score, topic description, and top words**—into a **JSONL file**, where each line has the following example format:

```json
{"k": 0, "main meaning": "...", "llm_score": "...", "top_words": [{"scsi": 3938}, {"ide": 2129}, {"controller": 979}, {"drives": 1356}, {"bios": 467}, {"jumper": 2390}, {"isa": 2316}, {"drive": 1352}, {"floppy": 1745}, {"slave": 4119}, ...]}
```

Here, **k** is the topic index. Each top word is stored as `"word_name": vocab_index` (indices start from 0).  
For **top-10, top-15, top-20, and top-25**, save a separate JSONL file in the same format.

  - Next, use **sentence embeddings** to select **5 additional words** (outside the top 25) from the vocabulary that are most semantically similar to the topic description (embed the topic description, embed all words in the vocabulary, then compute similarity).
  - From the resulting **30 words** (the original top 25 plus 5 additional words), instruct the LLM to identify:
    - **win words**: semantically closest to the topic
    - **loose words**: semantically unrelated to the topic
  - To do this, provide the LLM with:
    - The topic index
    - The topic description phrase
    - Two word lists:
      - A list of the **original top 15 words**, formatted as:

```json
[{"scsi": 3938}, {"ide": 2129}, {"controller": 979}, {"drives": 1356}, {"bios": 467}, {"jumper": 2390}, {"isa": 2316}, {"drive": 1352}, {"floppy": 1745}, {"slave": 4119}, ...]
```

      - A list of **15 words outside the top 15**, consisting of words ranked 11–25 plus the 5 embedding-selected words, using the same key–value format.
      - Here, **keys** are words and **values** are vocabulary indices starting from 0.
  - Then, prompt the LLM to identify **win–loose pairs**, with the following rules:
    - The number of win and loose words can vary by topic; the LLM does not need to be overly strict.
    - Any **bad words within the top 15** must be included in the **loose** list.
    - Any **good words outside the top 15** must be included in the **win** list.
  - Prompt the LLM to return the **preference dataset** in the following format:

```json
{
  "k": <topic_index>,
  "w_win_indices": [<indices of words related to the main topic>],
  "w_loose_indices": [<indices of words not related to the main topic>]
}
```

- After obtaining the preference dataset, construct the **DPO loss**. Then, for the final **(500 − E) epochs**, train the model using the original loss plus the DPO loss (with a tunable coefficient to control the influence of the DPO loss).
- Implement the DPO loss so that it can be **computed in parallel and run efficiently**. The DPO loss formula is provided in *NTM_DPO.pdf* and *main.tex*. You may modify the loss formulation as long as **win words are ranked higher than loose words**.
- I want to add an additional setting for the DPO loss: topics that are already good should **not** use DPO loss, while weak topics should.
  - To do this, compute **topic coherence (CV)** for all topics, and apply DPO only to topics whose CV is **below the average CV**.
  - The second option applies the DPO loss only to topics scored 1, or 1–2 by LLM.
- For the LLM, please use **gpt-4o**. The **OPENAI_API_KEY** is stored in the `.env` file.
  - Use **function calling** to ensure the LLM outputs strictly follow the required format.
- Use ECRTM as the base model. Do not change the base model too much. I think all new things can be added in new code files.
