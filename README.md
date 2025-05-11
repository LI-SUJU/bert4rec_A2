<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">BERT4REC_A2</h1></p>
<p align="center">
	<em><code>❯ REPLACE-ME</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/LI-SUJU/bert4rec_A2?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/LI-SUJU/bert4rec_A2?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/LI-SUJU/bert4rec_A2?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/LI-SUJU/bert4rec_A2?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)

- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

BERT4Rec mitigates the limitations of traditional sequential models by employing Transformer encoders with a Masked Language Modeling (MLM) objective. In this project, we implement BERT4Rec using PyTorch and train it on the MovieLens 1M dataset. We construct fixed-length user sequences from positive interactions (ratings $\geq$ 4). The model uses item and positional embeddings, followed by a multi-layer Transformer encoder. Prediction is performed via a softmax over the item vocabulary. Training minimizes cross-entropy loss using the Adam optimizer with learning rate scheduling, and early stopping based on NDCG@10 in the validation set. Evaluation metrics include Recall@10 and NDCG@10 using a re-ranking strategy with 99 negative samples per ground truth item.

---

##  Features

Bert4Rec on the MovieLens 1M dataset
The dataset is available here: https://grouplens.org/datasets/movielens/

---

##  Project Structure

```sh
└── bert4rec_A2/
    ├── BERT4Rec.py           # Main training and evaluation script
    └── data/                 # Folder containing the MovieLens 1M dataset
        ├── ratings.dat       # User-Movie rating records
        ├── movies.dat        # Movie metadata (e.g., titles, genres)
        ├── users.dat         # User metadata (e.g., age, gender, occupation)
        └── ...               # Other relevant files if needed
```



---
##  Getting Started

###  Prerequisites

Before getting started with bert4rec_A2, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


###  Installation

Install bert4rec_A2 using one of the following methods:

**Build from source:**

1. Clone the bert4rec_A2 repository:
```sh
❯ git clone https://github.com/LI-SUJU/bert4rec_A2
```

2. Navigate to the project directory:
```sh
❯ cd bert4rec_A2
```




###  Usage
Run bert4rec_A2 using the following command:
```sh
python BERT4Rec.py
```
This will execute the full pipeline:

Pre-training with the Cloze-style objective using random masking

Fine-tuning for next-item prediction (masking only the final item)

Evaluation using Recall@10 and NDCG@10 with a re-ranking strategy (99 negative samples + 1 positive)
You can modify model architecture, training parameters, or masking strategy by editing BERT4Rec.py

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/LI-SUJU/bert4rec_A2/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/LI-SUJU/bert4rec_A2/issues)**: Submit bugs found or log feature requests for the `bert4rec_A2` project.
- **💡 [Submit Pull Requests](https://github.com/LI-SUJU/bert4rec_A2/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/LI-SUJU/bert4rec_A2
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/LI-SUJU/bert4rec_A2/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=LI-SUJU/bert4rec_A2">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
