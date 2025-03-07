# Adversarial Network Embedding with Bootstrapped Representations for Sparse Networks
[![DOI](https://img.shields.io/static/v1?label=DOI&message=10.1007/s10489-025-06343-2&color=green&logo=doi)](https://doi.org/10.1007/s10489-025-06343-2)

## Abstract

![0](https://github.com/user-attachments/assets/c6a51c8f-6724-4406-bbf6-d4561f6d6ec1)

**Fig. 1**: The overall framework of ANEBR. Firstly, network augmentation is performed for positive sampling to extract and filter useful information from the sparse network. ANEBR augments the adjacency matrix $\mathbf{A}$ by extracting latent high-order information and refining it with the $\alpha$-entmax to produce an augmented adjacency matrix $\tilde{\mathbf{A}}$. Secondly, the bootstrapped embedding representation is robustly derived to avoid negative sampling. The target network uses an encoder $\mathcal{E}_\xi$ to iteratively bootstrap embeddings $\tilde{\mathbf{H}}$ from $\tilde{\mathbf{A}}$ as targets, updating parameters online based on $\mathcal{E}_\theta$ in the online network to stabilize the process. The online network, in turn, uses $\mathcal{E}_\theta$ and $\mathcal{D}_\theta$ to learn the embedding $\mathbf{H}$ and the reconstruction $\hat{\mathbf{A}}$. The discriminator $\textit{D}_\varphi$ differentiates embeddings from both networks, guiding $\mathcal{E}_\theta$ to generate embeddings $\mathbf{H}$ that closely match $\tilde{\mathbf{H}}$ through adversarial interaction. Thirdly, besides learning rich embedding, end-to-end accurate reconstruction is realized. ANEBR seeks low-rank reconstruction errors, i.e., minimizing the sum of singular values in the reconstruction space by the nuclear norm for accurate reconstruction.

## Dependencies

```
entmax==1.3
matplotlib==3.10.0
networkx==3.4.2
numpy==2.2.2
scikit_learn==1.6.1
scipy==1.15.1
torch==2.5.1+cu124
torch_geometric==2.6.1
```

## Dataset

In `./data/`, the `Adjnoun`,`20-Newsgroups`, `PPI`, `Wikipedia` and `BlogCatalog` datasets are provided, along with the corresponding processed versions for link prediction.

## Training

Detailed training results and configs for network reconstruction, node classification, link prediction and network visualization are provided in `./Results.ipynb`.

Besides, it is also easy to run `./reconstruction.py` directly to perform network reconstruction, as is the case for the other graph learning tasks.

## Citation

If you find the code useful for your research, we kindly request to consider citing our work:

```
@article{wu2025adversarial,
  title={Adversarial network embedding with bootstrapped representations for sparse networks},
  author={Wu, Zelong and Wang, Yidan and Lin, Guoliang and Liu, Junlong},
  journal={Applied Intelligence},
  volume={55},
  number={6},
  pages={498},
  year={2025},
  publisher={Springer}
}
```



