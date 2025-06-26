This report investigates the use of federated learning (FL) for image classification on the CIFAR
100 dataset using the DINO ViT-S/16 transformer model. We evaluate several training strategies, 
including centralized training, standard Federated Averaging (FedAvg), and a sparse fine-tuning 
variant of FedAvg designed to reduce communication costs. The sparse variant leverages 
sensitivity-based gradient masking to update only the most influential parameters during client
side training. To better understand performance under realistic settings, we simulate both IID and 
non-IID client distributions by varying the number of classes per client (Nc ∈ \{1, 5, 10, 50\}) and 
local training steps (J ∈ \{4, 8, 16\}). Our experiments show that sparse fine-tuning achieves 
competitive accuracy while significantly reducing the update size. We also demonstrate a secure 
aggregation mechanism to protect client updates, and we compare its output with standard 
averaging. Overall, the study highlights how combining sparse updates with privacy-preserving 
aggregation can make FL more efficient and practical for modern vision models.
