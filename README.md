Federated Learning Project

The main entry point of this Federated Learning project is the main.ipynb Colab notebook, which serves as the central orchestrator for running experiments and managing the overall pipeline. 

It integrates functionalities from various Python scripts such as data_utils.py, centralized_baseline.py, fl_model_utils.py, secure_aggregation.py, and sparse_utils.py, modesaver.py which handle different components including centralized training, federated model updates, privacy-preserving aggregation, and sparsity and saving models results. 

The project is structured with supporting folders like centralized/, fed_learning/, fed_baseline/, and fed_sparse_models/, sparse_finetune/ and report/ that contain modular implementations of federated training strategies and baseline comparisons and the report of this project. 

After executing experiments through the notebook or scripts, results such as training curves, evaluation metrics, and sparsity plots are saved to the plots/ folder. Final model checkpoints and experiment outputs are stored in 
final_test_results/.

