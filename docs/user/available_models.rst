================
Available models
================

The following table summarises the supported hyperbox-based learning algorithms in this toolbox.

.. list-table::
   :widths: 20 10 10 10 30 10 10
   :align: left
   :header-rows: 1

   * - Model
     - Feature type 
     - Model type
     - Learning type 
     - Implementation 
     - Example 
     - References 
   * - EIOL-GFMM
     - Mixed
     - Single 
     - Instance-incremental 
     - `ExtendedImprovedOnlineGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/mixed_data/eiol_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/mixed_data/eiol_gfmm_general_use.ipynb>`_
     - [1]_
   * - Freq-Cat-Onln-GFMM 
     - Mixed 
     - Single 
     - Batch-incremental 
     - `FreqCatOnlineGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/mixed_data/freq_cat_onln_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/mixed_data/freq_cat_onln_gfmm_general_use.ipynb>`_
     - [2]_
   * - OneHot-Onln-GFMM 
     - Mixed 
     - Single 
     - Batch-incremental 
     - `OneHotOnlineGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/mixed_data/onehot_onln_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/mixed_data/onehot_onln_gfmm_general_use.ipynb>`_
     - [2]_
   * - Onln-GFMM 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `OnlineGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/incremental_learner/onln_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/incremental_learner/onln_gfmm_general_use.ipynb>`_
     - [3]_, [4]_
   * - IOL-GFMM 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `ImprovedOnlineGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/incremental_learner/iol_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/incremental_learner/iol_gfmm_general_use.ipynb>`_
     - [5]_, [4]_
   * - FMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `FMNNClassifier <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/incremental_learner/fmnn.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/incremental_learner/fmnn_general_use.ipynb>`_
     - [6]_
   * - EFMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `EFMNNClassifier <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/incremental_learner/efmnn.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/incremental_learner/efmnn_general_use.ipynb>`_
     - [7]_ 
   * - KNEFMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `KNEFMNNClassifier <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/incremental_learner/knefmnn.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/incremental_learner/knefmnn_general_use.ipynb>`_
     - [8]_ 
   * - RFMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `RFMNNClassifier <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/incremental_learner/rfmnn.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/incremental_learner/rfmnn_general_use.ipynb>`_
     - [9]_ 
   * - AGGLO-SM 
     - Continuous 
     - Single 
     - Batch 
     - `AgglomerativeLearningGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/batch_learner/agglo_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/batch_learner/agglo_gfmm_general_use.ipynb>`_
     - [10]_, [4]_
   * - AGGLO-2
     - Continuous 
     - Single 
     - Batch
     - `AccelAgglomerativeLearningGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/batch_learner/accel_agglo_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/batch_learner/accel_agglo_gfmm_general_use.ipynb>`_
     - [10]_, [4]_
   * - MRHGRC
     - Continuous 
     - Granularity 
     - Multi-Granular learning 
     - `MultiGranularGFMM <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/multigranular_learner/multi_resolution_gfmm.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/multigranular_learner/multi_resolution_gfmm_general_use.ipynb>`_
     - [11]_ 
   * - Decision-level Bagging of hyperbox-based learners
     - Continuous 
     - Combination 
     - Ensemble 
     - `DecisionCombinationBagging <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/ensemble_learner/decision_comb_bagging.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/ensemble_learner/decision_comb_bagging_general_use.ipynb>`_
     - [12]_
   * - Decision-level Bagging of hyperbox-based learners with hyper-parameter optimisation
     - Continuous
     - Combination 
     - Ensemble 
     - `DecisionCombinationCrossValBagging <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/ensemble_learner/decision_comb_cross_val_bagging.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/ensemble_learner/decision_comb_cross_val_bagging_general_use.ipynb>`_
     -  
   * - Model-level Bagging of hyperbox-based learners
     - Continuous 
     - Combination 
     - Ensemble 
     - `ModelCombinationBagging <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/ensemble_learner/model_comb_bagging.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/ensemble_learner/model_comb_bagging_general_use.ipynb>`_
     - [12]_
   * - Model-level Bagging of hyperbox-based learners with hyper-parameter optimisation 
     - Continuous 
     - Combination 
     - Ensemble 
     - `ModelCombinationCrossValBagging <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/ensemble_learner/model_comb_cross_val_bagging.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/ensemble_learner/model_comb_cross_val_bagging_general_use.ipynb>`_
     -   
   * - Random hyperboxes 
     - Continuous 
     - Combination 
     - Ensemble 
     - `RandomHyperboxesClassifier <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/ensemble_learner/random_hyperboxes.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/ensemble_learner/random_hyperboxes_general_use.ipynb>`_
     - [13]_
   * - Random hyperboxes with hyper-parameter optimisation for base learners 
     - Continuous 
     - Combination 
     - Ensemble 
     - `CrossValRandomHyperboxesClassifier <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/hbbrain/numerical_data/ensemble_learner/cross_val_random_hyperboxes.py>`_
     - `Notebook <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/examples/numerical_data/ensemble_learner/cross_val_random_hyperboxes_general_use.ipynb>`_
     -  

References
~~~~~~~~~~

.. [1] T. T. Khuat and B. Gabrys "`An Online Learning Algorithm for a Neuro-Fuzzy Classifier with Mixed-Attribute Data <https://arxiv.org/abs/2009.14670>`_", ArXiv preprint, arXiv:2009.14670, 2020.
.. [2] T. T. Khuat and B. Gabrys "`An in-depth comparison of methods handling mixed-attribute data for general fuzzy min-max neural network <https://doi.org/10.1016/j.neucom.2021.08.083>`_", Neurocomputing, vol 464, pp. 175-202, 2021.
.. [3] B. Gabrys and A. Bargiela, "`General fuzzy min-max neural network for clustering and classification <https://doi.org/10.1109/72.846747>`_", IEEE Transactions on Neural Networks, vol. 11, no. 3, pp. 769-783, 2000.
.. [4] T. T. Khuat and B. Gabrys, "`Accelerated learning algorithms of general fuzzy min-max neural network using a novel hyperbox selection rule <https://doi.org/10.1016/j.ins.2020.08.046>`_", Information Sciences, vol. 547, pp. 887-909, 2021.
.. [5] T. T. Khuat, F. Chen, and B. Gabrys, "`An improved online learning algorithm for general fuzzy min-max neural network <https://doi.org/10.1109/IJCNN48605.2020.9207534>`_", in Proceedings of the International Joint Conference on Neural Networks (IJCNN), pp. 1-9, 2020.
.. [6] P. Simpson, "`Fuzzy min—max neural networks—Part 1: Classiﬁcation <https://doi.org/10.1109/72.159066>`_", IEEE Transactions on Neural Networks, vol. 3, no. 5, pp. 776-786, 1992.
.. [7] M. Mohammed and C. P. Lim, "`An enhanced fuzzy min-max neural network for pattern classification <https://doi.org/10.1109/TNNLS.2014.2315214>`_", IEEE Transactions on Neural Networks and Learning Systems, vol. 26, no. 3, pp. 417-429, 2014.
.. [8] M. Mohammed and C. P. Lim, "`Improving the Fuzzy Min-Max neural network with a k-nearest hyperbox expansion rule for pattern classification <https://doi.org/10.1016/j.asoc.2016.12.001>`_", Applied Soft Computing, vol. 52, pp. 135-145, 2017.
.. [9] O. N. Al-Sayaydeh, M. F. Mohammed, E. Alhroob, H. Tao, and C. P. Lim, "`A refined fuzzy min-max neural network with new learning procedures for pattern classification <https://doi.org/10.1109/TFUZZ.2019.2939975>`_", IEEE Transactions on Fuzzy Systems, vol. 28, no. 10, pp. 2480-2494, 2019.
.. [10] B. Gabrys, "`Agglomerative learning algorithms for general fuzzy min-max neural network <https://link.springer.com/article/10.1023/A:1016315401940>`_", Journal of VLSI Signal Processing Systems for Signal, Image and Video Technology, vol. 32, no. 1, pp. 67-82, 2002.
.. [11] T.T. Khuat, F. Chen, and B. Gabrys, "`An Effective Multiresolution Hierarchical Granular Representation Based Classifier Using General Fuzzy Min-Max Neural Network <https://doi.org/10.1109/TFUZZ.2019.2956917>`_", IEEE Transactions on Fuzzy Systems, vol. 29, no. 2, pp. 427-441, 2021.
.. [12] B. Gabrys, "`Combining neuro-fuzzy classifiers for improved generalisation and reliability <https://doi.org/10.1109/IJCNN.2002.1007519>`_", in Proceedings of the 2002 International Joint Conference on Neural Networks, vol. 3, pp. 2410-2415, 2002.
.. [13] T. T. Khuat and B. Gabrys, "`Random Hyperboxes <https://doi.org/10.1109/TNNLS.2021.3104896>`_", IEEE Transactions on Neural Networks and Learning Systems, 2021.
