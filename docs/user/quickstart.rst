==========
Quickstart
==========

Training a model
----------------

Simply use an estimator by initialising, fitting and predicting:

.. code:: python

   from sklearn.datasets import load_iris
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
   # Load dataset
   X, y = load_iris(return_X_y=True)
   # Normalise features into the range of [0, 1] because hyperbox-based models only work in a unit range
   scaler = MinMaxScaler()
   scaler.fit(X)
   X = scaler.transform(X)
   # Split data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   # Training a model
   clf = OnlineGFMM(theta=0.1).fit(X_train, y_train)
   # Make prediction
   y_pred = clf.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   print(f'Accuracy = {acc * 100: .2f}%')


In an sklearn Pipeline
----------------------

Using hyperbox-based estimators in a `sklearn Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_:

.. code:: python

   from sklearn.datasets import load_iris
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.pipeline import Pipeline
   from sklearn.model_selection import train_test_split
   from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM

   # Load dataset
   X, y = load_iris(return_X_y=True)
   # Split data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   # Create a GFMM model
   onln_gfmm_clf = OnlineGFMM(theta=0.1)
   # Create a pipeline
   pipe = Pipeline([
      ('scaler', MinMaxScaler()),
      ('onln_gfmm', onln_gfmm_clf)
   ])
   # Training
   pipe.fit(X_train, y_train)
   # Make prediction
   acc = pipe.score(X_test, y_test)
   print(f'Testing accuracy = {acc * 100: .2f}%')

Hyper-parameter search
----------------------

This example shows how to use hyperbox-based models with `sklearn random search <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_:

.. code:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.metrics import accuracy_score
   from sklearn.model_selection import RandomizedSearchCV
   from sklearn.model_selection import train_test_split
   from hbbrain.numerical_data.ensemble_learner.random_hyperboxes import RandomHyperboxesClassifier
   from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM

   # Load dataset
   X, y = load_breast_cancer(return_X_y=True)
   # Normalise features into the range of [0, 1] because hyperbox-based models only work in a unit range
   scaler = MinMaxScaler()
   X = scaler.fit_transform(X)
   # Split data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   # Initialise search ranges for hyper-parameters
   parameters = {'n_estimators': [20, 30, 50, 100, 200, 500], 
              'max_samples': [0.2, 0.3, 0.4, 0.5, 0.6],
              'max_features' : [0.2, 0.3, 0.4, 0.5, 0.6],
              'class_balanced' : [True, False],
              'feature_balanced' : [True, False],
              'n_jobs' : [4],
              'random_state' : [0],
              'base_estimator__theta' : np.arange(0.05, 0.61, 0.05),
              'base_estimator__gamma' : [0.5, 1, 2, 4, 8, 16]}
   # Init base learner. This example uses the original online learning algorithm to train a GFMM classifier
   base_estimator = OnlineGFMM()
   # Using random search with only 40 random combinations of parameters
   random_hyperboxes_clf = RandomHyperboxesClassifier(base_estimator=base_estimator)
   clf_rd_search = RandomizedSearchCV(random_hyperboxes_clf, parameters, n_iter=40, cv=5, random_state=0)
   # Fit model
   clf_rd_search.fit(X_train, y_train)
   # Print out best scores and hyper-parameters
   print("Best average score = ", clf_rd_search.best_score_)
   print("Best params: ", clf_rd_search.best_params_)
   # Using the best model to make prediction
   best_gfmm_rd_search = clf_rd_search.best_estimator_
   y_pred_rd_search = best_gfmm_rd_search.predict(X_test)
   acc_rd_search = accuracy_score(y_test, y_pred_rd_search)
   print(f'Accuracy (random-search) = {acc_rd_search * 100: .2f}%')
